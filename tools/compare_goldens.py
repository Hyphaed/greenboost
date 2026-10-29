#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score a measurement set against pinned golden values.

Ported from Megatron-LM's `update-golden-values` skill (crosswalk P5-10). The
GitHub-Actions plumbing around it does not transfer -- GreenBoost's numbers come
off this box, not off CI artifacts -- but the scoring idea does, and it is the
part that matters: a per-metric

    rel_diff = (new - old) / old

so a change is read as "decode got 4% slower" rather than eyeballed out of two
tables. Numbers that live only in prose cannot regress loudly, and every
performance figure this repo trusts currently lives in
docs/performance-casebook.md as prose.

Usage:

    python3 tools/compare_goldens.py tests/golden/nvfp4_gemm.json measured.json
    python3 tools/compare_goldens.py <golden> <measured> --tolerance 0.10 --json

Exit 0 = every metric within tolerance (or improved on a lower_is_better
metric). Exit 1 = at least one regression beyond tolerance. Exit 2 = the two
files do not describe the same metric set, which is a different failure and
must not be reported as a pass.
"""
import argparse
import json
import sys

#: Metrics where a LOWER value is better. Everything else is higher-is-better.
#: Named per file in the golden's own `direction` map; this is only the default.
_DEFAULT_DIRECTION = "higher_is_better"


def _flat(d, prefix=""):
    """Flatten nested dicts to dotted keys, keeping only numeric leaves."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def compare(golden: dict, measured: dict, tolerance: float = 0.10) -> dict:
    direction = golden.get("direction", {}) or {}
    g = _flat(golden.get("metrics", {}))
    m = _flat(measured.get("metrics", {}))

    missing = sorted(set(g) - set(m))
    extra = sorted(set(m) - set(g))
    rows, regressions = [], []
    for name in sorted(set(g) & set(m)):
        old, new = g[name], m[name]
        if old == 0:
            rel = 0.0 if new == 0 else float("inf")
        else:
            rel = (new - old) / old
        want = direction.get(name, _DEFAULT_DIRECTION)
        # A regression is movement in the WRONG direction beyond tolerance.
        # Movement in the right direction is never a failure, however large:
        # a kernel that got 5x faster must not fail its own golden.
        if want == "lower_is_better":
            bad = rel > tolerance
        else:
            bad = rel < -tolerance
        row = {"metric": name, "golden": old, "measured": new,
               "rel_diff": round(rel, 4), "direction": want, "regressed": bad}
        rows.append(row)
        if bad:
            regressions.append(row)

    finite = [r["rel_diff"] for r in rows if r["rel_diff"] not in (float("inf"),)]
    return {"rows": rows, "regressions": regressions,
            "missing_in_measured": missing, "extra_in_measured": extra,
            "avg_rel_diff": round(sum(finite) / len(finite), 4) if finite else 0.0,
            "tolerance": tolerance}


def main() -> int:
    ap = argparse.ArgumentParser(prog="compare_goldens")
    ap.add_argument("golden")
    ap.add_argument("measured")
    ap.add_argument("--tolerance", type=float, default=0.10,
                    help="fractional drift allowed in the wrong direction (default 0.10)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    with open(a.golden) as f:
        golden = json.load(f)
    with open(a.measured) as f:
        measured = json.load(f)
    res = compare(golden, measured, a.tolerance)

    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"{'metric':<44} {'golden':>10} {'measured':>10} {'rel':>8}")
        for r in res["rows"]:
            flag = "  REGRESSED" if r["regressed"] else ""
            print(f"{r['metric']:<44} {r['golden']:>10.4g} {r['measured']:>10.4g} "
                  f"{r['rel_diff']:>+7.1%}{flag}")
        print(f"\navg rel diff: {res['avg_rel_diff']:+.2%}  (tolerance "
              f"{res['tolerance']:.0%} against the metric's own direction)")

    if res["missing_in_measured"] or res["extra_in_measured"]:
        print(f"\nMETRIC SET MISMATCH -- not a pass and not a regression:", file=sys.stderr)
        for n in res["missing_in_measured"]:
            print(f"  missing from measured: {n}", file=sys.stderr)
        for n in res["extra_in_measured"]:
            print(f"  not in golden:         {n}", file=sys.stderr)
        return 2
    if res["regressions"]:
        print(f"\n{len(res['regressions'])} metric(s) regressed beyond tolerance.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
