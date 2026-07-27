"""gb_aviary.py — GB-Synapse's context-extension, certification and coherence layer.

Adapted from the aviary-1m harness (SatGeze, MIT: bake_yarn.py, niah_test.py,
tools/smoke_gate.py). Three ideas from that project earn their place in
GreenBoost, and each is stronger here than it was standalone:

  * bake_yarn()   — write YaRN rope-scaling metadata into a GGUF so llama.cpp
                    serves it at 1M context with no flags. No weights change:
                    the trunk stays bit-identical, only rope metadata moves.
  * niah_certify()— plant N needles at spread depths and score retrieval, so a
                    long-context claim is MEASURED, never assumed. The upstream
                    tool talks to a hand-rolled llama-server; ours runs against
                    gb-synapse, so a model is certified on the SAME cluster
                    (RPC split, gb-quant, tiering) that will serve it — a
                    certificate from a different configuration proves nothing
                    about the one you run.
  * smoke_gate()  — catch repetition-collapse (the low-bit failure signature)
                    before a quant is trusted. This is the quality floor that
                    makes "never drop below fp8" enforceable rather than
                    aspirational: gb-quant can propose a quant, this refutes it.

Every result lands in dataflux (`yarn_bake`, `niah_cert`, `smoke_gate`) instead
of a local results.jsonl, so certification history is queryable over MCP next to
the tier moves and tok/s that explain it. Upstream publishes the imperfect runs
too; so do we — a FAIL is emitted exactly like a PASS.
"""
from __future__ import annotations

import collections
import json
import os
import random
import time
import urllib.request
from pathlib import Path

def _default_synapse_url() -> str:
    """gb-synapse's own port (GB_SYNAPSE_PORT, default 11435) , not raw
    Ollama's legacy :11434. gb-synapse is THE Ollama replacement (owner
    rule, 2026-07-15); certifying against the wrong port would measure a
    different serving path than the one that actually runs the model."""
    try:
        import gb_synapse
        return f"http://127.0.0.1:{gb_synapse.DEFAULT_PORT}"
    except Exception:
        return "http://127.0.0.1:11435"


# 1M is the default target because it is the certified frontier, not because it
# is the maximum: the upstream "Beyond 1M" ladder study found clean retrieval to
# 1.31M, one dropped needle at 1.57M, and a visible bend at 2M (6/10).
YARN_TARGET_CTX = 1048576

# The factor itself costs quality, independently of the length you use: the SAME
# model at the SAME 1M rung scored 10/10 baked at factor 6 and 9/10 at factor 8,
# because interpolating rope frequencies across a wider range blurs positional
# geometry even below the new maximum. So we never take a factor as input — we
# derive the SMALLEST one that reaches the target. Past this the study's data
# stops supporting the extension, so we warn rather than pretend.
YARN_FACTOR_CERTIFIED_MAX = 6.0

_CITIES = ["Wellington", "Auckland", "Christchurch", "Dunedin", "Hamilton",
           "Tauranga", "Napier", "Nelson", "Queenstown", "Rotorua",
           "Invercargill", "Whangarei", "Gisborne", "Timaru", "Blenheim"]

_FILLER = [
    "The morning fog rolled slowly across the harbour while fishing boats prepared their nets for the day ahead.",
    "Economists continue to debate whether interest rate adjustments have any measurable effect on regional housing markets.",
    "The old library on Cuba Street holds thousands of maps that nobody has catalogued since the nineteen seventies.",
    "Migration patterns of coastal birds shift subtly each decade in response to changing ocean temperatures.",
    "A good sourdough starter requires patience, consistent feeding, and a kitchen that stays reasonably warm overnight.",
    "The tramway commission rejected three separate proposals before settling on the current route through the valley.",
    "Volcanic soil in the region produces wines with a mineral character that critics struggle to describe precisely.",
    "Software projects tend to accumulate complexity gradually until someone insists on deleting half of everything.",
    "The lighthouse keeper's journal records forty years of storms, shipwrecks, and the occasional visiting whale.",
    "Local rugby clubs have merged twice in the past decade as rural populations drift toward the cities.",
]


def _emit(payload: dict) -> None:
    try:
        import gb_dataflux
        gb_dataflux.emit(payload)
    except Exception:
        pass


# ── YaRN baking ───────────────────────────────────────────────────────────────

def bake_yarn(src: str | Path, dst: str | Path | None = None,
              target_ctx: int = YARN_TARGET_CTX) -> Path:
    """Copy a GGUF with YaRN rope-scaling metadata baked in, extending its
    usable context to `target_ctx` without touching a single weight.

    The rope keys are written under the model's OWN architecture prefix (read
    from the file, never assumed — qwen35moe, qwen35, gemma4, ... each namespace
    their own keys), which is why one function covers every supported family.

    The factor is DERIVED (target / native), never passed in, because the
    smallest factor that reaches a length is always the best one at that length:
    a bigger factor blurs positional geometry even in the range it didn't need
    to stretch.
    """
    from gb_gguf_tensor_map import _load_gguf_reader
    _load_gguf_reader()                       # puts the vendored gguf-py on sys.path
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.stem + f"-{target_ctx // 1024}K.gguf")

    reader = gguf.GGUFReader(str(src), "r")
    arch = reader.get_field("general.architecture").contents()
    native = int(reader.get_field(f"{arch}.context_length").contents())

    if target_ctx <= native:
        raise ValueError(
            f"{src.name} already serves {native} tokens natively; baking YaRN for "
            f"{target_ctx} would scale rope for no reason and cost retrieval quality.")

    factor = target_ctx / native
    if factor > YARN_FACTOR_CERTIFIED_MAX:
        print(f"  [gb-aviary] WARNING: factor {factor:.1f} exceeds the certified "
              f"frontier ({YARN_FACTOR_CERTIFIED_MAX:.0f}x). Upstream measured a visible "
              f"retrieval bend there (6/10 needles at 2M). Certify before trusting this "
              f"build: gb_aviary.niah_certify(...).", flush=True)

    new_meta = {
        f"{arch}.context_length":
            MetadataDetails(gguf.GGUFValueType.UINT32, target_ctx),
        f"{arch}.rope.scaling.type":
            MetadataDetails(gguf.GGUFValueType.STRING, "yarn"),
        f"{arch}.rope.scaling.factor":
            MetadataDetails(gguf.GGUFValueType.FLOAT32, float(factor)),
        f"{arch}.rope.scaling.original_context_length":
            MetadataDetails(gguf.GGUFValueType.UINT32, native),
    }
    writer = gguf.GGUFWriter(str(dst), arch=arch, endianess=reader.endianess)
    copy_with_new_metadata(reader, writer, new_meta, [])

    _emit({"kind": "yarn_bake", "status": "ok", "src": src.name, "dst": dst.name,
           "arch": arch, "native_ctx": native, "target_ctx": target_ctx,
           "factor": factor})
    return dst


# ── Needle-in-a-haystack certification ────────────────────────────────────────

def _tokenize_len(url: str, text: str) -> int:
    req = urllib.request.Request(f"{url}/tokenize",
                                 data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return len(json.load(r)["tokens"])


def _build_haystack(target_tokens: int, needles, chars_per_token: float, seed: int) -> str:
    rng = random.Random(seed)
    chunks, total = [], 0
    target_chars = int(target_tokens * chars_per_token)
    while total < target_chars:
        s = rng.choice(_FILLER)
        chunks.append(s)
        total += len(s) + 1
    at = {int(f * len(chunks)): s for f, s in needles}
    out = []
    for i, c in enumerate(chunks):
        if i in at:
            out.append(at[i])
        out.append(c)
    return " ".join(out)


def niah_certify(model: str, tokens: int, needles: int = 10,
                 url: "str | None" = None, seed: int = 1337,
                 kv_type: str = "unknown") -> dict:
    """Plant `needles` secret codes at evenly spread depths in a `tokens`-long
    haystack and score how many the model can retrieve in one pass.

    Runs against gb-synapse's OpenAI surface, so what gets certified is the real
    serving path — cluster RPC split, gb-quant, tiering and all. `kv_type` is
    recorded with the score because it changes the meaning of the number:
    upstream certifies on f16 KV only and labels quantized-KV runs as budget
    configs. An unlabelled score is a claim, not a certificate.
    """
    if url is None:
        url = _default_synapse_url()
    rng = random.Random(seed)
    cities = rng.sample(_CITIES, needles)
    codes = {c: str(rng.randint(1000000, 9999999)) for c in cities}
    depths = [round((i + 0.5) / needles, 3) for i in range(needles)]
    planted = [(d, f"Remember this fact: the secret code for {c} is {codes[c]}.")
               for d, c in zip(depths, cities)]

    sample = " ".join(random.Random(7).choices(_FILLER, k=300))
    cpt = len(sample) / _tokenize_len(url, sample)

    hay = _build_haystack(tokens - 600, planted, cpt, seed=seed * 31 + 42)
    question = ("Now answer using only the text above. List the secret code for each of "
                "these cities, one per line in the format 'City: code'. Cities: "
                + ", ".join(sorted(cities)) + ".")
    prompt = hay + "\n\n" + question
    actual = _tokenize_len(url, prompt)

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=100000) as r:
        resp = json.load(r)
    dt = time.time() - t0

    answer = resp["choices"][0]["message"]["content"] or ""
    hits = [{"depth": d, "city": c, "hit": codes[c] in answer}
            for d, c in sorted(zip(depths, cities))]
    score = sum(h["hit"] for h in hits)

    out = {"kind": "niah_cert", "status": "ok" if score == needles else "error",
           "model": model, "target_tokens": tokens, "prompt_tokens": actual,
           "score": score, "needles": needles, "kv_type": kv_type,
           "wall_s": round(dt, 1),
           "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
           "depths": hits}
    _emit(out)                      # misses are published exactly like passes
    return out


# ── Coherence gate ────────────────────────────────────────────────────────────

def smoke_gate(model: str, url: "str | None" = None) -> dict:
    """Refuse a model that has collapsed into repetition.

    A quant that is too low doesn't error — it loops. Six-gram repetition and
    token-uniqueness catch that signature in one cheap turn, which is what lets
    gb-quant's "never below fp8" rule be enforced against evidence instead of
    against a table of assumptions.
    """
    if url is None:
        url = _default_synapse_url()
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say hello and name three colors."}],
        "temperature": 0.0, "max_tokens": 220,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        content = (json.load(r)["choices"][0]["message"]["content"] or "")

    toks = content.split()
    if len(toks) < 8:
        out = {"verdict": "FAIL", "reason": f"only {len(toks)} tokens of content",
               "max6gram": 0, "uniq": 0.0}
    else:
        grams = collections.Counter(tuple(toks[i:i + 6]) for i in range(len(toks) - 5))
        worst = grams.most_common(1)[0][1] if grams else 0
        uniq = len(set(toks)) / len(toks)
        fail = worst >= 4 or uniq < 0.25
        out = {"verdict": "FAIL" if fail else "PASS",
               "reason": "repetition collapse" if fail else "",
               "max6gram": worst, "uniq": round(uniq, 2)}

    out.update({"kind": "smoke_gate", "model": model, "sample": content[:160],
                "status": "ok" if out["verdict"] == "PASS" else "error"})
    _emit(out)
    return out
