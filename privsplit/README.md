# privsplit: root helper for the GreenBoost supervisor (SP41-12, QC1)

Design: `docs/greenboost-audit/privsplit-design.md`. Everything here is new and
self-contained; no existing supervisor, installer, service or `tools/manifest/`
file was changed. Nothing is installed or run as root by anything in this directory.

## What is here

| path | what |
|---|---|
| `gb_root_helper.py` | helper daemon, stdlib only, 10 fixed verbs, JSON schemas, SO_PEERCRED uid allow-list, rate limits, JSON-line audit, `Backend` interface for every privileged action |
| `gb_helper_client.py` | client for the unprivileged brain; returns `failed/helper-unavailable` and never falls back to local `rmmod`/`modprobe` |
| `units/` | proposals: `.socket`, helper `.service`, `greenboost-brain.service`, tmpfiles.d entry, drop-ins for A2A and Ollama |
| `patches/supervisor-use-helper.patch` | not applied; routes the supervisor's rmmod/modprobe/apparmor calls through the client behind `GB_ROOT_HELPER=1` (made against the working-tree `gb_supervisor.py` of 2026-10-10, checked with `git apply --check`) |
| `install-privsplit.sh.template` | owner script: `--dry-run`, install, `--rollback`; asks nothing |
| `tests/` | 23 unprivileged tests, stubbed backend, temp sockets under `~/.cache/gb-privsplit-tests` |

## Wire protocol

One request and one reply per connection. Frame: 4-byte big-endian length (max 4096) then one JSON object.
Request `{"verb", "args", "id"}` (exactly these keys; `id` matches `[A-Za-z0-9._-]{1,64}`).
Reply `{"ok", "code", "detail", "id"}`, `code` in `ok bad-verb bad-arg peer-denied precondition rate-limited failed`.
Verbs: `module-load module-unload pcie-maxreadreq gpu-clocks gpu-power cpu-governor vm-sysctl drop-caches apparmor-reload read-gpu-config`.
Check order: peer uid, size, JSON, keys, verb, schema, rate limit, preconditions, action. Garbage does not use rate budget.

## Install (owner, as root; nothing asks a question)

1. Read `units/` and the script. Copy `install-privsplit.sh.template` if you want a local edit.
2. `./install-privsplit.sh.template --dry-run` (no root; prints every step).
3. `sudo ./install-privsplit.sh.template` installs the helper socket and service only; the supervisor keeps running as today.
4. Verify: `systemd-analyze security greenboost-root-helper.service`, then `sudo -u greenboost` against the socket with `gb_helper_client.py`, and `sudo -u nobody` must get `peer-denied` (or fail to connect).
5. Only after the design's tests 6 to 10 pass: apply the patch by hand, install `greenboost-brain.service`, disable the old supervisor unit.
6. Undo: `sudo ./install-privsplit.sh.template --rollback` (restores backed-up files, removes the rest, leaves the `greenboost` user).

## Offline checks run on 2026-10-10 (no root, no install)

`systemd-analyze verify`: no complaint about the three new units. `systemd-analyze security --offline=yes` exposure (the checklist score, not proof; the earlier 9.6 and 9.2 numbers were measured on the live units):
helper 2.0, brain 1.7; existing a2a unit 6.9 and ollama unit 9.0 become 1.8 each with the drop-ins merged in a scratch copy. Targets were 7, 5, 5, 5. Offline scores ignore whether CUDA still works; the ollama and A2A drop-ins must pass the cudaMalloc smoke test (design test 8) before anyone trusts them.

## Open risks

1. Capabilities are inferred, not measured. `setpci` (needs sysfs PCI config write; maybe `CAP_SYS_ADMIN`, or blocked by kernel lockdown) and `apparmor_parser -r` (`CAP_MAC_ADMIN`, maybe `CAP_SYS_ADMIN`) must be tried under `capsh`. If either needs `CAP_SYS_ADMIN`, split that verb into a second unit; do not widen this one.
2. `gpu-clocks` and `gpu-power` use `nvidia-smi`; whether it works without `CAP_SYS_ADMIN` under `DevicePolicy=closed` is untested.
3. Device-in-use check reads `/sys/module/<m>/refcnt` (no capability) instead of scanning other users' `/proc/*/fd`, which would need `CAP_SYS_PTRACE`. If refcnt does not move with open handles on this driver, `rmmod` still refuses (no `-f`), so the cost is a `failed` instead of a `precondition`.
4. The supervisor also writes `/etc/apparmor.d/local/*` as root. That is not a verb. Under the brain it will fail; either add a `apparmor-write-local` verb with a fixed content template or drop the write.
5. The OS tuner writes other sysfs and `vm.*` values than the verbs cover; each uncovered write fails silently until it becomes a verb or is dropped.
6. Helper size is about 450 non-blank lines against the 400 goal in the design.
7. `DeviceAllow` timing trap (design risk 2) is still untested; the socket parent directory needs the tmpfiles entry, because systemd would create it as `root:root 0750` and lock the brain out.
8. Audit lines go to the journal, not yet to the dataflux log.
