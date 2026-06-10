# GreenBoost Profile Format

Profiles are Markdown files with YAML-style frontmatter and inline field declarations.
The YAML block is machine-parseable by bash (`grep`/`cut`) and Python (`pyyaml`).
The Markdown body is human-readable.

## Storage

```
/etc/greenboost/
├── profiles/
│   ├── default.md                    # auto-generated on install
│   └── resolved_<timestamp>.md       # written on --profile conflict resolution
└── active_profile.md                 # symlink → profiles/<active>.md
```

## CLI

```bash
# Profile management
sudo ./greenboost_setup.sh profile create           # auto-detect hardware → write default.md
sudo ./greenboost_setup.sh profile show             # print active profile
sudo ./greenboost_setup.sh profile show <file>      # print specific file
sudo ./greenboost_setup.sh profile list             # list available profiles
sudo ./greenboost_setup.sh profile activate <file>  # set active_profile.md symlink
sudo ./greenboost_setup.sh profile diff [file]      # compare profile vs live hardware

# Load with a user-supplied profile (any command accepts --profile)
sudo ./greenboost_setup.sh --profile ~/my_profile.md load
sudo ./greenboost_setup.sh --profile ~/my_profile.md full-install
```

## Resolution Priority

```
CLI flags > environment vars > active profile > compiled-in defaults
```

## Required GreenBoost Parameter Fields

| Field | Type | Notes |
|-------|------|-------|
| `physical_vram_gb` | int | GPU VRAM in GB |
| `virtual_vram_gb` | int | System RAM pool (T2) in GB |
| `safety_reserve_gb` | int | Min free RAM to always keep |
| `nvme_swap_gb` | int | T3 NVMe swap cap |
| `nvme_pool_gb` | int | GreenBoost T3 soft cap |
| `use_hugepages` | 0\|1 | 2 MB THP for T2 allocations |
| `pcores_only` | 0\|1 | Pin watchdog to P-cores (Intel hybrid only) |
| `tier3_backend` | string | `nvme` \| `lustre` \| `gpfs` |

## Conflict Resolution Rules

When `--profile <file>` is supplied, GreenBoost runs auto-detection then
cross-checks against the profile. Conflicts are resolved as follows:

| Field | Rule |
|-------|------|
| `physical_vram_gb` | Always use detected - cannot claim VRAM that doesn't exist |
| `virtual_vram_gb` | Use profile value if ≤ 90% of RAM, else cap |
| `safety_reserve_gb` | Use max(profile, 10% of RAM) |
| `nvme_swap_gb` | Use profile value if ≤ NVMe capacity, else cap |
| `nvme_pool_gb` | Use min(profile, nvme_swap × 0.89) |
| `nvlink_pool` | Override to false if no NVLink detected |
| All others | Use profile value |

## Profile Types

| Type | Description |
|------|-------------|
| `workstation` | Single GPU, desktop OS, NVMe local swap |
| `server` | Single or small multi-GPU, rack server |
| `cluster_node` | Multi-GPU with NVLink, InfiniBand, parallel FS |
| `edge` | Low-power device (Jetson, embedded) |

## Example Profiles

- `examples/workstation_i9_rtx5070.md` - i9-14900KF + RTX 5070, 64 GB DDR4
- `examples/cluster_node_v100_8gpu.md` - Dual Xeon + 8× V100, 384 GB DDR4 ECC

## Sysfs (post-load)

```bash
cat /sys/class/greenboost/greenboost/active_profile   # profile name
```
