#!/usr/bin/env bash
# G-QC2 container gate: a real install then uninstall inside a throwaway ubuntu
# container with a zero tree diff. Needs docker (no sudo if your user is in the
# docker group) and network for the image and apt. Nothing on this machine is
# touched: the repo is mounted read-only, output goes to a directory you name.
# Usage: tests/installer/container_gate.sh <output-dir> [image]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:?output dir (created if missing)}"; IMG="${2:-ubuntu:24.04}"
mkdir -p "$OUT"
docker info >/dev/null 2>&1 || { echo "docker daemon not reachable: use the fake-root harness (tools/manifest/tests) instead"; exit 2; }
docker run --rm --network bridge -v "$REPO":/repo:ro -v "$HERE/container_inner.sh":/inner.sh:ro -v "$(cd "$OUT" && pwd)":/out "$IMG" bash /inner.sh
