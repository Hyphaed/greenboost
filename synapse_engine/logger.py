# SPDX-License-Identifier: GPL-2.0-only
"""Stdlib shim for the vendored gllm/ tree's `from logger import logger`.

Upstream gLLM expects a third-party PyPI package literally named `logger`;
GreenBoost ships this shim instead (see NOTICE) so the vendored source
needs zero edits and the venv pulls one fewer dependency."""
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [gllm] %(levelname)s %(message)s")
logger = logging.getLogger("gllm")
