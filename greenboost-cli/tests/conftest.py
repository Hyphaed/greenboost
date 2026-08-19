"""Test isolation for the greenboost-cli suite.

Found the way it should be found , by reading the flight recorder. After the
AE-5 telemetry landed, the production dataflux log contained
`agent_tool_schema_miss` events with `requested: "optimize_mesh"` and
`known_tools: 3`: not a real session, but this suite's own MCP fixtures,
written straight into `~/.local/share/greenboost/dataflux.jsonl`.

That is worse than untidy. AE-5 exists to measure how often the MODEL calls a
tool by a name it was not shown, and the rename decision hangs on that rate.
A suite that injects its own synthetic misses into the same log poisons the
measurement with fabricated evidence , and the more often the tests run, the
more confident the wrong conclusion looks.

The main repo's `tests/conftest.py` already quarantines the log for its own
suite. This one had no conftest at all, so nothing quarantined it here. Set at
module level, before any test module is imported, because `gb_dataflux`
resolves the log path at import time.
"""
import os
import tempfile
from pathlib import Path

_SESSION_DIR = tempfile.mkdtemp(prefix="gb_cli_dataflux_")
os.environ["GREENBOOST_DATAFLUX_LOG"] = str(Path(_SESSION_DIR) / "session.jsonl")

# Same reasoning as the main suite: several modules read this at import time
# and bootstrap real background telemetry when it is "1".
os.environ["GREENBOOST_ACTIVE"] = "0"
