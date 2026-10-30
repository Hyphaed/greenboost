"""Role -> local model catalog (environment/models.yaml).

Answers "which model for which job, and when not to" without touching how a model
is served; placement lives in the serving recipe each role points at. Read-only and
cheap: no network, no inference, no GPU probing.
"""
from __future__ import annotations

import os
from pathlib import Path

_CATALOG_PATH = Path(__file__).with_name("models.yaml")
_ENV_PREFIX = "GB_MODEL_"


def load_catalog(path: Path | None = None) -> dict:
    """Parsed catalog, or {} when the file is missing or unreadable. Never raises:
    a broken catalog must not stop the CLI from starting."""
    try:
        import yaml
        data = yaml.safe_load((path or _CATALOG_PATH).read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_role(role: str, catalog: dict | None = None, environ=None) -> dict:
    """The role's entry with `model` replaced by GB_MODEL_<ROLE> when that is set.
    {} for an unknown role. `override` records whether the env var won."""
    cat = catalog if catalog is not None else load_catalog()
    entry = dict((cat.get("roles") or {}).get(role) or {})
    if not entry:
        return {}
    env = os.environ if environ is None else environ
    override = (env.get(_ENV_PREFIX + role.upper()) or "").strip()
    entry["override"] = bool(override)
    if override:
        entry["model"] = override
    return entry


def escalation_chain(catalog: dict | None = None, environ=None) -> list[dict]:
    """Roles in escalation order, each resolved. Unknown names are skipped."""
    cat = catalog if catalog is not None else load_catalog()
    out = []
    for name in cat.get("escalation") or []:
        entry = resolve_role(name, cat, environ)
        if entry:
            out.append({"role": name, **entry})
    return out


def default_agent_model(catalog: dict | None = None, environ=None) -> str:
    """Model name for the default coding agent role, "" when none is configured."""
    return resolve_role("agent_fast", catalog, environ).get("model", "")
