"""Partial transformer-model loader + layer-range forward for cluster peers.

Each peer owns a contiguous slice of a causal LM:

  role = "embed_head"  → embed_tokens + layers[start:end] + (norm + lm_head)
  role = "embed"       → embed_tokens + layers[start:end]
  role = "middle"      → layers[start:end] only
  role = "head"        → layers[start:end] + norm + lm_head

After load_shard(), the peer's transformer config is preserved (hidden size,
num_layers, head_dim, num_kv_heads) so the coordinator can sanity-check
matching shapes between peers.

Tensors travel across the SSH-tunneled JSON-RPC as base64-encoded numpy
arrays + a shape/dtype descriptor (see `pack_tensor` / `unpack_tensor`).
Per-token hidden state for a typical 4096-dim model is ~8 KB before base64,
so the transport overhead per generation step is negligible compared to
matmul cost.

Per-session KV cache state lives in `_SESSIONS[session_id]`. The coordinator
allocates a unique session_id per `generate()` call and tears it down at
the end. No global mutable state survives across sessions (other than the
loaded shard).

This module imports torch / transformers lazily so the peer_worker process
can import shard_model without paying the torch import cost when only
ping / vram_stats are called.
"""
from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# ── Tensor wire format ───────────────────────────────────────────────────────

def pack_tensor(t: Any) -> dict:
    """torch.Tensor → JSON-serialisable dict.

    Float tensors are serialised as bfloat16 to keep payloads small; ints
    stay int64. The receiver inverts via unpack_tensor.
    """
    import numpy as np
    import torch
    if isinstance(t, np.ndarray):
        arr = t
        dtype = str(arr.dtype)
    else:
        if t.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            arr = t.detach().to(dtype=torch.float32).cpu().numpy().astype(
                np.float16, copy=False)
            dtype = "float16"
        elif t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                         torch.long):
            arr = t.detach().to(dtype=torch.int64).cpu().numpy()
            dtype = "int64"
        elif t.dtype == torch.bool:
            arr = t.detach().cpu().numpy().astype(np.uint8)
            dtype = "bool"
        else:
            arr = t.detach().to(dtype=torch.float32).cpu().numpy().astype(
                np.float16, copy=False)
            dtype = "float16"
    return {
        "shape": list(arr.shape),
        "dtype": dtype,
        "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def unpack_tensor(d: dict, *, device: str = "cpu",
                  target_dtype: str | None = None) -> Any:
    """Inverse of pack_tensor. Returns a torch.Tensor on `device`."""
    import numpy as np
    import torch
    if not isinstance(d, dict) or "b64" not in d:
        raise ValueError("packed tensor expects {shape, dtype, b64}")
    raw = base64.b64decode(d["b64"])
    dtype = d.get("dtype", "float16")
    if dtype == "bool":
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(d["shape"]).astype(bool)
    else:
        np_dtype = np.dtype(dtype) if dtype else np.float16
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(d["shape"]).copy()
    t = torch.from_numpy(arr).to(device)
    if target_dtype == "bfloat16":
        t = t.to(torch.bfloat16)
    elif target_dtype == "float16":
        t = t.to(torch.float16)
    elif target_dtype == "float32":
        t = t.to(torch.float32)
    return t


# ── Shard state ──────────────────────────────────────────────────────────────

@dataclass
class ShardState:
    model_id: str = ""
    role: str = ""                    # embed_head | embed | middle | head
    layer_start: int = 0
    layer_end: int = 0
    total_layers: int = 0
    hidden_size: int = 0
    head_dim: int = 0
    num_kv_heads: int = 0
    device: str = "cpu"
    dtype: str = "bfloat16"
    loaded_at: float = 0.0
    # Filled by load_shard, kept private so this module can be json-dumped
    model: Any = None
    layers: Any = None                # nn.ModuleList sliced down to our range
    embed: Any = None
    norm: Any = None
    lm_head: Any = None
    config: Any = None

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "total_layers": self.total_layers,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "num_kv_heads": self.num_kv_heads,
            "device": self.device,
            "dtype": self.dtype,
            "loaded_at": self.loaded_at,
        }


@dataclass
class SessionState:
    session_id: str
    past_key_values: Any = None
    seen_tokens: int = 0
    created_at: float = field(default_factory=time.time)


# Process-wide state (peer_worker is single-process)
_SHARD: ShardState | None = None
_SESSIONS: dict[str, SessionState] = {}


def get_shard() -> ShardState | None:
    return _SHARD


# ── Loader ───────────────────────────────────────────────────────────────────

_VALID_ROLES = {"embed_head", "embed", "middle", "head"}


def load_shard(
    model_id: str,
    layer_range: list[int],
    role: str,
    *,
    device: str = "auto",
    dtype: str = "bfloat16",
) -> dict:
    """Load only the parts of `model_id` this peer owns.

    Returns a description dict the coordinator can match against expectations.
    Idempotent: if the same (model_id, role, layer_range) is already loaded,
    returns the existing description without reloading.
    """
    global _SHARD, _SESSIONS

    if role not in _VALID_ROLES:
        raise ValueError(f"role must be one of {_VALID_ROLES}, got {role!r}")
    if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
        raise ValueError("layer_range must be [start, end]")
    start, end = int(layer_range[0]), int(layer_range[1])
    if start < 0 or end <= start:
        raise ValueError(f"invalid layer_range [{start}, {end}]")

    # Idempotent fast path
    if _SHARD is not None and (
        _SHARD.model_id == model_id and _SHARD.role == role
        and _SHARD.layer_start == start and _SHARD.layer_end == end
    ):
        return _SHARD.describe()

    # Tear down any old shard + sessions
    unload_shard()

    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    if device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CPU spillover not permitted — GreenBoost must provide CUDA. "
                "Check that the kernel module is loaded and torch is the cu130-tagged build."
            )
        device = "cuda"
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }.get(dtype, torch.bfloat16)

    cfg = AutoConfig.from_pretrained(model_id)
    total_layers = getattr(cfg, "num_hidden_layers", None) or getattr(
        cfg, "n_layer", None)
    if total_layers is None:
        raise RuntimeError(
            f"could not determine num_hidden_layers from {model_id}'s config")
    if end > total_layers:
        raise ValueError(
            f"layer_range end {end} exceeds total_layers {total_layers}")

    # Load full model on CPU first to keep VRAM low during the slice step.
    # We trust transformers to dispatch the weights; we'll prune to our slice
    # immediately and move only what we keep to the target device.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    # Locate the components. Most modern HF causal LMs (Llama, Mistral, Qwen,
    # Gemma…) expose model.model.{embed_tokens, layers, norm} and model.lm_head.
    inner = getattr(model, "model", None) or model
    full_layers = getattr(inner, "layers", None)
    embed = getattr(inner, "embed_tokens", None)
    norm = getattr(inner, "norm", None)
    lm_head = getattr(model, "lm_head", None)

    if full_layers is None:
        raise RuntimeError(
            f"could not locate transformer layers in {type(model).__name__}; "
            "this model is not currently supported by sharded_forward")

    # Slice + free the rest
    kept_layers = torch.nn.ModuleList([full_layers[i] for i in range(start, end)])

    keep_embed = role in {"embed_head", "embed"}
    keep_head = role in {"embed_head", "head"}

    embed_mod = embed if keep_embed else None
    norm_mod = norm if keep_head else None
    head_mod = lm_head if keep_head else None

    # Move what we keep to device
    kept_layers = kept_layers.to(device=device, dtype=torch_dtype)
    if embed_mod is not None:
        embed_mod = embed_mod.to(device=device, dtype=torch_dtype)
    if norm_mod is not None:
        norm_mod = norm_mod.to(device=device, dtype=torch_dtype)
    if head_mod is not None:
        head_mod = head_mod.to(device=device, dtype=torch_dtype)

    # Drop the full model wrapper to free memory
    del full_layers, inner, model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    head_dim = getattr(cfg, "head_dim", 0) or (
        getattr(cfg, "hidden_size", 0) // max(1, getattr(cfg, "num_attention_heads", 1)))

    _SHARD = ShardState(
        model_id=model_id,
        role=role,
        layer_start=start,
        layer_end=end,
        total_layers=total_layers,
        hidden_size=getattr(cfg, "hidden_size", 0),
        head_dim=head_dim,
        num_kv_heads=getattr(cfg, "num_key_value_heads", getattr(
            cfg, "num_attention_heads", 0)),
        device=device,
        dtype=dtype,
        loaded_at=time.time(),
        model=None,
        layers=kept_layers,
        embed=embed_mod,
        norm=norm_mod,
        lm_head=head_mod,
        config=cfg,
    )
    return _SHARD.describe()


def unload_shard() -> dict:
    """Drop the loaded shard and all sessions. Frees VRAM."""
    global _SHARD, _SESSIONS
    _SESSIONS.clear()
    if _SHARD is None:
        return {"unloaded": False}
    _SHARD = None
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"unloaded": True}


# ── Session lifecycle ────────────────────────────────────────────────────────

def start_session(session_id: str | None = None) -> str:
    sid = session_id or uuid.uuid4().hex[:12]
    _SESSIONS[sid] = SessionState(session_id=sid)
    return sid


def end_session(session_id: str) -> dict:
    sess = _SESSIONS.pop(session_id, None)
    return {"ended": bool(sess)}


def session_count() -> int:
    return len(_SESSIONS)


# ── Forward steps ────────────────────────────────────────────────────────────

def _need_shard() -> ShardState:
    if _SHARD is None:
        raise RuntimeError("no shard loaded — call load_shard first")
    return _SHARD


def _get_session(session_id: str) -> SessionState:
    sess = _SESSIONS.get(session_id)
    if sess is None:
        raise RuntimeError(f"unknown session_id: {session_id}")
    return sess


def embed_step(session_id: str, token_ids: dict) -> dict:
    """Embed tokens + run our layer slice. Only valid for role embed*.

    token_ids: packed tensor of shape (batch, seq).
    Returns: packed hidden states (batch, seq, hidden_size).
    """
    import torch
    shard = _need_shard()
    if shard.role not in {"embed_head", "embed"}:
        raise RuntimeError(
            f"embed_step requires role embed* (have {shard.role})")
    sess = _get_session(session_id)

    ids = unpack_tensor(token_ids, device=shard.device).to(torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    hidden = shard.embed(ids)
    hidden = _run_layers(shard, hidden, sess)
    if shard.role == "embed_head":
        # This peer also runs norm + lm_head; return logits in a separate call
        # for symmetry with the multi-peer path. Coordinator will call head_step.
        pass
    sess.seen_tokens += ids.shape[1]
    return {"hidden_states": pack_tensor(hidden),
            "seen_tokens": sess.seen_tokens}


def middle_step(session_id: str, hidden_states: dict) -> dict:
    """Run our layer slice on incoming hidden states. Role: middle, head, or
    embed (in pipelines with 3+ peers)."""
    shard = _need_shard()
    sess = _get_session(session_id)
    hidden = unpack_tensor(hidden_states, device=shard.device,
                            target_dtype=shard.dtype)
    hidden = _run_layers(shard, hidden, sess)
    return {"hidden_states": pack_tensor(hidden),
            "seen_tokens": sess.seen_tokens}


def head_step(session_id: str, hidden_states: dict) -> dict:
    """Apply final norm + lm_head. Returns logits over vocab.

    Only valid for role head* (head, embed_head). We always return only the
    logits for the *last* token (greedy/sample is done on the coordinator).
    """
    shard = _need_shard()
    if shard.role not in {"embed_head", "head"}:
        raise RuntimeError(
            f"head_step requires role head* (have {shard.role})")
    _get_session(session_id)  # presence check
    hidden = unpack_tensor(hidden_states, device=shard.device,
                            target_dtype=shard.dtype)
    if shard.norm is not None:
        hidden = shard.norm(hidden)
    logits = shard.lm_head(hidden[:, -1, :])
    return {"logits": pack_tensor(logits)}


def _run_layers(shard: ShardState, hidden, sess: SessionState):
    """Run hidden through this peer's layers, updating sess.past_key_values."""
    import torch
    layer_kwargs = {"use_cache": True}
    # transformers' DynamicCache is the new default; some older models still
    # accept tuple-of-tuples. We attempt DynamicCache first.
    if sess.past_key_values is None:
        try:
            from transformers.cache_utils import DynamicCache
            sess.past_key_values = DynamicCache()
        except ImportError:
            sess.past_key_values = None

    for i, layer in enumerate(shard.layers):
        out = layer(
            hidden,
            past_key_value=sess.past_key_values,
            use_cache=True,
        )
        # Layer call returns a tuple; first element is hidden states.
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden
