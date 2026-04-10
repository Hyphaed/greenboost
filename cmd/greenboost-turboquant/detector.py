"""
GreenBoost TurboQuant — Ollama model detector.

Queries the Ollama /api/ps endpoint to detect the currently loaded model
and extract its size and context length for KV cache estimation.
"""

import urllib.request
import json
import os


def query_ollama_ps(base_url=None):
    """
    Query the Ollama /api/ps endpoint to detect the active model.

    Args:
        base_url: Ollama API base URL. Defaults to OLLAMA_HOST env var or
                  http://localhost:11434.

    Returns:
        dict with keys: name, size_gb, ctx_len, kv_type
        None if no model is loaded or Ollama is unreachable.
    """
    if base_url is None:
        base_url = os.environ.get('OLLAMA_HOST', 'http://localhost:11434').rstrip('/')

    try:
        with urllib.request.urlopen(f'{base_url}/api/ps', timeout=2) as r:
            data = json.loads(r.read())

        models = data.get('models', [])
        if not models:
            return None

        m = models[0]

        # Extract context length from model info if available
        model_meta = m.get('model', {}) or {}
        ctx_len = model_meta.get('context_length', 0)

        # Fall back to OLLAMA_NUM_CTX env var if not reported by API
        if not ctx_len:
            env_ctx = os.environ.get('OLLAMA_NUM_CTX', '0')
            try:
                ctx_len = int(env_ctx)
            except ValueError:
                ctx_len = 8192  # conservative default

        # KV cache type from OLLAMA_KV_CACHE_TYPE env var (set by GreenBoost install)
        kv_type = os.environ.get('OLLAMA_KV_CACHE_TYPE', 'q8_0')

        return {
            'name':    m.get('name', ''),
            'size_gb': m.get('size', 0) / 1e9,
            'ctx_len': ctx_len,
            'kv_type': kv_type,
        }

    except Exception:
        return None
