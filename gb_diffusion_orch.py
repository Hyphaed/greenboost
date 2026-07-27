#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_diffusion_orch.py , GreenBoost diffusion pipeline orchestrator.

Ties together all GreenBoost layers for Hugging Face Diffusers pipelines
(FLUX.2-klein, SD-XL, LTX-2.3, etc.):

  GpuTelemetry   , live HBM / PCIe / power metrics
  ModelTierManager , T1/T2/T3 model paging
  MemPoolManager  , per-purpose CUDA pools (no empty_cache!)
  Stream scheduler , overlap compute / transfers / quantization

Stage-aware lifecycle:
  1. encode   : text encoders on GPU, UNet/VAE on CPU
  2. denoise  : UNet on GPU, encoders freed, VAE on CPU
     (async-prefetch VAE during last denoising step)
  3. decode   : VAE on GPU, UNet demoted

GreenBoost-specific rules enforced here:
  - NEVER calls torch.cuda.empty_cache() , patched to no-op at startup
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False must remain set
  - Uses gb_quant batch mode: quantize → encode_all → free_encoders → quantize_denoiser

Usage:
    from diffusers import FluxPipeline
    from gb_diffusion_orch import DiffusionOrchestrator

    pipe = FluxPipeline.from_pretrained(...)
    orch = DiffusionOrchestrator(pipe)   # headroom %-derived from local VRAM

    # Standard two-phase batch gen:
    with orch.encode_phase():
        embeds = gb_quant.encode_prompts(pipe, prompts)
    with orch.denoise_phase():
        images = [pipe(..., prompt_embeds=e).images[0] for e in embeds]
    with orch.decode_phase():
        pass  # VAE already promoted during last denoise step

    orch.close()
"""
from __future__ import annotations

import contextlib
import gc
import os
from typing import List, Optional

import torch

# Bootstrap all GreenBoost layers; no-op when not active.
try:
    import gb_init as _gb_init
except ImportError:
    _gb_init = None


class DiffusionOrchestrator:
    """
    Orchestrates a Diffusers pipeline across GreenBoost T1/T2/T3.

    Parameters
    ----------
    pipe : diffusers pipeline
        Any Diffusers pipeline with sub-models accessible as attributes.
    encoder_attrs : tuple of str
        Names of text encoder sub-models (freed after encode phase).
    denoiser_attrs : tuple of str
        Names of denoiser sub-models (UNet / transformer).
    vae_attrs : tuple of str
        Names of VAE sub-models (kept on CPU until decode phase).
    hbm_headroom_mb : int
        Minimum free VRAM before a promote operation; excess models are demoted.
    telemetry_interval_s : float
        Background telemetry polling interval. 0 = disable.
    feeder_block_attr : str, optional
        Name of a uniformly-called transformer block list on the denoiser
        (e.g. "single_transformer_blocks" for Flux/Flux2) to offload to a
        connected feeder's GPU , see gb_cluster.offload_tail_blocks. When
        set, denoise_phase() implicitly tries the offload the first time
        it's entered: no per-pipeline cluster code required, mirroring the
        auto-on feeder use ai-forge's gen_art.py already does explicitly.
        None (default) , no implicit feeder use; unaffected by feeders.
    feeder_vram_budget_gb : float
        VRAM budget on the feeder to fill with offloaded blocks.
    """

    def __init__(
        self,
        pipe,
        encoder_attrs: tuple = ("text_encoder", "text_encoder_2", "text_encoder_3"),
        denoiser_attrs: tuple = ("transformer", "unet"),
        vae_attrs: tuple = ("vae",),
        hbm_headroom_mb: "int | None" = None,   # None → gb_topology.hbm_headroom_mb()
                                                # (%-derived; owner rule 2026-07-13)
        telemetry_interval_s: float = 0.0,
        feeder_block_attr: Optional[str] = None,
        feeder_vram_budget_gb: float = 5.5,
    ):
        self.pipe = pipe
        self.encoder_attrs = encoder_attrs
        self.denoiser_attrs = denoiser_attrs
        self.vae_attrs = vae_attrs
        self.feeder_block_attr = feeder_block_attr
        self.feeder_vram_budget_gb = feeder_vram_budget_gb
        self._feeder_block_client = None
        self._feeder_offload_tried = False

        from gb_model_tier import ModelTierManager, Tier
        from gb_mem_pool import MemPoolManager
        import gb_stream_sched as gs

        # Prefer the gb_init singleton (avoids spawning a duplicate thread);
        # fall back to creating a local instance if gb_init is not active.
        if _gb_init is not None and _gb_init.get_telemetry() is not None:
            self.tel = _gb_init.get_telemetry()
        else:
            from gb_telemetry import TelemetryManager
            self.tel = TelemetryManager(device=0, poll_ms=100)
            self.tel.start()
        self._tel_owned = (
            _gb_init is None or _gb_init.get_telemetry() is None
        )

        # Prefer gb_init singletons for tier/pool managers too.
        self.tm    = (_gb_init.get_tier_manager() if _gb_init else None) \
                     or ModelTierManager(hbm_headroom_mb=hbm_headroom_mb)
        self.pools = (_gb_init.get_mem_pools() if _gb_init else None) \
                     or MemPoolManager()
        self.gs    = (_gb_init.get_stream_sched() if _gb_init else None) or gs

        # Register all sub-models
        for attr in encoder_attrs:
            mod = getattr(pipe, attr, None)
            if mod is not None:
                self.tm.register(attr, mod, tier=Tier.T1)
        for attr in denoiser_attrs:
            mod = getattr(pipe, attr, None)
            if mod is not None:
                self.tm.register(attr, mod, tier=Tier.T1)
        for attr in vae_attrs:
            mod = getattr(pipe, attr, None)
            if mod is not None:
                self.tm.register(attr, mod, tier=Tier.T1)

        if telemetry_interval_s > 0:
            self.tel.add_callback(self._on_metrics)

    # ── telemetry callback ────────────────────────────────────────────────────

    def _on_metrics(self, m):
        if m.should_demote:
            self.tm.auto_evict(m)

    def _maybe_offload_to_feeder(self):
        """Implicit cluster flux: if feeder_block_attr was given at
        construction and a feeder is online (checked once, lazily, so
        pipelines that never connect a feeder pay zero cost), move the tail
        of the denoiser onto the feeder GPU. Silent no-op with zero feeders
        or on any offload failure , this is a speed optimization, never a
        requirement. Opt-out: GB_AUTO_CLUSTER=0.
        """
        if self._feeder_offload_tried or self.feeder_block_attr is None:
            return
        self._feeder_offload_tried = True
        if os.environ.get("GB_AUTO_CLUSTER", "1") == "0":
            return
        try:
            import gb_cluster
            if not gb_cluster.cluster_available():
                return
            for attr in self.denoiser_attrs:
                mod = getattr(self.pipe, attr, None)
                if mod is not None and hasattr(mod, self.feeder_block_attr):
                    self._feeder_block_client = gb_cluster.offload_tail_blocks(
                        mod, self.feeder_block_attr,
                        vram_budget_gb=self.feeder_vram_budget_gb)
                    if self._feeder_block_client is not None:
                        print(f"[gb_orch] denoise_phase: offloaded tail of "
                              f"{attr}.{self.feeder_block_attr} to feeder",
                              flush=True)
                    break
        except Exception as e:
            print(f"[gb_orch] feeder block offload skipped: "
                  f"{e.__class__.__name__}: {e}", flush=True)

    # ── phase context managers ────────────────────────────────────────────────

    def _emit_phase(self, phase: str, promoted: tuple, demoted: tuple) -> None:
        """Best-effort dataflux event for a real T1/T2 phase transition , this
        choreographer is the phase-promotion decision-maker for every
        ai-forge image-gen pipeline that runs through gb_quant's batch mode
        (encode -> denoise -> decode), and previously left zero dataflux
        trace of which components moved where and when."""
        try:
            import gb_dataflux
            gb_dataflux.emit({
                "node": "host", "label": "gb_diffusion_orch", "kind": "placement",
                "runtime": "diffusion_phase", "phase": phase,
                "promoted": list(promoted), "demoted": list(demoted),
            })
        except Exception:
            pass

    @contextlib.contextmanager
    def encode_phase(self):
        """
        Prepare for prompt encoding:
          - Encoders on GPU (T1)
          - Denoiser / VAE on CPU (T2) to free VRAM
        Yields into the encoding context; on exit frees encoder memory.
        """
        print("[gb_orch] encode_phase: promoting encoders → T1", flush=True)
        self._emit_phase("encode", self.encoder_attrs, self.denoiser_attrs + self.vae_attrs)
        self.tm.session_active()

        # Demote denoiser + VAE first to make room
        for attr in self.denoiser_attrs + self.vae_attrs:
            if attr in self.tm._entries:
                self.tm.demote(attr)

        for attr in self.encoder_attrs:
            if attr in self.tm._entries:
                self.tm.promote(attr)

        with self.pools.use("activations"):
            yield

        # Free encoders from GPU after encoding; no empty_cache
        print("[gb_orch] encode_phase: freeing encoders from T1", flush=True)
        for attr in self.encoder_attrs:
            mod = getattr(self.pipe, attr, None)
            if mod is not None:
                try:
                    delattr(self.pipe, attr)
                except AttributeError:
                    pass
                try:
                    setattr(self.pipe, attr, None)
                except Exception:
                    pass
        gc.collect()
        self.pools.trim("activations")

    @contextlib.contextmanager
    def denoise_phase(self, prefetch_vae: bool = True, total_steps: int = 0,
                       diffcache_threshold: Optional[float] = None,
                       diffcache_max_skip: int = 2):
        """
        Prepare for denoising loop:
          - Denoiser on GPU (T1)
          - VAE on CPU (T2)
          - Optionally prefetch VAE on transfer stream during last step
          - Optionally enable activation caching (gb_diffcache) on the
            denoiser for this phase, skipping recompute on near-identical
            consecutive timesteps. Pass diffcache_threshold to enable
            (e.g. 0.1); None (default) leaves the denoiser unpatched.
        """
        print("[gb_orch] denoise_phase: promoting denoiser → T1", flush=True)
        self._emit_phase("denoise", self.denoiser_attrs, self.vae_attrs)

        for attr in self.denoiser_attrs:
            if attr in self.tm._entries:
                self.tm.promote(attr)
        for attr in self.vae_attrs:
            if attr in self.tm._entries:
                self.tm.demote(attr)

        self._maybe_offload_to_feeder()

        self._prefetch_vae_done = False
        self._total_steps = total_steps

        self._diffcache_modules: list = []
        if diffcache_threshold is not None:
            import gb_diffcache
            for attr in self.denoiser_attrs:
                mod = getattr(self.pipe, attr, None)
                if mod is not None:
                    gb_diffcache.patch(mod, threshold=diffcache_threshold,
                                        max_skip=diffcache_max_skip)
                    gb_diffcache.reset(mod)
                    self._diffcache_modules.append(mod)

        try:
            with self.pools.use("latents"):
                yield self
        finally:
            if self._diffcache_modules:
                import gb_diffcache
                for mod in self._diffcache_modules:
                    print(f"[gb_orch] diffcache: {gb_diffcache.status(mod)}", flush=True)
                    gb_diffcache.unpatch(mod)

        self.pools.trim("latents")

    def diffcache_reset(self):
        """Clear diffcache state for all patched denoiser modules. Call at
        the start of each new image/video generation within a batch so the
        first timestep never reuses the previous generation's last step."""
        if not getattr(self, "_diffcache_modules", None):
            return
        import gb_diffcache
        for mod in self._diffcache_modules:
            gb_diffcache.reset(mod)

    def step(self, step_idx: int):
        """
        Call from inside the denoising loop to trigger VAE prefetch on the
        last N steps (so VAE→GPU transfer overlaps the final denoising step).
        """
        if self._total_steps > 0 and not self._prefetch_vae_done:
            if step_idx >= self._total_steps - 2:
                # Async-prefetch VAE during the last couple of steps
                for attr in self.vae_attrs:
                    mod = getattr(self.pipe, attr, None)
                    if mod is not None and attr in self.tm._entries:
                        import gb_stream_sched as gs
                        with gs.on("transfer"):
                            mod.to("cuda", non_blocking=True)
                self._prefetch_vae_done = True

    @contextlib.contextmanager
    def decode_phase(self):
        """
        Prepare for VAE decode:
          - VAE on GPU (T1), denoiser demoted to T2
        """
        print("[gb_orch] decode_phase: promoting VAE → T1", flush=True)
        self._emit_phase("decode", self.vae_attrs, self.denoiser_attrs)

        for attr in self.denoiser_attrs:
            if attr in self.tm._entries:
                self.tm.demote(attr)

        if not self._prefetch_vae_done:
            for attr in self.vae_attrs:
                if attr in self.tm._entries:
                    self.tm.promote(attr)
        else:
            # Already prefetched; just wait for the transfer stream
            self.gs.wait_for("transfer", on="gemm")
            for attr in self.vae_attrs:
                if attr in self.tm._entries:
                    self.tm._entries[attr].tier = "T1_HBM"

        with self.pools.use("latents"):
            yield

        self.pools.trim("latents")

    # ── batch helper (simple path, no manual phase management) ───────────────

    def simple_batch(
        self,
        prompts: List[str],
        height: int = 1216,
        width: int = 832,
        steps: int = 4,
        guidance: float = 1.0,
        seeds: Optional[List[int]] = None,
        diffcache_threshold: Optional[float] = None,
        diffcache_max_skip: int = 2,
    ):
        """
        Run the full gb_quant int4 batch pipeline in one call.
        Returns list of PIL images.

        Requires gb_quant to be importable (i.e. running inside _gen_gb.sh env).

        diffcache_threshold: pass e.g. 0.1 to enable gb_diffcache activation
        caching on the denoiser for this batch (skip recompute on near-
        identical consecutive timesteps). None (default) disables it.
        """
        import gb_quant

        seeds = seeds or [42] * len(prompts)

        with self.encode_phase():
            gb_quant.quantize_encoders(self.pipe, bits=4)
            all_embeds = gb_quant.encode_prompts(self.pipe, prompts)

        gb_quant.quantize_denoiser(self.pipe, bits=4)

        images = []
        with self.denoise_phase(prefetch_vae=True, total_steps=steps,
                                 diffcache_threshold=diffcache_threshold,
                                 diffcache_max_skip=diffcache_max_skip) as phase:
            for i, (emb, seed) in enumerate(zip(all_embeds, seeds)):
                phase.diffcache_reset()
                gen = torch.Generator(device="cpu").manual_seed(seed)
                with self.pools.use("latents"):
                    with self.gs.on("gemm"):
                        img = self.pipe(
                            prompt_embeds=emb.to("cuda"),
                            height=height,
                            width=width,
                            guidance_scale=guidance,
                            num_inference_steps=steps,
                            generator=gen,
                        ).images[0]
                phase.step(i)
                images.append(img)

        gc.collect()
        return images

    # ── cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        # Only stop the telemetry thread if we own it (not the gb_init singleton).
        if self._tel_owned:
            self.tel.stop()
        if self._feeder_block_client is not None:
            self._feeder_block_client.close()
            self._feeder_block_client = None
        self.tm.close()
        self.pools.trim_all()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
