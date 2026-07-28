import torch
from sgl_kernel import top_k_renorm_prob as top_k_renorm_probs
from sgl_kernel import top_p_renorm_prob as top_p_renorm_probs

from gllm.input_data import InputData
from gllm.layers.repetition_penalty import apply_scaling_penalties


def _fused_top_k_top_p_sample(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Top-k / top-p sampling via sgl_kernel's renorm kernels + torch.multinomial.

    GREENBOOST PATCH (2026-07-28, see NOTICE): sgl-kernel 0.4.x (the
    sglang_kernel distribution) dropped the fused
    `top_k_top_p_sampling_from_probs` convenience function 0.3.x exposed —
    only the renormalization kernels remain (`top_k_renorm_probs`/
    `top_p_renorm_probs`), each explicitly documented in sgl_kernel's own
    sampling.py as "should be equivalent to `top_k_sampling_from_probs`"
    when paired with a separate sampling step. Sequential top-k-then-top-p
    renorm followed by torch.multinomial is the standard composition
    (matches HF transformers' own TopKLogitsWarper -> TopPLogitsWarper
    ordering, and flashinfer's reference implementation sgl_kernel's own
    docstring says it adapts from) — not a novel algorithm, the documented
    equivalent of what the old fused call did.
    """
    probs = probs.float().contiguous()
    probs = top_k_renorm_probs(probs, top_ks.to(torch.int32))
    probs = top_p_renorm_probs(probs, top_ps)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class Sampler:

    def forward_gpu(self, logits: torch.Tensor, input_data: InputData) -> torch.Tensor:
        """Sample on GPU; caller is responsible for D2H."""
        flags = self._get_sampling_flags(input_data)

        if flags["need_repetition_penalty"]:
            apply_scaling_penalties(logits, input_data.repetition_penalty)

        if flags["is_all_greedy"]:
            # argmax is invariant to positive temperature scaling, so the
            # full-vocab div_ would be wasted work here -- skip it.
            return torch.argmax(logits, dim=-1)

        if flags["need_temperature"]:
            logits.div_(input_data.temperature.unsqueeze(1))

        probs = torch.softmax(logits, dim=-1)
        return _fused_top_k_top_p_sample(probs, input_data.top_k, input_data.top_p)

    def forward(self, logits: torch.Tensor, input_data: InputData) -> list[int]:
        return self.forward_gpu(logits, input_data).cpu().tolist()

    @staticmethod
    def _get_sampling_flags(input_data: InputData) -> dict[str, bool]:
        seqs = input_data.seqs
        return {
            "is_all_greedy": all(seq.top_k == 1 for seq in seqs),
            "need_repetition_penalty": getattr(
                input_data, "needs_repetition_penalty", False
            ),
            "need_temperature": any(
                seq.temperature > 1e-5 and abs(seq.temperature - 1.0) > 1e-5
                for seq in seqs
            ),
        }
