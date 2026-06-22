# Third-Party Notices

GreenBoost is GPL v2. This file lists third-party code, patterns, and
concepts incorporated into GreenBoost, with their original license and
where each is used.

---

## bpf_uvm

- **Author:** Valentin Churavy and contributors
- **Source:** https://github.com/vchuravy/bpf_uvm
- **License:** MIT
- **Used in:** `ebpf/gb_trace.bpf.c`, `ebpf/gb_trace.c`,
  `ebpf/reference/uvm_perf_events_ref.h`

The kprobe → ringbuf → userspace-aggregation pattern for tracing kernel
memory-migration events is adapted from bpf_uvm. bpf_uvm traces NVIDIA's
`nvidia_uvm` driver fault/migration events; GreenBoost's tensor memory is
pinned DDR / DMA-BUF mapped over PCIe and generates no `nvidia_uvm` page
faults, so the pattern is repointed onto GreenBoost's own kernel module
paths (`gb_t3_evict_buf`, `gb_t3_promote_buf`, `gb_auto_evict_cold`,
`gb_alloc_buf`, `gb_pin_user_buf` in `greenboost.c`). The optional
UVM-fault probe (`uvm_perf_event_notify`, attached only if that symbol is
present in `/proc/kallsyms`) keeps the original bpf_uvm struct layout
reference for systems that do use managed/UVM memory outside GreenBoost's
own allocation paths.

MIT License (full text):

```
The MIT License (MIT)

Copyright © 2022: Valentin Churavy, and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## DeepSpeed / ZeRO-Infinity

- **Authors:** Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden
  Smith, Yuxiong He (Microsoft)
- **Source:** https://www.deepspeed.ai/ — "ZeRO-Infinity: Breaking the GPU
  Memory Wall for Extreme Scale Deep Learning" (arXiv:2104.07857)
- **License:** Apache-2.0
- **Used in:** `greenboost_cuda_shim.c` (`prefetch_worker`'s T3→T2 admission
  control, labeled "U5: DeepSpeed-style T3→T2 prefetch admission control")

Concept attribution only — no DeepSpeed code is vendored or copied.
GreenBoost's prefetch admission control (back off when T2 is near its
warn threshold, re-queue rather than push T2 past cap) is inspired by
ZeRO-Infinity's overlap-centric, bandwidth-aware offload design, applied
to GreenBoost's own explicit T1/T2/T3 tiers rather than ZeRO's
GPU/CPU/NVMe training-state partitioning.

---

## ExpertFlow

- **Authors:** Xin He, Shunkang Zhang, Kaijie Tang, Shaohuai Shi, Yuxin
  Wang, Zihao Zeng, Zhenheng Tang, Xiaowen Chu, Haiyan Yin, Ivor W. Tsang,
  Yew Soon Ong
- **Source:** "ExpertFlow: Efficient Mixture-of-Experts Inference via
  Predictive Expert Caching and Token Scheduling", DAC '26 (2026)
- **License:** N/A (paper citation, no code released/vendored)
- **Used in:** `gb_moe.py` (routing-frequency histogram, predictive
  next-block expert prefetch, synchronous promote-on-misprediction
  fallback)

Concept attribution only. `gb_moe.py`'s design (predictive locality-aware
expert caching with a real-time correction fallback) follows the same
shape as ExpertFlow's PLEC + real-time correction, implemented as an
online historical-frequency heuristic rather than ExpertFlow's trained
T5-style routing-path predictor — see `gb_moe.py`'s module docstring for
the documented limitation (no learned router lookahead).
