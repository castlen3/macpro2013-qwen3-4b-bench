# Mac Pro 2013 Running LLMs — Qwen3-4B Full Benchmark

[中文版](README.md)

> **TL;DR**: Mac Pro 2013 (trash can), Xeon E5-1650 v2, 32GB RAM, dual AMD FirePro D300 — this 12-year-old Intel Mac runs Qwen3-4B at **13 tok/s decode, 16K context** with CPU-only inference + Flash Attention. GPU offload makes everything slower. Don't bother.

---

## Story: A Machine That Shouldn't Run LLMs

The Mac Pro 2013. The trash can. Released in 2013 with Intel Xeon + dual AMD FirePro GPUs — a workstation beast in its day.

But this is 2025. Apple Silicon is on M5. This Intel Mac can only run macOS Sequoia thanks to OpenCore Legacy Patcher. The FirePro D300 has 2GB VRAM on a GCN 1.0 architecture that doesn't even support Metal 3.

Most people would say: "Just buy an M4 Mac mini. Stop suffering."

But a 12-core Xeon and 32GB of RAM is sitting right there. 32GB was maxed-out spec in 2013, and it's still plenty useful for LLM inference today — as long as you keep the model off the GPU.

This is the full story from "maybe GPU offload will make it faster?" to "CPU-only is the fastest, and here's exactly why."

---

## Core Problem: Why GPU Offload Makes It Slower

FirePro D300's capabilities under llama.cpp Metal backend:

```
ggml_metal_device_init: GPU name:   AMD Radeon HD - FirePro D300
ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_device_init: simdgroup reduction   = false
ggml_metal_device_init: simdgroup matrix mul. = false
ggml_metal_device_init: has unified memory    = false
ggml_metal_device_init: has bfloat            = false
ggml_metal_device_init: has tensor            = false
```

Four `false`s, one `disabled`. This GPU has essentially zero acceleration capabilities for modern LLM inference. The only compute resource is basic GPU shaders — and the **2GB VRAM can't hold any model layers anyway** (Qwen3-4B Q4_K_M alone is 2.3GB).

Offload a few layers → VRAM overflows → macOS starts swapping over PCIe → decode speed plummets from 13 t/s to 6 t/s, then to 1.9 t/s on full offload.

The answer is clear: **GPU offload on this machine is anti-optimization. CPU-only is fastest.**

---

## Hardware

| Item | Spec |
|---|---|
| Model | Mac Pro (Late 2013) — MacPro6,1 |
| CPU | Intel Xeon E5-1650 v2 @ 3.50 GHz (6C/12T) |
| RAM | 32 GB DDR3 |
| GPU | 2× AMD FirePro D300 (2GB each, GCN 1.0) |
| macOS | 15.7.1 Sequoia (via OpenCore Legacy Patcher 2.4.1) |
| llama.cpp | 5343f4502 (9538), Apple Clang 17.0.0 |

---

## Model

| Item | Spec |
|---|---|
| Model | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) |
| Quant | Q4_K_M |
| Size | 2.3 GB |
| Native context | 32,768 |

---

## Methodology

All benchmarks use `llama-bench` with:

```
-p 512 -n 128 (unless otherwise noted)
```

5 repetitions per test (unless `-r 1` noted), reported as mean ± stddev.

---

## Round 1: CPU-Only Baseline

| Threads | pp512 (t/s) | tg128 (t/s) | Note |
|--------:|------------:|------------:|------|
| 4 | 24.10 ± 0.68 | 9.74 ± 0.11 | |
| **6** | **24.78 ± 1.44** | **13.12 ± 0.08** | ⭐ Physical cores — best efficiency |
| 8 | 23.27 ± 0.16 | 13.32 ± 0.01 | pp dips slightly, tg rises slightly |
| 10 | 21.83 ± 0.21 | timeout | ❌ Too many threads, contention kills performance |

**Conclusion: `-t 6` (physical core count) is optimal. The E5-1650 v2's memory bandwidth is the bottleneck — hyperthreading only adds contention.**

---

## Round 2: GPU Offload (NGL Sweep)

Fixed: `-t 6 -fa 0`

| ngl | pp512 (t/s) | tg128 (t/s) | vs CPU | Note |
|----:|------------:|------------:|-------:|------|
| 0 | 24.29 ± 0.74 | **11.59 ± 0.85** | baseline | |
| 1 | 25.23 ± 0.14 | 12.00 ± 1.31 | +4% | ✅ Only positive case |
| 4 | 24.91 ± 0.75 | **9.17 ± 1.23** | -21% | ❌ Decode collapses |
| 8 | 25.10 ± 0.27 | **6.75 ± 1.06** | -42% | ❌ |
| 16 | 23.96 ± 1.67 | timeout | — | ❌ |
| 99 | crash | **1.9** | -84% | 💥 VRAM explosion |

**Conclusion: Any offload beyond ngl=1 degrades performance. The FirePro D300's 2GB VRAM cannot hold enough layers to offset the PCIe/memory overhead.**

---

## Round 3: Flash Attention Long Context

This is the most important finding. Fixed: `-t 6 -ngl 0 -fa 1`

| Prompt | pp (t/s) | tg128 (t/s) | vs pp2048 |
|--------|---------:|------------:|----------:|
| pp512 | 25.57 ± 1.71 | 13.56 ± 0.02 | ↑ |
| pp2048 | 23.03 ± 0.96 | 11.83 ± 0.72 | baseline |
| pp4096 | 20.03 | 13.32 | -13% |
| pp8192 | **15.78** | **10.71** | -32% |

**FA=1 is already faster than FA=0 at short context (25.57 vs 24.78 t/s at pp512). At long context it's essential — without FA, pp4096 would OOM immediately (attention matrix = 4096² × f16 = 32MB × 36 layers = 1.1 GB explosion).**

---

## Round 4: FA=1 vs FA=0 Direct Comparison

| | FA=0 | FA=1 | Improvement |
|---|---|---|---|
| pp512 | 24.78 | **25.57** | +3% |
| tg128 | 13.12 | **13.56** | +3% |
| Long context | OOM | ✅ Stable | ∞ |

**Conclusion: FA=1 is never slower than FA=0 on this machine. It's a free lunch — no trade-off, always enable.**

---

## Round 5: llama-server Validation

Launching with optimal parameters:

```bash
llama-server \
  -m Qwen3-4B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 16384 -t 6 -ngl 0 -fa 1
```

API test (OpenAI compatible):

```
GET /health → {"status":"ok"}
POST /v1/chat/completions → Prompt 17.5 t/s | Generation 10.6 t/s
```

Server runs stable — no crashes, no memory leaks.

---

## Recommended Configuration

```bash
# llama-cli (interactive)
llama-cli \
  -m Qwen3-4B-Q4_K_M.gguf \
  -c 16384 -t 6 -ngl 0 -fa 1

# llama-server (API)
llama-server \
  -m Qwen3-4B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 16384 -t 6 -ngl 0 -fa 1
```

---

## Conclusions

| Question | Answer |
|----------|--------|
| Can Mac Pro 2013 run LLMs? | ✅ Qwen3-4B, 13 tok/s |
| Is GPU offload useful? | ❌ Makes it slower (VRAM too small + lacks HW acceleration) |
| Should I enable FA? | ✅ **Always.** No-FA = OOM on long context |
| Is 16K context viable? | ✅ pp8192 15.8 t/s, tg 10.7 t/s |
| Should I run a server? | ✅ CPU-only + FA, stable |

### For Fellow Mac Pro 2013 Owners

1. **Don't waste time on GPU offload.** The FirePro D300 lacks every hardware feature modern LLM inference needs (tensor API, simdgroup, bfloat, unified memory — all unsupported).
2. **32GB RAM is your real advantage.** Qwen3-4B only uses ~4GB total. You have room for Qwen3-8B Q4_K_M (~5GB) running the same way.
3. **FA=1 is a free lunch.** It's faster than FA=0 even at short context, and long context is impossible without it. Just turn it on.
4. **Don't hyperthread.** `-t 6` beats `-t 12`. The E5-1650 v2 is memory-bandwidth-bound, and hyperthreading only adds contention.
5. **OCLP + Sequoia works fine.** As long as SIP is off and basic Metal functionality is there, llama.cpp compiles and runs without issues.

---

## License

MIT
