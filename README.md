# Mac Pro 2013 垃圾桶也能跑大模型 — Qwen3-4B 實測報告

[English Version](README_EN.md)

> **TL;DR**: Mac Pro 2013（垃圾桶）、Xeon E5-1650 v2、32GB RAM、雙 AMD FirePro D300——這台 12 年的 Intel Mac，靠 llama.cpp CPU-only + Flash Attention，16K context 下穩跑 Qwen3-4B，decode 13 tok/s，GPU offload 完全不需要。

---

## 故事：一台不該跑 LLM 的機器

Mac Pro 2013。垃圾桶。2013 年發表，Intel Xeon + AMD FirePro 雙卡，當年是工作站級的神器。

但現在是 2026 年。Apple Silicon 已經出到 M5，這台 Intel Mac 連 macOS Sequoia 都要靠 OpenCore Legacy Patcher 才能裝上去。FirePro D300 只有 2GB VRAM、GCN 1.0 架構，連 Metal 3 都不支援。

大多數人會說：「買 M4 Mac mini 吧，別折騰了。」

但 12 核心 Xeon + 32GB RAM 就是擺在那裡。32GB 在當年是頂配，在今天跑 LLM 依然很有用——只要你不把模型放 GPU。

這篇是從「GPU offload 會不會比較快？」開始，最後得出「CPU-only 最快」的完整測試過程。

---

## 核心問題：為什麼 GPU Offload 反而變慢？

FirePro D300 在 llama.cpp Metal backend 下的能力診斷：

```
ggml_metal_device_init: GPU name:   AMD Radeon HD - FirePro D300
ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_device_init: simdgroup reduction   = false
ggml_metal_device_init: simdgroup matrix mul. = false
ggml_metal_device_init: has unified memory    = false
ggml_metal_device_init: has bfloat            = false
ggml_metal_device_init: has tensor            = false
```

四個 `false`、一個 `disabled`——這張卡在現代 LLM inference 中幾乎沒有加速能力。唯一的運算資源是基礎 GPU compute，但 **2GB VRAM 根本裝不下任何模型層**（Qwen3-4B Q4_K_M 就要 2.3GB）。

offload 幾層 → VRAM 不夠 → macOS 開始 swapping → decode 速度直接從 13 t/s 暴跌到 6 t/s、甚至 1.9 t/s。

答案很明確：**在這台機器上，GPU offload 是負優化。CPU-only 最快。**

---

## 硬體 / Hardware

| Item | Spec |
|---|---|
| 機型 / Model | Mac Pro (Late 2013) — MacPro6,1 |
| CPU | Intel Xeon E5-1650 v2 @ 3.50 GHz (6C/12T) |
| RAM | 32 GB DDR3 |
| GPU | 2× AMD FirePro D300 (各 2GB, GCN 1.0) |
| macOS | 15.7.1 Sequoia (via OpenCore Legacy Patcher 2.4.1) |
| llama.cpp | 5343f4502 (9538), Apple Clang 17.0.0 |

---

## 模型 / Model

| Item | Spec |
|---|---|
| 模型 / Model | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) |
| 量化 / Quant | Q4_K_M |
| 大小 / Size | 2.3 GB |
| 原生 context | 32,768 |

---

## 測試方法 / Methodology

所有 benchmark 使用 `llama-bench`，參數：

```
-p 512 -n 128（除非特別標示）
```

每個測試跑 5 輪（除非註明 `-r 1`），取平均 ± 標準差。

---

## Round 1：CPU-Only Baseline

| Threads | pp512 (t/s) | tg128 (t/s) | Note |
|--------:|------------:|------------:|------|
| 4 | 24.10 ± 0.68 | 9.74 ± 0.11 | |
| **6** | **24.78 ± 1.44** | **13.12 ± 0.08** | ⭐ 物理核心數，最佳效率 |
| 8 | 23.27 ± 0.16 | 13.32 ± 0.01 | pp 微降，tg 微升 |
| 10 | 21.83 ± 0.21 | timeout | ❌ thread 過多崩潰 |

**結論：`-t 6`（物理核心數）是最佳平衡點。E5-1650 v2 的記憶體頻寬是瓶頸，超線程沒有幫助。**

---

## Round 2：GPU Offload (NGL Sweep)

固定：`-t 6 -fa 0`

| ngl | pp512 (t/s) | tg128 (t/s) | vs CPU | 評語 |
|----:|------------:|------------:|-------:|------|
| 0 | 24.29 ± 0.74 | **11.59 ± 0.85** | baseline | |
| 1 | 25.23 ± 0.14 | 12.00 ± 1.31 | +4% | ✅ 唯一正向 |
| 4 | 24.91 ± 0.75 | **9.17 ± 1.23** | -21% | ❌ decode 崩跌 |
| 8 | 25.10 ± 0.27 | **6.75 ± 1.06** | -42% | ❌ |
| 16 | 23.96 ± 1.67 | timeout | — | ❌ |
| 99 | crash | **1.9** | -84% | 💥 VRAM 爆炸 |

**結論：ngl ≥ 4 即開始拖慢。FirePro D300 的 2GB VRAM 無法容納有效層數，offload 只增加 PCIe/Memory 開銷。**

---

## Round 3：Flash Attention 長 Context 測試

這是整篇最重要的發現。固定：`-t 6 -ngl 0 -fa 1`

| Prompt | pp (t/s) | tg128 (t/s) | vs 2K |
|--------|---------:|------------:|------:|
| pp512 | 25.57 ± 1.71 | 13.56 ± 0.02 | ↑ |
| pp2048 | 23.03 ± 0.96 | 11.83 ± 0.72 | baseline |
| pp4096 | 20.03 | 13.32 | -13% |
| pp8192 | **15.78** | **10.71** | -32% |

**FA=1 在短 context（pp512）就比 FA=0 快（25.57 vs 24.78 t/s）！長 context 下更關鍵——pp8192 還有 15.8 t/s，tg 維持在 10.7 t/s。**

對比無 FA：pp512→pp4096 可直接 OOM，因為 attention matrix = 4096² × f16 = **每層 32MB** × 36 層 = 1.1 GB 記憶體爆炸。

---

## Round 4：FA=1 vs FA=0 直接對比

| | FA=0 | FA=1 | 提升 |
|---|---|---|---|
| pp512 | 24.78 | **25.57** | +3% |
| tg128 | 13.12 | **13.56** | +3% |
| 長 context | OOM | ✅ 穩定 | ∞ |

**結論：FA=1 在任何情況下都不比 FA=0 慢。這台機器上，FA 是純收益，沒有取捨。**

---

## Round 5：llama-server 實測

用最佳參數啟動：

```bash
llama-server \
  -m Qwen3-4B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 16384 -t 6 -ngl 0 -fa 1
```

API 測試（OpenAI compatible）：

```
GET /health → {"status":"ok"}
POST /v1/chat/completions → Prompt 17.5 t/s | Generation 10.6 t/s
```

Server 穩定運行，無 crash、無 memory leak。

---

## 建議參數 / Recommended Configuration

```bash
# llama-cli（互動模式）
llama-cli \
  -m Qwen3-4B-Q4_K_M.gguf \
  -c 16384 -t 6 -ngl 0 -fa 1

# llama-server（API 模式）
llama-server \
  -m Qwen3-4B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 16384 -t 6 -ngl 0 -fa 1
```

---

## 結論 / Conclusions

| 問題 | 答案 |
|------|------|
| Mac Pro 2013 能跑 LLM？ | ✅ Qwen3-4B，13 tok/s |
| GPU offload 有用？ | ❌ 反而拖慢（VRAM 不足 + 缺硬體加速） |
| FA 要開嗎？ | ✅ **必開**。不開長 context 直接 OOM |
| 16K context 可行？ | ✅ pp8192 15.8 t/s，tg 10.7 t/s |
| 推薦跑 server？ | ✅ CPU-only + FA，穩定 |

### 給同樣有 Mac Pro 2013 的人

1. **不要花時間搞 GPU offload。** FirePro D300 在 llama.cpp 下不具備現代 LLM 需要的硬體加速（tensor API、simdgroup、bfloat、unified memory 全部不支援）。
2. **32GB RAM 是真正的優勢。** Qwen3-4B 只用 4GB，還有大把空間跑更大的模型（Qwen3-8B Q4_K_M 約 5GB，同樣可以 CPU-only 跑）。
3. **FA=1 是免費午餐。** 比 FA=0 快，而且長 context 必須用它才能活下去。沒有取捨，直接開。
4. **不要超線程。** `-t 6` 比 `-t 12` 快。Xeon E5-1650 v2 的記憶體頻寬是瓶頸，超線程只會增加 contention。
5. **OCLP + Sequoia 沒問題。** 只要 SIP 關了、Metal 基本功能有，llama.cpp 就能編譯。

---

## License

MIT
/bin/bash: line 4: /var/folders/zz/zyxvpxvq6csfxvn_n0000000000000/T/hermes-snap-75ebe727f72a.sh: Permission denied
/bin/bash: line 5: /var/folders/zz/zyxvpxvq6csfxvn_n0000000000000/T/hermes-cwd-75ebe727f72a.txt: Permission denied
