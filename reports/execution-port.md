# Resident execution and batched prompts — 2 October 2026

This release adds persistent dense weights, real complete-expert GPU/RAM residency, batched prompt processing and batched backend scheduling to the original portable engine. The shared CUDA source now has 41 kernels compiled to both WebGPU and threaded WebAssembly. The real-model GPU result is bit-identical to the original portable baseline on the measured prompt. Native Strata numerical parity remains incomplete; the comparison below records the differences.

Source revisions: Strata `1678de333d0e0711bc414ad992b640e1a37dd814` and CUDA WebShader `ef46ff1bf02a306bad94ddc18286d25d3d902c14`. Strata's `v0.1.34` release points at that same commit. The only changed vendored WebCuda file is its WASM CPU shim, which adds four CUDA bit-cast intrinsics; `PROVENANCE.json` records every vendor hash. The original `D:\cuda-webshader` checkout was not modified.

## What changed

- Dense tensors become resident on first use and remain protected from expert eviction. Lossless BF16 storage reduces the active dense footprint to **4,339,198,220 bytes (4.04 GiB)**. GPU expert allocations retain the original Q2 packed codes and half scales: one **1,382,400-byte** blob contains gate, up and down together.
- The default GPU weight budget is **8 GiB**, with 12 GiB and streaming alternatives. Frequency-based expert admission uses actual residency. GPU admission releases the duplicate host blob; eviction can return the complete blob to a bounded **1 GiB** host cache. The WASM tensor cache remains **64 MiB**, separately constrained by its heap. GPU scratch pooling is capped at 128 MiB. Recurrent state, KV, staging and browser overhead are additional allocations.
- Prompt processing runs layer by layer in chunks of 16 tokens, batching projections, causal attention and routing. Distinct experts process all their selected rows together. Recurrent GDN and PLE histories advance causally within each chunk. In hybrid mode, prompt processing uses the GPU; decode assigns resident experts to GPU and cold experts to WASM workers.
- WebGPU dispatches share command submissions, and WASM workers receive ordered command batches. Scratch buffers and bounded binding caches are reused. Redundant temporary-buffer clears, dense uploads and layer completion waits are removed. Required readbacks for routing and final tokens remain.
- A new conversation resets recurrent/KV state while retaining weights. Changing the model, backend, context or GPU weight budget creates a new session. Stop discards partial recurrent state.

## Real-model measurement

Hardware: RTX 5080 with about 16 GiB VRAM, Ryzen 7 9800X3D, 32 GiB RAM, installed Microsoft Edge; WASM uses four workers. The model is the exhaustively verified Qwen3.8-Flash-Next GSQ-RCO Q2_0 pack described in [model-validation.md](model-validation.md).

Every benchmark consumes the same six teacher-forced IDs: `760, 6511, 314, 9338, 369, 11751`. The first five encode `The capital of France is`; the last is ` Paris`. The measured predictions are `11751` and `13` (` Paris.`). The baseline processes the prompt sequentially; the optimized path batches the first five tokens and then decodes the sixth. All final vocabulary logits and all routing rows are compared.

| Measurement | Original GPU streaming | GPU resident + batched | Ratio |
|---|---:|---:|---:|
| First run: five prompt tokens + one decode | 316.016 s | **26.247 s** | **12.04× faster** |
| Final decode, including newly encountered experts | 51.311 s | **1.230 s** | 41.72× faster |
| Model source bytes read | 35.318 GiB | **7.528 GiB** | 4.69× fewer |
| Source read calls | 179,588 | **5,035** | 35.67× fewer |
| GPU command submissions | 108,364 | **245** | 442.30× fewer |
| GPU kernel dispatches | 108,364 | **16,386** | 6.61× fewer |

These first runs begin with empty application caches. The OS file cache was not flushed, and browser/pipeline creation occurs before the measured interval. These are individual local runs, not physical cold-disk measurements or a sustained tokens-per-second benchmark. Instrumented source-read durations overlap and must not be summed as an additive wall-clock breakdown. Earlier optimized runs varied from about 24 to 32 seconds; the table uses the final recorded run.

Resetting the state and repeating the same six tokens with retained weights took **0.554 s**: **0.346 s** for the five-token prompt and **0.206 s** for the next token. The repeat read **zero dense/expert weight bytes**; its 12 small embedding reads totalled 11,520 bytes. It produced exactly the same logits. That is a warm repeat of one short sequence, not a prediction of performance for new text or a new model. The first run retained 1,678 experts, occupying 2,319,667,200 bytes in addition to dense weights.

| Backend, batched prompt | First run | Final decode | Max final-logit difference from GPU | Ordered routing rows matching baseline |
|---|---:|---:|---:|---:|
| WebGPU, 8 GiB weight budget | 26.247 s | 1.230 s | **0 (bit-identical)** | 288 / 288 |
| Threaded WASM, four workers | 79.426 s | 22.102 s | 0.000007868 | 288 / 288 |
| Hybrid, four CPU workers | 27.049 s | 2.104 s | 0.000006676 | 288 / 288 |

Hybrid executed 2,726 selected expert evaluations on GPU and 154 on WASM. Its prompt uses GPU; cold experts in the subsequent decode demonstrate the CPU lane. WASM/hybrid runs preceded the source file's LF normalization; the recorded source hash `b0119b7b...` differs from the final `5aeaa179...` only by CRLF versus LF. Their arithmetic kernels are identical. The final GPU run and all three current browser conformance runs use `5aeaa179bd3a7aef888e63f112b2a1990d2c9e4b48343a1695cee2c4fbbc31f0`.

The production text interface separately passed a real folder-picker chat test: `Hi` produced **`Hello!`** (13 prompt tokens, two reply tokens) in **26.246 s**, versus the historical streaming run's 393.468 s. A new conversation retained weights and generated **`Hello! How can I help you today`** (eight reply tokens) in **13.671 s**. That longer response introduced 695 additional resident experts; it is not an all-warm generation measurement. Stop returned in 0.632 s in this run, and the incomplete engine state was discarded. Layouts at 1440 × 1024 and 390 × 844 passed, including the mobile settings drawer.

Raw results are in [benchmark-baseline.json](benchmark-baseline.json), [benchmark-release.json](benchmark-release.json), [benchmark-wasm-prefill.json](benchmark-wasm-prefill.json), [benchmark-hybrid-prefill.json](benchmark-hybrid-prefill.json), [execution-comparison.json](execution-comparison.json), [chat.json](chat.json) and [chat-warm.json](chat-warm.json). Full logits are saved locally by the runners under `.local/benchmarks/`; their SHA-256 values are in the JSON reports. All GPU runs have final-logit SHA-256 `88fd69736db3148099c8c4627cc5c30f78c45de5f99255a9e526602fdaacab19`.

## Comparison with upstream native Strata

The independent reference is the published Windows binary from [Strata v0.1.34](https://github.com/Niko1221/Strata/releases/tag/v0.1.34), using the same source model, converted pack, six teacher-forced inputs and 256-cell context. Both canonical-pack execution and the production `--native` path were run. The archive SHA-256 is `20dc548a96803f025a8b12274a1a31fccb80462f47361e10f59eb972243ac523`; the executable SHA-256 is recorded in [native-comparison.json](native-comparison.json), alongside exact commands and output hashes.

| Comparison of portable GPU against upstream | Native canonical pack | Native `--native` |
|---|---:|---:|
| Final predicted token | 13, same as portable | 13, same as portable |
| Final-logit maximum absolute difference | 0.997252 | 0.939795 |
| Final-logit RMS difference | 0.162742 | 0.151689 |
| Cosine similarity | 0.997743 | 0.998034 |
| Identical ordered top-10 expert rows | 81 / 288 | 144 / 288 |
| Identical selected expert sets | 213 / 288 | 244 / 288 |
| Selected expert overlap | 2,802 / 2,880 | 2,834 / 2,880 |

**This is not a native parity pass.** The same final token on a short prompt does not establish equivalent model quality. The portable engine uses F32 activation, reduction and KV storage; upstream uses additional BF16/F16 rounding, quantized activation products and native reduction trees. Those are known execution differences, but this test does not isolate the contribution of each or rule out further bugs. Longer-context sparse selection, IQ2/IQ3 packs, speculative/MTP decoding and the other unsupported upstream features remain listed in the README. Native executable timings exclude its loading phases and are not used as an end-to-end browser speed comparison.

## Reproduce

With the model converted and the application built:

```powershell
npm test
npm run test:wasm
npm run test:browser
npm run benchmark:model
node scripts/benchmark-model.mjs --label=wasm-prefill --mode=wasm --prefill=true
node scripts/benchmark-model.mjs --label=hybrid-prefill --mode=hybrid --resident=8 --prefill=true
npm run test:chat -- --extended
```

The original baseline is commit `875ef67`, measured before development changes. `--resident=0` on the new engine selects streaming storage but still uses the new kernels and scheduler, so it does not recreate that original implementation.

To obtain the independent native reference and run the comparison, install the Python requirements described in the README, then:

```powershell
gh release download v0.1.34 --repo Niko1221/Strata --pattern strata-windows-x64.zip --dir .local/native-upstream
Get-FileHash .local/native-upstream/strata-windows-x64.zip -Algorithm SHA256
Expand-Archive .local/native-upstream/strata-windows-x64.zip -DestinationPath .local/native-upstream/bin
.local/pack-env/Scripts/python.exe -B scripts/compare-native.py
```

Check the archive hash against the value above before running the executable. `compare-native.py` writes the upstream text index alongside the existing pack, without modifying model weights. `--existing` compares previously saved native output; `--exe`, `--pack`, `--portable` and `--label` select alternative local paths. The native binary and trained weights are excluded from this repository and its release archive.

The current validation is 41 unit tests, 29 shared numerical checks on each backend plus two GPU residency/storage checks, real-model cross-backend comparisons, the main-page tests and the separate native comparison. [validation.md](validation.md) records the precise scope. Long-session eviction throughput and broader model-quality evaluation remain future validation work.
