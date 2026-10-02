# Validation — 2 October 2026

Source: Strata `1678de333d0e0711bc414ad992b640e1a37dd814`; CUDA WebShader `ef46ff1bf02a306bad94ddc18286d25d3d902c14`. The original WebCuda checkout was not modified.

All 49 kernels compiled from `kernels/strata.cu` to WGSL and to threaded WASM with SIMD. The kernel source hash is `ecad731efa5c67b228efebf4331570e977cc63be512115c5c9722ae99a7cc7b5`; it and each WGSL hash are recorded in `generated/manifest.json`.

The browser suite ran in installed Microsoft Edge on the exposed `nvidia / blackwell` WebGPU adapter. All three backends passed 39 shared conformance checks; GPU and hybrid also passed two GPU residency/storage checks. The independent reference uses separate scalar/F64 equations transcribed from upstream source.

| Execution | Checks | Maximum absolute error, complete fixture logits | Expected token IDs |
|---|---:|---:|---|
| WebGPU | 41 passed | 3.41e-8 or lower | 29, 2, 15, 9, 15 |
| Browser threaded WASM | 39 passed | 3.32e-8 or lower | 29, 2, 15, 9, 15 |
| Hybrid | 41 passed | 3.41e-8 or lower | 29, 2, 15, 9, 15 |

Across all individual arithmetic checks the largest observed absolute error was about 2.12e-6 in the signed S8 projection. The production-size router used 512 experts and selected 10. The GDN test updated a 128 × 48 × 128 state three times. Full decode used a four-layer untrained model with three GDN layers and one QSA layer, PLE before layer 1, two HC streams, six routed experts per layer, a shared expert and the output head.

The hybrid conformance sequence executed 215 expert calls on GPU and 41 on WASM, with 20 promotions. These are execution-path checks, not throughput claims. Node WASM also passed all 39 shared checks with four CPU workers; workgroup counters confirmed real participation by multiple threads.

New CUDA checks additionally cover fused gate/up/SiLU with reversed input mapping, down projection with sparse destination rows and padding sentinels, CUDA-expanded expert scales, S2/S4/S8 embedding decoding and stream replication, unaligned IQ4_NL PLE blocks with signed scales, and RoPE through position 2047 against independent double-precision equations. File tests cover packed scale padding and row ordering, bounded adjacent-plane reads, and four concurrent expert reads with error draining.

Earlier checks exercise all 65,536 packed half-float bit patterns on each backend, all three packed Q2 expert projections with multiple tokens and strided output, prompt chunks of 1/2/5 tokens starting at a nonzero context position, and a decode after each chunk. GPU checks also cover resident tiled dense/expert weights and lossless compact BF16 storage. These compare against the independent F64 fixture, not just against another invocation of the new code.

Attention tests include non-contiguous/reversed cells, 70-component workgroup tails, and 256 components with 2048 selected cells. The optimized kernel remains within the F64 reference tolerance. Prompt-only steps skip the vocabulary projection and preserve the last prompt token's logits. The isolated before/after attention benchmark is recorded in `performance.md` and `attention-performance.json`.

Checkpoint restore reproduced logits exactly on the same backend. Reset matched a fresh scalar session. The browser file-picker test loaded `generated/fixture/manifest.json` and `weights.bin`, consumed prompt IDs `2, 7, 4`, and generated `15, 18, 3, 15`, matching an independent autoregressive reference. Desktop and mobile screenshots were inspected; the mobile page did not overflow horizontally. No browser console, page, GPU runtime or HTTP errors were recorded.

After real-model loading was added, the model picker was corrected to show the loaded vocabulary and context limits. A browser layout check using ten expert chips per layer reproduced mobile horizontal overflow that the two-expert fixture had missed. Routing rows now wrap; the same check passes at 390 pixels. This layout check does not claim mobile full-model execution.

The native source oracle compiled unchanged upstream C++ helpers and matched:

- All 65,536 half-float bit patterns, including signed zero, subnormals, infinities and NaNs.
- 16,384 Q2_0 decoded values across signed scales and varied packed codes.
- 8,192 IQ4_NL decoded values with split-half nibble ordering.
- 80 n-gram row indices, including EOS cutoff, token zero, reversed history and 64-bit overflow cases.

All 44 unit tests pass. File/contract tests cover portable pack roundtrips, F16 scale widths, offset planes, interleaved gate/up expert rows, malformed planes, missing files, out-of-range reads, non-finite weights/PLE scales, codebook contracts, unsupported geometry, GGUF header bounds and HTTP range behavior. Runtime regressions exercise bounded allocation, partial-upload cleanup, oversize tiles, hybrid failure draining, stable expert scheduling, exclusive state operations, worker failures, production-size vocabulary rendering, and skipped prompt output weights. New regressions check complete-expert residency and eviction without evicting dense weights, partial-admission cleanup, separate GPU/WASM budgets, coalesced expert reads, ordered worker command batches and their failures. Text-session regressions check capacity before allocation, EOS, prefix reuse including the last unconsumed token, context changes, mid-layer abort and disposal of partial state, and weight retention across new conversations. HTTP model reads reject ignored ranges and incorrect response bodies. They use synthetic files and controlled failure injection; arithmetic is separately checked on the real backends.

The main text interface is separate from the diagnostic page at `/test.html`. Its JavaScript tokenizer matches the unchanged upstream Python token IDs and round trips in all 208 checks, covering both special-token modes, Unicode whitespace/BOM, combining marks, control bytes, emoji, long CJK and deterministic mixed strings. Four single/multi-turn formats match the model's original Jinja template with thinking disabled, including system instructions. Streaming UTF-8 preserves incomplete characters until their bytes arrive. Desktop and mobile layouts were inspected at 1440 × 1024 and 390 × 844; the main page has no horizontal overflow. The mobile settings drawer, Test lab link and model-required Send gate passed.

Separate real-model validation converts and exhaustively verifies the downloaded Qwen weights, then runs a short text completion through all 48 layers; see `model-validation.md` for the original streaming results. The current resident/batched path was checked on WebGPU, WASM and hybrid with all 248,320 final logits and all 288 routing rows. The v0.2 WebGPU build preserved the streaming baseline exactly. The v0.3 CUDA-generated RoPE tables cause small F32 differences; the current measured errors and route comparisons are in `cuda-performance.md`. The main page also generated `Hello!` and `Hello! How can I help you today`, tested Stop, and retained weights across a new conversation. Desktop and mobile layout checks still pass.

The current performance report, `cuda-performance.md`, and historical execution report, `execution-port.md`, record the actual dense/expert residency, cold and warm performance, and an independent run of native Strata 0.1.34 at the pinned source revision. Native logits and some routing choices differ: native numerical parity has not been achieved. Broader model quality, attention beyond 2048 cells and sustained production throughput remain unvalidated. The weights are excluded from the application distribution. Missing features and the F32 portable numerical contract are stated in the README.
