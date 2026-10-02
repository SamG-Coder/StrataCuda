# Validation — 2 October 2026

Source: Strata `1678de333d0e0711bc414ad992b640e1a37dd814`; CUDA WebShader `ef46ff1bf02a306bad94ddc18286d25d3d902c14`. The original WebCuda checkout was not modified.

All 21 kernels compiled from `kernels/strata.cu` to WGSL and to threaded WASM with SIMD. The kernel source hash and each WGSL hash are recorded in `generated/manifest.json`.

The browser suite ran in installed Microsoft Edge on the exposed `nvidia / blackwell` WebGPU adapter. It compared the same 19 conformance checks in GPU-only, WASM-only and hybrid execution. The independent reference uses separate scalar/F64 equations transcribed from upstream source.

| Execution | Checks | Maximum absolute error, complete fixture logits | Expected token IDs |
|---|---:|---:|---|
| WebGPU | 19 passed | 3.41e-8 or lower | 29, 2, 15, 9, 15 |
| Browser threaded WASM | 19 passed | 3.32e-8 or lower | 29, 2, 15, 9, 15 |
| Hybrid | 19 passed | 3.41e-8 or lower | 29, 2, 15, 9, 15 |

Across all individual arithmetic checks the largest observed absolute error was about 2.12e-6 in the signed S8 projection. The production-size router used 512 experts and selected 10. The GDN test updated a 128 × 48 × 128 state three times. Full decode used a four-layer untrained model with three GDN layers and one QSA layer, PLE before layer 1, two HC streams, six routed experts per layer, a shared expert and the output head.

The hybrid conformance sequence executed 49 expert calls on GPU and 39 on WASM, with 19 promotions. These are execution-path checks, not throughput claims. Node WASM also passed all 19 checks with four CPU workers; workgroup counters confirmed real participation by multiple threads.

Attention tests include non-contiguous/reversed cells, 70-component workgroup tails, and 256 components with 2048 selected cells. The optimized kernel remains within the F64 reference tolerance. Prompt-only steps skip the vocabulary projection and preserve the last prompt token's logits. The isolated before/after attention benchmark is recorded in `performance.md` and `attention-performance.json`.

Checkpoint restore reproduced logits exactly on the same backend. Reset matched a fresh scalar session. The browser file-picker test loaded `generated/fixture/manifest.json` and `weights.bin`, consumed prompt IDs `2, 7, 4`, and generated `15, 18, 3, 15`, matching an independent autoregressive reference. Desktop and mobile screenshots were inspected; the mobile page did not overflow horizontally. No browser console, page, GPU runtime or HTTP errors were recorded.

After real-model loading was added, the model picker was corrected to show the loaded vocabulary and context limits. A browser layout check using ten expert chips per layer reproduced mobile horizontal overflow that the two-expert fixture had missed. Routing rows now wrap; the same check passes at 390 pixels. This layout check does not claim mobile full-model execution.

The native source oracle compiled unchanged upstream C++ helpers and matched:

- All 65,536 half-float bit patterns, including signed zero, subnormals, infinities and NaNs.
- 16,384 Q2_0 decoded values across signed scales and varied packed codes.
- 8,192 IQ4_NL decoded values with split-half nibble ordering.
- 80 n-gram row indices, including EOS cutoff, token zero, reversed history and 64-bit overflow cases.

All 34 unit tests pass. File/contract tests cover portable pack roundtrips, F16 scale widths, offset planes, interleaved gate/up expert rows, malformed planes, missing files, out-of-range reads, non-finite weights/PLE scales, codebook contracts, unsupported geometry, GGUF header bounds and HTTP range behavior. Runtime regressions exercise bounded allocation, partial-upload cleanup, oversize tiles, hybrid failure draining, stable expert scheduling, exclusive state operations, worker failures, production-size vocabulary rendering, and skipped prompt output weights. Text-session regressions check capacity before allocation, EOS, prefix reuse including the last unconsumed token, context changes, mid-layer abort and disposal of partial state. HTTP model reads reject ignored ranges and incorrect response bodies. They use synthetic files and controlled failure injection; arithmetic is separately checked on the real backends.

The main text interface is separate from the diagnostic page at `/test.html`. Its JavaScript tokenizer matches the unchanged upstream Python token IDs and round trips in all 208 checks, covering both special-token modes, Unicode whitespace/BOM, combining marks, control bytes, emoji, long CJK and deterministic mixed strings. Four single/multi-turn formats match the model's original Jinja template with thinking disabled, including system instructions. Streaming UTF-8 preserves incomplete characters until their bytes arrive. Desktop and mobile layouts were inspected at 1440 × 1024 and 390 × 844; the main page has no horizontal overflow. The mobile settings drawer, Test lab link and model-required Send gate passed.

The checks above use the untrained fixture. Separate real-model validation now converts and exhaustively verifies the downloaded Qwen weights, then runs a short text completion through all 48 layers; see `model-validation.md`. The weights are excluded from the application distribution. Native Strata logit parity, full-model weight residency in device memory, broader model quality, attention beyond 2048 cells and sustained production throughput remain untested. The F32 portable numerical contract differs from native activation/KV quantization. Missing features are listed in the repository README.
