# CUDA ownership and performance — v0.3.0

For the later fixed 32-token generation test and native Strata comparison, see [upstream-performance.md](upstream-performance.md). Its longer expert working set exposes a larger performance gap than the short repeated sequence measured below.

The live Q2 model path now performs embedding decoding, expert half-scale expansion, IQ4_NL PLE decoding and RoPE frequency/table generation in `kernels/strata.cu`. The same source contains **49 kernels**, compiled to both WebGPU and threaded WebAssembly. Its SHA-256 is `ecad731efa5c67b228efebf4331570e977cc63be512115c5c9722ae99a7cc7b5`.

WebGPU combines expert gate/up projections and SiLU into one kernel, then writes the down projection directly into the selected MoE output slots. Prompt row mappings remove intermediate gather/scatter buffers. WASM retains separate CUDA projection kernels and expands each expert scale plane once with a CUDA kernel.

JavaScript still owns the browser interface, tokenizer and chat template, file access and validation, n-gram file-row addressing, buffer lifetime, placement policy and dispatch order. Some loader/reference conversion helpers also remain in JavaScript. Moving these host responsibilities into a `.cu` file would not itself improve GPU execution. The inference tensor arithmetic is compiled from CUDA source.

## Controlled GPU comparison

Hardware: RTX 5080, Ryzen 7 9800X3D, 32 GiB RAM, installed Microsoft Edge. Model: the verified Qwen3.8-Flash-Next GSQ-RCO Q2_0 pack. Both builds use an 8 GiB GPU weight budget and consume `760, 6511, 314, 9338, 369, 11751`: five prompt tokens followed by one teacher-forced decode, predicting ` Paris.`.

The control was extracted from the verified v0.2.0 release archive into an isolated directory. Only benchmark instrumentation was updated to collect five warm repetitions; the release's application and generated kernels were unchanged. Earlier runs made while other GPU allocations occupied about 8 GiB were excluded from this comparison.

| Measurement | v0.2.0 control | v0.3.0 | Change |
|---|---:|---:|---:|
| First run, empty application caches | 25.060 s | **19.312 s** | **22.9% less time** |
| Warm repeat, median of five | 570.910 ms | **468.040 ms** | **18.0% less time** |
| Warm five-token prompt, median | 360.825 ms | **282.295 ms** | **21.8% less time** |
| Warm next-token decode, median | 209.760 ms | **185.505 ms** | **11.6% less time** |
| Source reads, first run | 5,035 | **2,295** | **54.4% fewer** |
| GPU dispatches, first run | 16,386 | **9,342** | **43.0% fewer** |
| GPU submissions, first run | 245 | **196** | **20.0% fewer** |

Every warm repeat preserved its logits exactly and read **zero bytes from model source files**, including embeddings and PLE. The first run retains 4,339,198,220 bytes of dense weights and 2,319,667,200 bytes of experts, unchanged from the control. Fewer read calls come from coalesced weight planes, up to four parallel expert reads and a bounded 128 MiB dense-layer window. Bulk windows read slightly more data overall: 8,162,188,480 bytes versus 8,083,532,992. The localhost server uses 1 MiB streaming chunks.

These are local measurements on one short sequence. The OS file cache was not flushed, and browser creation and initial pipeline compilation precede the measured interval. Warm timings describe repeated text with retained weights; new text may select uncached experts. Overlapping source-read durations are not an additive wall-clock breakdown. The earlier 316-second streaming implementation is documented separately in [execution-port.md](execution-port.md).

## Backend and correctness checks

The final backend timings and all-logit comparisons are recorded in [cuda-performance.json](cuda-performance.json). All three backends predict the same two tokens and match all **288 ordered routing rows** of the v0.2.0 control. The new GPU path differs from the control by at most **7.153e-6** across all 248,320 final vocabulary logits, with RMS **1.096e-6**, due to CUDA-generated RoPE trigonometry. This change is numerically small on the measured sequence; it is not bit parity or a model-quality evaluation.

| Backend | First six-token run | Final-logit maximum difference from current GPU | Ordered routes matching control |
|---|---:|---:|---:|
| WebGPU, 8 GiB weight budget | 19.312 s | 0 | 288 / 288 |
| Threaded WASM, four workers | 66.904 s | 6.438e-6 | 288 / 288 |
| Hybrid, four CPU workers | 18.784 s | 6.974e-6 | 288 / 288 |

These are individual runs; the small GPU/hybrid difference is not evidence of a general hybrid speed advantage. Hybrid exercised 2,726 expert evaluations on GPU and 154 on WASM. The final CPU run is below the historical v0.2.0 measurement of 79.426 seconds, but only the GPU comparison above uses a fresh isolated release control.

During CPU validation, bulk weight slices exposed a structured-clone problem: sending a small typed-array view to a worker also copied its entire backing layer buffer. The worker adapter now sends a compact copy of each selected view. A regression test verifies both allocation and write messages preserve values and offsets while transmitting only the selected bytes. Intermediate WASM runs with the oversized copies are diagnostic results and do not describe the released path.

Validation includes **44 unit tests**, **41 WebGPU**, **39 browser WASM**, **41 hybrid** conformance checks and **39 Node WASM** checks. New independent equations cover mapped fused experts, untouched destination padding, packed scale expansion, S2/S4/S8 embeddings, IQ4_NL PLE and RoPE positions through 2047. RoPE kernel errors remain below 3.5e-7 on GPU and 1.9e-7 on WASM for the recorded cases. The browser suite also verifies served shader hashes, file loading, desktop/mobile layout, real text chat, context rejection, cancellation and retained weights across a new conversation.

The main-page `Hi` test produced **`Hello!` in 18.229 seconds** with 13 prompt tokens and two reply tokens. A new conversation retained weights and produced **`Hello! How can I help you today` in 13.959 seconds** with eight reply tokens. The longer response loaded additional experts, so its time is not comparable to the repeated fixed-sequence warm benchmark. Mid-layer cancellation completed in about 0.50 seconds and the next conversation succeeded. These are short functional checks, not broad chat-quality evaluation.

Native Strata 0.1.34 remains a separate numerical contract. Comparing the final GPU logits against the saved native run gives RMS **0.151689**, maximum absolute error **0.939795**, and **144/288** exact ordered routing rows in native mode. Native parity, broader model quality, contexts above 2048, IQ2/IQ3 expert formats and speculative decoding remain outside the validated scope. See [validation.md](validation.md) and [native-comparison.json](native-comparison.json).

Reproduce the current measurements with `npm run benchmark:model`, then run `node scripts/benchmark-model.mjs --label=cuda-wasm-final --mode=wasm --prefill=true` and `node scripts/benchmark-model.mjs --label=cuda-hybrid --mode=hybrid --resident=8 --prefill=true`. The original control requires the v0.2.0 release files; turning off residency in the new engine does not recreate the previous engine.
