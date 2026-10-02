# Performance and bug audit — 2 October 2026

This is the original attention-kernel audit. Subsequent weight-residency, prompt-batching and full-model measurements are in [execution-port.md](execution-port.md).

The attention kernel previously recomputed each query/key dot product once for every output component. It now computes scores cooperatively, shares scores and softmax values within a 64-component output tile, and retains the 2048-cell context limit. The same CUDA source compiles for both backends; no vendored WebCuda compiler/runtime files changed.

The benchmark uses 24 query heads, 2 KV heads and 256 components per head, matching the production attention geometry. Inputs are deterministic synthetic arrays. Values below are median warm dispatch-plus-completion wall times, excluding buffer upload, readback and the scalar correctness oracle. WebGPU uses seven batches of five dispatches; four-thread WASM uses five single-dispatch samples. The model download was active during measurement.

| Backend | Context cells | Before | After | Ratio |
|---|---:|---:|---:|---:|
| WebGPU | 256 | 3.027 ms | 0.727 ms | 4.16× |
| WebGPU | 1024 | 11.205 ms | 0.741 ms | 15.12× |
| WebGPU | 2048 | 22.232 ms | 1.145 ms | 19.42× |
| WASM, 4 threads | 256 | 38.154 ms | 1.147 ms | 33.26× |
| WASM, 4 threads | 1024 | 156.281 ms | 4.725 ms | 33.08× |
| WASM, 4 threads | 2048 | 306.318 ms | 10.318 ms | 29.69× |

Hardware: AMD Ryzen 7 9800X3D and NVIDIA RTX 5080; Microsoft Edge reported the `nvidia / blackwell` adapter. These are kernel measurements, not full-model generation throughput. Short GPU measurements vary with dispatch/completion overhead; the raw report includes minimum and maximum samples. Both versions were checked against independent F64 attention equations before timing. Every attention benchmark's maximum absolute error was below 4.1e-8.

`reports/attention-performance.json` records timings, hardware, and hashes of the before/after WASM and GPU artifacts. Reproduce with `node scripts/benchmark-attention.mjs <baseline-directory>`, where the baseline contains the old `strata.mjs`, `strata.wasm`, `strata.abi.json` and `strata_attention.json`. The local pre-change artifacts are in `.local/attention-before/` and are excluded from the distribution.

Additional changes remove work and fix reproduced failures:

- Prompt-only steps skip the vocabulary projection, argmax and logits readback. A regression checks that the output weights are not loaded; numerical conformance confirms the final prompt prediction is unchanged.
- Weight eviction occurs before replacement allocation. Partial uploads are cleaned up, and oversize tiles do not exceed the cache's retained-byte budget.
- GPU expert selection updates between tokens and keeps stable ties by frequency. A repeated multi-layer scan with two slots now schedules recurring experts on the GPU instead of continually evicting them before reuse.
- Hybrid failures wait for both execution lanes before unlocking the session. Reset, checkpoint and restore exclude concurrent decode; disposed sessions cannot restart.
- Worker crashes reject pending and subsequent RPCs, failed initialization terminates its worker, and message-cloning failures do not leak pending requests.
- The logits chart handles 248,320 entries without spreading them into function arguments; at most 120 peak-preserving bars include the selected token.
- Model validation rejects fractional reads/layer indices, incompatible codebooks, overflowing code biases and non-finite raw PLE scales.
- Numeric state/output allocations avoid allocating and copying an extra host array of zeros.

Nine new orchestration regressions were run before their fixes and all failed as expected. The final unit suite and actual WebGPU/WASM/hybrid conformance suite pass; see `validation.md`. New attention tests cover dimension/workgroup tails and the full 2048-cell budget. One test-oracle issue found during this work was also corrected: mapping floating-point scores over `Int32Array` cell IDs had truncated the scores; the oracle now constructs a floating-point JavaScript array.
