# Native upstream performance — 2 October 2026

On this machine, unchanged native Strata **v0.1.34** generated a fixed 32-token sequence at a median **28.8 tokens/s**, versus **1.62–1.71 tokens/s** for warmed WebCuda v0.3.0. The generation-time ratio is **17.29×** on this workload. Both engines generated the same 32 token IDs in every measured run; this is still not numerical parity, which is separately assessed in [native-comparison.json](native-comparison.json).

## Machine and controls

- NVIDIA GeForce RTX 5080, 16,303 MiB VRAM; driver 616.64.
- AMD Ryzen 7 9800X3D, eight physical cores, 32 GiB RAM.
- Verified Qwen3.8-Flash-Next GSQ-RCO Q2_0 weights, the same canonical pack and original native/PLE shards as the port validation.
- Native executable version 0.1.34, source `1678de333d0e0711bc414ad992b640e1a37dd814`, CUDA 13.0 release build with SM120 support. Executable SHA-256: `f0838beeb5b630483262456b1d0a596c76c7e19633bbb58eebe27ab86b497bb4`.
- Native flags include `--native`, `--mmap-experts`, `--pool-workers 4`, the PLE shard and a 256-token context. The pool has four workers plus the host thread. The complete commands are in the JSON reports. Mapped experts avoid requiring the entire 31.6 GiB expert file in a resident RAM arena on this 32 GiB machine.
- WebCuda uses installed Edge, the released 49-kernel source, a 256-token context and its default 8 GiB GPU weight budget. Benchmarks run sequentially, without simultaneous native and browser inference.

All runs use input IDs `760,6511,314,9338,369,11751`, encoding `The capital of France is Paris`. Five positions are prompt-only; generation starts from the sixth position. The 32-token test runs to a fixed length, including past the model's end-of-turn token. It measures token execution rather than the length of a naturally stopped chat reply. Greedy sampling is used with no MTP or other speculative decoding.

The OS file cache was retained. Each native sample launches a fresh process; three samples are recorded per configuration. WebCuda records one run with empty application weight caches, followed by two state resets retaining weight caches. The 8 GiB limit still applies to retained weights, so a reset does not imply that every needed expert remains resident.

## Results

| Configuration | Generation time for 32 tokens | Generation rate | Prompt-only time | Time to first generated token |
|---|---:|---:|---:|---:|
| Native, no GPU expert cache; median of three | **1.111 s** | **28.80 tokens/s** | 0.239 s | 0.284 s |
| Native, 3,072 cached experts and 16-token prefill chunks; median of three | 1.816 s | 17.62 tokens/s | 0.385 s | 1.210 s |
| WebCuda, initially empty weight caches | 34.131 s | 0.94 tokens/s | 14.744 s | 18.039 s |
| WebCuda, retained weights, repeat 1 | 19.743 s | 1.62 tokens/s | 2.703 s | 3.131 s |
| WebCuda, retained weights, repeat 2 | 18.675 s | 1.71 tokens/s | 2.472 s | 2.875 s |

Native generation rates ranged from **28.66 to 29.60 tokens/s**. Complete native process time, including weight loading, setup, prompt, generation and shutdown, had a median of **3.671 seconds**. The cached variant took **5.489 seconds** for the complete process. The GPU cache configuration is an additional measured case; it did not improve this short test, and this benchmark does not establish its behavior for longer prompts or a preloaded expert profile.

Native's generation and prompt timing begin after weight loading. Its time-to-first-token field also excludes loading and may include later graph preparation. WebCuda loads weights on demand inside the first prompt/decode intervals: its first prompt plus generation took **48.875 seconds**, excluding **2.118 seconds** of browser-side model/backend/session setup. These timing boundaries differ; the 17.29× figure compares native's median generation interval with the median of the two retained-weight WebCuda generation intervals, not total application startup.

For the previous single-output short test, native's median was **34.8 ms** for the final token, **239.9 ms** for the five conditioning tokens and **2.023 seconds** for the complete process. The earlier WebCuda report's 0.468-second warm repeat described the entire repeated six-token sequence. It should not be extrapolated to multi-token reply throughput.

## What the longer run reveals

WebCuda's expert allocation reaches **3,074 resident experts**, using 4,249,497,600 bytes alongside 4,339,198,220 bytes of dense weights. Cumulative evictions rise from **1,267** after the first run to **2,209** and **3,090** after the repeats. The bounded 1 GiB host expert cache is also full. This establishes that the longer sequence exceeds the retained expert working set; the benchmark does not isolate the precise time spent in file reads, admission, host dispatch or shader execution.

Native logs confirm its ordinary configuration captures all 48 layers into a token graph, and uses native CUDA projections with the CPU expert pool. The browser port uses a different execution schedule and portable F32 arithmetic. Profiling would be needed to attribute the full 17× gap to individual implementation choices.

## Evidence and reproduction

- [Native, one generated token](benchmark-upstream-native-1.json)
- [Native, 32 generated tokens](benchmark-upstream-native-32.json)
- [Native cached configuration, 32 tokens](benchmark-upstream-cached-32.json)
- [Browser WebCuda, 32 tokens and retained-weight repeats](benchmark-generation-webgpu.json)
- [Comparison summary](upstream-performance.json)

The original binary stays in `.local/native-upstream/bin`; neither it nor the model weights are included in Git. Raw native logs and their report-recorded hashes are under `.local/native-benchmark` on the test machine.

```powershell
python -B scripts/benchmark-upstream.py --runs 3 --new-tokens 1 --variant native
python -B scripts/benchmark-upstream.py --runs 3 --new-tokens 32 --variant native
python -B scripts/benchmark-upstream.py --runs 3 --new-tokens 32 --variant cached
node scripts/benchmark-generation.mjs
```

The native runner checks the executable version and exact SHA-256 before running. Both runners check the generated sequence is stable across repetitions. No application inference code was changed for these measurements.
