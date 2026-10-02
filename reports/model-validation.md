# Real Qwen model validation — 2 October 2026

This records the original streaming baseline and first text interface. The later resident-weight implementation, batched prompt processing and native Strata comparison are recorded in [execution-port.md](execution-port.md). The original timings below are historical, not the current execution path.

The downloaded Qwen3.8-Flash-Next GSQ-RCO Q2_0 model was converted to a canonical pack and executed through the real browser file picker. Inference uses the 21 kernels compiled from `kernels/strata.cu`. The three initial raw-completion runs use Python only to tokenize/decode; the separate main-page chat test performs all text I/O in the browser.

## Source and conversion integrity

- Strata source: `1678de333d0e0711bc414ad992b640e1a37dd814`.
- CUDA WebShader source: `ef46ff1bf02a306bad94ddc18286d25d3d902c14`.
- CUDA kernel source SHA-256: `caadb9622f18afaa6dd7ca72fe6a6998711d503a983196951439c29a31524652`.
- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, Q2_0 shards.
- Original GGUF bytes: 66,423,878,624. Both SHA-256 values were re-read and matched the download's published LFS hashes.
- Canonical `.bin` bytes: 40,293,137,920. The original 28,800,138,432-byte IQ4_NL PLE shard is additionally needed and is hard-linked beside the pack without duplicating its data.

The unchanged upstream Python pack builder emitted all 48 layers, 512 experts per layer, plus dense/embedding weights. The bounded verifier compared **every one of 4,947,698,560 dense values across 1,079 tensors** with upstream reference dequantization, including FP32 bit patterns. There were zero value mismatches and zero signed-zero differences. It also checked **every code and scale byte of all 24,576 experts**, comparing 33,973,862,400 bytes across 73,728 gate/up/down role checks. No bytes differed. This was exhaustive verification rather than sampling.

`models/Qwen3.8-Flash-Next-WebCuda-Q2_0/VERIFIED.json` records the complete result and source/output hashes. The readback and hashing pass took 223.65 seconds. Its synthetic corruption check confirms a changed expert code is detected. The extracted upstream tokenizer passed its 16-string round-trip corpus; the command-line adapter also round-tripped `Café 🙂 العربية 中文` with UTF-8 input/output.

## Real inference

Each run starts a fresh browser page and model session, loads the same pack through the file picker, and executes five prompt tokens plus one autoregressive token through all 48 layers. Geometry is width 2,560, 512 routed experts per layer, top-10 routing, 36 GDN layers, 12 QSA layers, 248,320 vocabulary entries, and 256 allocated context cells. Context occupied during the test was six tokens.

Input text: `The capital of France is`

Input IDs: `760, 6511, 314, 9338, 369`

WebGPU, four-thread WASM and hybrid all generated `11751, 13`, which decodes to ` Paris.`. All 248,320 final logits were finite on every backend. Each run's 288 layer-routing decisions (six steps × 48 layers) selected identical ordered expert IDs.

| Comparison to WebGPU | Maximum absolute logit difference | RMS logit difference |
|---|---:|---:|
| Threaded WASM | 7.867813110351562e-6 | 1.3535694366591968e-6 |
| Hybrid | 7.152557373046875e-6 | 1.1213772711181353e-6 |

Hardware: NVIDIA RTX 5080, AMD Ryzen 7 9800X3D, 32 GB RAM, installed Microsoft Edge. The browser reports `nvidia / blackwell`. WebAssembly uses four CPU threads. These are single short, sequential end-to-end checks with file streaming and bounded caches, not a steady-state benchmark.

| Backend | Complete run | Last autoregressive step | Generated text |
|---|---:|---:|---|
| WebGPU | 154.299 s | 26.609 s | ` Paris.` |
| Threaded WASM, 4 threads | 208.966 s | 37.534 s | ` Paris.` |
| Hybrid | 195.854 s | 33.511 s | ` Paris.` |

Complete-run timing includes backend/model initialization, five prompt steps, one additional autoregressive step, output rendering and final detokenization. The last-step measurement is the engine's dispatch, weight streaming and computation time. File selection and initial prompt tokenization occur before this timer. WebGPU executed 2,880 routed expert calls; WASM executed the same count on CPU. Hybrid executed 128 on GPU and 2,752 on WASM, with 85 promotions into its 48 GPU scheduling slots. Hybrid was slower than GPU-only in this short run. The raw reports retain every step's routing and time. No runtime or page errors were reported; the separate browser regression suite also passed all 19 checks in each mode after the UI changes.

## Reproduce and inspect

```powershell
cd D:\StrataCuda
npm run test:model -- --modes=webgpu,wasm,hybrid
npm run compare:model
```

Per-backend reports and screenshots are `reports/model-<backend>.json` and `.png`. Raw final FP32 logit files are in `.local/model-validation/`, with hashes in the reports. `reports/model-comparison.json` checks matching prompt IDs, generated IDs, pack verification and kernel revision before comparing logits and routed expert order. The locally converted pack is at `D:\StrataCuda\models\Qwen3.8-Flash-Next-WebCuda-Q2_0`; its README explains manual loading and the example token IDs.

## Boundaries

This verifies conversion integrity and short real trained-model completions through the portable inference graph. Agreement between the portable backends is not independent evidence of native Strata model parity. Native BF16/F16 activation rounding, quantized KV, longer-context behavior, broad model quality, arbitrary chat templates and sustained production throughput were not validated. The portable graph intentionally uses F32 activations/reductions/KV and has the implementation limits documented in the root README. The PLE table was preserved as its original hashed GGUF; only the rows needed by the prompts were read during inference.

## Main text interface

`npm run test:chat` exercised the new main page in installed Edge:

- Loaded the downloaded model and tokenizer using the localhost byte-range path, then executed real model layers.
- Rejected a 141-token chat request at a 128-token context limit before adding messages or creating a model session.
- Stopped the real HTTP-backed request at a layer boundary in 1.22 seconds and discarded the partial state.
- Loaded the same model again through the actual folder picker, including its tokenizer subfolder.
- Selected Chat, a **512-token context** and a two-token reply limit, entered `Hi`, and sent with Enter.
- The original Qwen text-only, non-thinking chat format encoded to **13 prompt tokens**. WebGPU consumed those plus one autoregressive token through all 48 layers and generated IDs `9419, 0`, decoded as **`Hello!`**. Token 0 is punctuation in this vocabulary; stopping uses the model's actual EOS IDs.
- The observed run took **393.47 seconds**, stopped at the requested reply limit, and reached position 14 in the actual 512-cell engine. This is a functional run with ordinary development activity, not an isolated throughput benchmark.
- Clearing the conversation removed all visible messages and released its model session. No page, console, HTTP or recorded runtime errors occurred.

The recorded result is `reports/chat.json`, and the real response screenshot is `reports/chat-model.png`. The separate `reports/tokenizer.json` records 208 exact Python-tokenizer comparisons and four exact comparisons to the model's original Jinja chat template. Main-page desktop/mobile screenshots are `reports/chat-desktop.png`, `chat-mobile.png` and `chat-mobile-settings.png`; the original diagnostic interface remains at `/test.html`.
