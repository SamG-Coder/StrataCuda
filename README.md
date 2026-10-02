# Strata WebCuda

A browser inference port of [Niko1221/Strata](https://github.com/Niko1221/Strata), using [CUDA WebShader](https://github.com/SamG-Coder/cuda-webshader) to run shared CUDA source on WebGPU and threaded WebAssembly. Public source: [SamG-Coder/StrataCuda](https://github.com/SamG-Coder/StrataCuda).

`kernels/strata.cu` is the single source for **49 compute kernels**. CUDA WebShader compiles it to WGSL for WebGPU and to threaded WebAssembly through Emscripten. JavaScript handles files, buffers, dispatch order, cache policy and the browser interface. Matrix products, routing, recurrent updates, attention and expert arithmetic execute in the compiled kernels. The trained Q2 path also decodes embeddings, expert scales and IQ4_NL PLE rows there, and generates reusable RoPE frequency/position tables in CUDA. GPU execution fuses expert gate/up/SiLU; WASM uses separate CUDA projections to suit its CPU scheduler. JavaScript supplies file ranges, packed row layouts, buffers, geometry and dispatch order; the live trained-model path no longer calculates RoPE trigonometry or expands expert half scales and PLE rows in JavaScript.

This is an initial engine port, **not a complete replacement for upstream Strata**. The application bundles a deterministic, untrained four-layer fixture for conformance checks. The separately downloaded Qwen3.8-Flash-Next GSQ-RCO Q2_0 model has now been converted, exhaustively verified, and run through all 48 layers on WebGPU, threaded WebAssembly and hybrid execution. All three generated ` Paris.` from `The capital of France is`, with matching routed experts and less than 0.000008 maximum difference between final logits. The v0.2 GPU path preserved the streaming baseline logits exactly on the recorded real-model prompt. The current CUDA-generated RoPE tables introduce less than 0.000008 maximum final-logit difference from that baseline; selected experts still match on this test. Native Strata 0.1.34 chooses the same final token but has different logits and some different routed experts; native numerical parity has not been achieved. See [the current CUDA performance update](reports/cuda-performance.md), [the v0.2 execution report](reports/execution-port.md) and the [original model validation](reports/model-validation.md).

## Run

```powershell
git clone https://github.com/SamG-Coder/StrataCuda.git
cd StrataCuda
npm ci
npm run build
npm start
```

Open **http://127.0.0.1:8094**. `START.bat` starts an already built copy. The server binds to localhost and supplies COOP/COEP headers for shared-memory WASM. WebGPU also needs localhost or HTTPS.

The prebuilt Windows-friendly archive is available from [GitHub Releases](https://github.com/SamG-Coder/StrataCuda/releases); local packaging writes `dist/StrataCuda-WebCuda.zip`. Extract it and run `START.bat` with Node installed; its generated WGSL/WASM and fixture pack are included, so Emscripten is only needed to rebuild. `npm run package` regenerates the ZIP and `dist/SHA256SUMS.txt` after validation.

Requirements: Node.js 22+, a WebGPU-capable browser for GPU/hybrid execution, and Emscripten for building WASM. CPU-only execution does not need a GPU. The WebAssembly module uses SIMD, shared memory, a 256 MiB initial heap, a 512 MiB maximum heap, and 1â€“8 participating CPU threads. These limits come from the existing WebCuda backend.

On this machine the build automatically finds the existing SDK at `D:\WaterCuda\.native\emsdk`. On another machine activate your Emscripten SDK or set `EMXX` to `em++` / `em++.bat`. `EMSDK_PYTHON` can select its Python. `npm run build:gpu` builds WGSL without Emscripten; it does not enable the WASM/hybrid options until a WASM build exists.

The main page is a text interface. Click **Load downloaded model** to use the converted pack on this machine, or **Choose a model folder** to select the converted directory with its `tokenizer/` subfolder. Type a message and press Enter; Shift+Enter adds a newline. Responses appear as text while tokens arrive. **New conversation** clears conversation and recurrent/KV state while retaining loaded weights. Changing the model, backend, context or GPU budget releases the old session.

The sidebar selects WebGPU, threaded WASM or hybrid execution and an 8 GiB, 12 GiB or streaming GPU weight budget; the **context window** (128â€“2048 tokens); the **maximum reply** length; and optional model **instructions**. Chat mode uses the supplied Qwen text-only template with thinking disabled. **Text completion** sends raw text without a chat template or system instructions. The token counter includes chat-template overhead, conversation history and the reserved reply budget. Over-budget requests are rejected before model allocation; history is never silently truncated. Changed context/backend settings rebuild the session, and compatible conversation prefixes reuse existing state.

**Stop** cancels at a completed layer boundary. A partially updated recurrent state is discarded, then rebuilt from the visible conversation for the next request. Files stay local: folder loading uses browser File slices, while the downloaded-model button uses validated HTTP byte ranges from the localhost server. The main page needs no Python process for tokenization.

The original token-ID interface is now the separate **Test lab** at **http://127.0.0.1:8094/test.html**. It retains the fixture, logits, expert routing, file picker, reset and independent kernel checks. Its **Run decode** consumes token IDs and generates greedy IDs. Prompt-only tokens skip the vocabulary output projection; the last prompt token predicts the first generated token.

## Implemented

| Component | Portable implementation |
|---|---|
| Projections | Batched F32 and lossless packed BF16 weights; canonical S2/S4/S8 weights; packed Q2 expert blobs with FP16 scales; direct writes into strided output tiles |
| Normalization | Weighted RMSNorm and L2 normalization |
| Gated residual / HC | Per-stream normalization, low-rank down/SiLU/up mixing, stream collapse, injection and write-back |
| GDN | Causal convolution, SiLU, Q/K normalization, modulo head pairing, decay, gated delta state update, sigmoid output gate |
| QSA | Partial NeoX RoPE, grouped-query attention, causal KV append, stable softmax with scores shared across 64-component output tiles, and output gate; at most 2048 cells, where all cells fit upstream's selection budget |
| MoE | Stable all-expert softmax, top-k with lowest-ID tie-breaking, clamped renormalization, routed and shared SiLU experts, output combination |
| PLE | Upstream 64-bit wrapping n-gram hash, EOS context cutoff, local table row reads, key/query gate, broadcast, normalized dilated convolution and history update at layer 1 |
| Sessions | Greedy decode, reset, checkpoint/restore, exclusive state operations, hybrid-lane draining on failure, shape validation before allocation |
| Hybrid execution | Persistent GPU dense weights and complete expert blobs; CPU workers serve cold experts from bounded RAM storage; frequency-based GPU admission; evicted packed experts return to the RAM tier |
| Prompt processing | Layer-major chunks of up to 16 tokens by default, batched projections and routing, grouping by distinct expert, causal GDN/PLE updates and causal attention |
| Execution | Batched WebGPU submissions and WASM worker commands; bounded scratch-buffer and binding reuse; independent GPU, host-RAM and WASM heap budgets |
| Files | Bounded random reads, canonical pack import, F16 scale expansion, Q2_0 expert gate/up layout, IQ4_NL PLE rows, GGUF v3 header inspection |
| Text interface | Browser byte-level BPE, UTF-8 streaming, Qwen text-only chat formatting, optional instructions, context/reply limits, prefix reuse and cancellation between layers |

The default 8 GiB GPU weight budget reserves **4,339,198,220 bytes (4.04 GiB)** for the dense tensors used by this model and uses the remainder for complete expert blobs. Dense tensors become resident on first use and are protected from expert eviction. BF16 source weights regain their original compact storage without changing their values; one packed Q2 allocation serves each expert's gate, up and down projections. A separate bounded **1 GiB host-RAM tier** holds CPU/cold experts, and each WASM tensor cache remains **64 MiB** inside its 512 MiB maximum heap. A bounded dense-file window reads at most 128 MiB of one layer at a time in host memory. Scratch-buffer pooling adds at most 128 MiB on GPU. State, staging and browser overhead are separate from the weight budget.

Choose **Streaming** on a GPU with insufficient free memory. That path retains a small tile cache; the normal resident path does not repeatedly evict dense weights. The browser does not expose total free VRAM, so the budget is explicit. Model files remain random-access sources, and individual tiles remain bounded. This implements a portable tiered-memory policy; CUDA mapped pinned memory and native CPU vector kernels are not available through these browser backends.

Old cache entries are freed before uploading replacements, and partial uploads are freed after allocation failure. A tile larger than the cache budget remains temporary and is released after dispatch. State/output allocation avoids redundant zero-filled host arrays. See `reports/performance.md` for measured attention improvements and the limits of those measurements.

## Model loading

The original download is in `models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0/`: two GGUF shards totaling 66,423,878,624 bytes (66.42 GB / 61.86 GiB). Both SHA-256 hashes match the published LFS hashes at Strata's pinned revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` of [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0). Download metadata, the model card and parsed headers remain beside those originals.

The **ready-to-load pack** is in `models/Qwen3.8-Flash-Next-WebCuda-Q2_0/`. Its three canonical `.bin` files total 40,293,137,920 bytes. All 4,947,698,560 dense values and all 24,576 expert layouts passed verification against the original GGUF. `VERIFIED.json` and `SHA256SUMS.txt` record the checks and hashes. The 28.80 GB PLE shard is hard-linked into that directory on this machine, so it shares the original data without another copy. Model directories are ignored by Git and excluded from the application ZIP.

The main page's **Load downloaded model** button opens this pack directly. **Choose a model folder** accepts the entire converted directory, including `tokenizer/`. In the separate Test lab's **Open local pack files**, select these five files from the converted pack directory together:

- `manifest.json`
- `dense.bin`
- `embd.bin`
- `experts.bin`
- `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`

For the recorded raw completion, use **Text completion** on the main page with `The capital of France is` and a two-token reply limit. In the Test lab, the equivalent input IDs are `760, 6511, 314, 9338, 369`; output IDs `11751, 13` decode to ` Paris.`. The original streaming implementation took tens of seconds per model token. Resident weights and batched prompts substantially reduce that cost; current measurements, cold-load versus warm reuse, and limitations are in [the CUDA performance report](reports/cuda-performance.md).

The command-line runner accepts text and automates the same real browser file picker and CUDA inference path:

```powershell
npm run test:model -- --modes=webgpu --steps=2 "--prompt=The capital of France is"
# Run and compare all three backends, using four CPU threads:
npm run test:model -- --modes=webgpu,wasm,hybrid
npm run compare:model
```

Python runs the unchanged upstream tokenizer only; all inference arithmetic still runs in WebCuda. The local Python environment is already installed at `.local/pack-env`. On a fresh copy, create it with `python -m venv .local/pack-env`, then run `.local/pack-env/Scripts/python.exe -m pip install -r requirements-model.txt`. Install the development runner with `npm ci`. The CLI uses installed Edge by default; set `STRATA_BROWSER=chrome` for Chrome. Use `--pack=PATH` for a different converted pack location and `--python=PATH` for another Python executable.

To reproduce conversion into a **new or empty** directory:

```powershell
.local/pack-env/Scripts/python.exe -B scripts/convert-model.py --gguf models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --out models/Qwen3.8-Flash-Next-WebCuda-Q2_0
```

The wrapper uses the pinned, unchanged upstream pack builder, checks every dense value and every expert byte with bounded memory, hashes originals and outputs, extracts the tokenizer, and links the PLE shard beside the pack. If source and output are on volumes that cannot share a hard link, it copies the PLE shard instead, requiring another 28.80 GB. To recheck an existing pack, use `scripts/verify-model.py --gguf FIRST_SHARD --pack PACK_DIRECTORY` with the same Python executable. The source checkout is not required; the necessary upstream tools are vendored unchanged.

The file picker accepts two formats:

1. **`strata-webcuda-v1`**: a manifest with geometry, PLE hash constants and named tensors. `npm run build` writes a sample under `generated/fixture/`. Select its `manifest.json` and `weights.bin` together to test the real file-loading path.
2. **Upstream canonical `strata-pack` with Q2_0 experts**: select `manifest.json`, its `.bin` files, and the original PLE GGUF shard. Tensor names and shapes follow upstream. The PLE shard must contain `per_layer_token_embd.weight` in IQ4_NL. F32/P16 value planes and S2/S4/S8 canonical planes are supported. The default production geometry follows upstream `ModelGeometry`; the embedding supplies the vocabulary size. The portable session defaults to 256 context cells.

The loader rejects other expert formats, unsupported/missing planes, inconsistent shapes, truncated files and non-finite weights. A raw Qwen GGUF alone is **not** an executable model for this port. The GGUF reader inspects headers and locates PLE rows; it is not a generic GGUF inference backend. The current upstream installer can produce native IQ layouts that this loader does not accept.

## Numerical and compatibility boundaries

The portable baseline uses **F32 activations, F32 reductions and F32 KV storage**. Native Strata has BF16/F16 activation rounding, quantized activation products, native reduction trees and optional quantized KV formats. The equations and layout semantics are preserved within the implemented baseline, but this is not a claim of native bit-exact logits or equivalent model quality. Long-context YaRN/MRoPE policies and sparse indexer selection are not implemented; context beyond 2048 is refused.

The following upstream features remain outside this port: native IQ2/IQ3 expert packs and direct general GGUF model loading; MTP and prompt-lookup speculative decoding; multi-GPU execution; vision and tool calling; arbitrary chat-template interpretation; OpenAI/Anthropic API serving; quantized KV streaming; native pinned-memory/AVX/BLAS optimizations. Browser Chat supports the exact supplied Qwen template for text-only, non-thinking conversations; an unfamiliar template requires Text completion mode. The command-line model runner retains raw text completion through the unchanged upstream Python tokenizer.

## Verification

```powershell
npm test                 # file formats, corrupt-file negatives, hash and layout contracts
npm run build            # all 49 entries to WGSL and WASM
npm run test:wasm        # threaded module in Node against independent scalar equations
npm run test:browser     # real Edge: GPU, WASM, hybrid, file picker, UI and responsive layout
npm run test:upstream    # native C++ oracle from unchanged upstream source (needs clang++)
npm run test:model      # full downloaded model, real browser, raw text completion
npm run test:tokenizer  # browser tokenizer vs unchanged Python; chat format vs original Jinja
npm run test:chat       # main page, context limits, Stop and real-model text chat
npm run test:ui         # desktop/mobile main page without needing model files
npm run benchmark:model # real-model cold load, warm repeat, I/O and queue counters
```

Set `STRATA_BROWSER=chrome` to use installed Chrome instead of Edge. `STRATA_THREADS` sets the Node WASM test thread count. `CXX` selects the compiler for the native source oracle. `npm run check` runs unit tests, the build, Node WASM, the browser kernel suite and main-page layout checks. The native oracle and model-dependent checks are separate, so ordinary browser development requires neither a C++ toolchain nor the downloaded weights.

The conformance suite checks full 512-expert routing, 2560-wide projections, S2/S4/S8 scale/offset handling, three updates of the real 128 Ã— 48 Ã— 128 GDN state shape, PLE history, attention, RoPE, five complete fixture tokens, reset and checkpoint replay. GPU, WASM and hybrid outputs are compared against a separate scalar/F64 implementation. The file picker test loads the emitted pack and performs autoregressive decode through the UI. Served WGSL hashes are checked against the build manifest.

Fixture reports are written to `reports/wasm.json`, `reports/browser.json` and `reports/upstream.json`; screenshots go to `reports/browser.png` and `reports/mobile.png`. Those measurements describe fixture and kernel tests. The separate trained-model runs write `reports/model-<backend>.json` and screenshots; full final logits are saved under `.local/model-validation/`. See `reports/validation.md` and `reports/model-validation.md` for measured scope and limitations.

## Source map

- `kernels/strata.cu`: all 49 shared compute entries.
- `src/residency.js`: protected dense tensors and whole-expert GPU/RAM placement.
- `src/prefill.js`: layer-major prompt chunks and grouped expert projections.
- `scripts/benchmark-model.mjs`: real-model timings, I/O counters and warm-repeat checks.
- `scripts/compare-native.py`: pinned upstream executable comparison, including routing differences.
- `src/backend.js`, `src/wasm-worker.js`: both WebCuda runtime adapters and ordered CPU worker RPC.
- `src/engine.js`: GDN/QSA/PLE/HC/MoE token dispatch, adaptive expert placement and session state.
- `src/model.js`, `src/gguf.js`: validated random-access model files and upstream pack layouts.
- `src/tokenizer.js`, `src/text-session.js`, `src/model-files.js`: browser text I/O, conversation state and folder/HTTP model loading.
- `web/chat.js`, `web/chat.css`: main text interface; `test.html` retains the diagnostic interface.
- `src/ops.js`: bounded weight tiles and LRU device residency.
- `tests/reference.mjs`: independent scalar mathematical oracle.
- `vendor/webcuda/`: the user's compiler/runtime, with four bit-cast intrinsics added to its WASM shim; see `THIRD_PARTY_NOTICES.md`.
- `scripts/convert-model.py`, `scripts/verify-model.py`: bounded conversion workflow and exhaustive pack verification.
- `scripts/test-model.mjs`, `scripts/compare-model.mjs`: real-model browser runs and backend comparisons.
- `vendor/strata/`: unchanged upstream native oracle, pack conversion and tokenizer sources.

`PROVENANCE.json` records pinned revisions and source hashes. Upstream licenses and attribution are retained in `LICENSE`, `THIRD_PARTY_NOTICES.md` and `vendor/`.
