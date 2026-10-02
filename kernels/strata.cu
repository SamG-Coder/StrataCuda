// Strata portable compute, adapted from Niko1221/Strata at 1678de333d0e0711bc414ad992b640e1a37dd814.
// Upstream MIT license: vendor/STRATA-LICENSE. See PROVENANCE.json for source mapping.
// Every entry has 64 lanes. Only explicit buffers, scalar parameters and fixed
// shared arrays are used, so this source runs on BOTH WebCuda backends.
// Accumulation is F32: this is a portable numerical contract, not native CUDA bit parity.

__device__ float strata_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ float strata_silu(float x) { return x * strata_sigmoid(x); }
__device__ float strata_half_bits(unsigned int h) {
    unsigned int sign = (h & 32768u) << 16; unsigned int exponent = (h >> 10) & 31u; unsigned int fraction = h & 1023u;
    if (exponent == 0u) { if (fraction == 0u) return __uint_as_float(sign); float v = (float)fraction * 0.000000059604644775390625f; return sign != 0u ? -v : v; }
    if (exponent == 31u) return __uint_as_float(sign | 2139095040u | (fraction << 13));
    return __uint_as_float(sign | ((exponent + 112u) << 23) | (fraction << 13));
}
__global__ void strata_unpack_half(const unsigned int* Packed, float* Values, int count) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < count) Values[i] = strata_half_bits((Packed[i / 2] >> ((i % 2) * 16)) & 65535u);
}
__device__ float strata_iq4(int c) {
    if (c == 0) return -127.0f; if (c == 1) return -104.0f;
    if (c == 2) return -83.0f; if (c == 3) return -65.0f;
    if (c == 4) return -49.0f; if (c == 5) return -35.0f;
    if (c == 6) return -22.0f; if (c == 7) return -10.0f;
    if (c == 8) return 1.0f; if (c == 9) return 13.0f;
    if (c == 10) return 25.0f; if (c == 11) return 38.0f;
    if (c == 12) return 53.0f; if (c == 13) return 69.0f;
    if (c == 14) return 89.0f; return 113.0f;
}

// Row-major weights, batch-major activations. One workgroup per output row/token.
__global__ void strata_gemv(const float* X, const float* W, float* Y,
                          int cols, int rows, int tokens) {
    __shared__ float partial[64];
    int lane = (int)threadIdx.x; int row = (int)blockIdx.x; int token = (int)blockIdx.y;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 64) sum += W[row * cols + i] * X[token * cols + i];
    partial[lane] = sum; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] += partial[lane + s]; __syncthreads(); }
    if (lane == 0 && row < rows && token < tokens) Y[token * rows + row] = partial[0];
}

// Strata's canonical S2/S4/S8 form: packed codes, explicit scale AND offset planes.
// Codes are consecutive little-endian bits; their order is already canonicalized.
__global__ void strata_quant_gemv(const float* X, const unsigned int* Codes,
    const float* Scales, const float* Offsets, float* Y,
    int cols, int rows, int bits, int groupSize, int bias, int codebook, int hasOffset) {
    __shared__ float partial[64];
    int lane = (int)threadIdx.x; int row = (int)blockIdx.x;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 64) {
        int index = row * cols + i; int bit = index * bits;
        int code = (int)((Codes[bit / 32] >> (bit % 32)) & ((1u << bits) - 1u));
        float value = codebook == 1 ? strata_iq4(code) : (float)(code + bias);
        int group = index / groupSize;
        float weight = value * Scales[group]; if (hasOffset != 0) weight += Offsets[group];
        sum += weight * X[i];
    }
    partial[lane] = sum; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] += partial[lane + s]; __syncthreads(); }
    if (lane == 0 && row < rows) Y[row] = partial[0];
}

// mode 0: weighted RMSNorm; mode 1: L2 norm. weightRows=1 shares gamma across heads.
// A projection tile writes directly to its final strided batch output. The same
// weights serve every prompt token in this dispatch; no output-tile copies.
__global__ void strata_project(const float* X, const float* W, float* Y,
    int cols, int rows, int tokens, int outputStride, int outputOffset) {
    __shared__ float partial[64];
    int lane = (int)threadIdx.x; int row = (int)blockIdx.x; int token = (int)blockIdx.y;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 64) sum += W[row * cols + i] * X[token * cols + i];
    partial[lane] = sum; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] += partial[lane + s]; __syncthreads(); }
    if (lane == 0 && row < rows && token < tokens) Y[token * outputStride + outputOffset + row] = partial[0];
}
// BF16 values were promoted to F32 by the canonical pack. Restoring their original
// 16-bit storage is lossless; activations and accumulation still use F32.
__global__ void strata_bf16_project(const float* X, const unsigned int* Codes, float* Y,
    int cols, int rows, int tokens, int outputStride, int outputOffset) {
    __shared__ float partial[64];
    int lane = (int)threadIdx.x; int row = (int)blockIdx.x; int token = (int)blockIdx.y; float sum = 0.0f;
    for (int i = lane; i < cols; i += 64) { int ix = row * cols + i; unsigned int bits = (Codes[ix / 2] >> ((ix % 2) * 16)) & 65535u; sum += __uint_as_float(bits << 16) * X[token * cols + i]; }
    partial[lane] = sum; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] += partial[lane + s]; __syncthreads(); }
    if (lane == 0 && row < rows && token < tokens) Y[token * outputStride + outputOffset + row] = partial[0];
}
// One complete canonical Q2_0 expert allocation serves gate/up/down directly,
// including interleaved gate/up rows and packed FP16 scales.
__global__ void strata_q2_project(const float* X, const unsigned int* Blob, float* Y,
    int cols, int rows, int tokens, int outputStride, int outputOffset,
    int codeOffset, int scaleOffset, int weightRow, int rowStride) {
    __shared__ float partial[64];
    int lane = (int)threadIdx.x; int row = (int)blockIdx.x; int token = (int)blockIdx.y; float sum = 0.0f;
    for (int i = lane; i < cols; i += 64) {
        int ix = (weightRow + row * rowStride) * cols + i; int code = (int)((Blob[codeOffset + ix / 16] >> ((ix % 16) * 2)) & 3u) - 1;
        int si = scaleOffset + ix / 64; unsigned int half = (Blob[si / 2] >> ((si % 2) * 16)) & 65535u;
        float weight = (float)code * strata_half_bits(half); sum += weight * X[token * cols + i];
    }
    partial[lane] = sum; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] += partial[lane + s]; __syncthreads(); }
    if (lane == 0 && row < rows && token < tokens) Y[token * outputStride + outputOffset + row] = partial[0];
}
__global__ void strata_quant_project(const float* X, const unsigned int* Codes,
    const float* Scales, const float* Offsets, float* Y,
    int cols, int rows, int tokens, int outputStride, int outputOffset,
    int bits, int groupSize, int bias, int codebook, int hasOffset) {
    __shared__ float partial[64];
    int lane = (int)threadIdx.x; int row = (int)blockIdx.x; int token = (int)blockIdx.y;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 64) {
        int index = row * cols + i; int bit = index * bits;
        int code = (int)((Codes[bit / 32] >> (bit % 32)) & ((1u << bits) - 1u));
        float value = codebook == 1 ? strata_iq4(code) : (float)(code + bias);
        int group = index / groupSize;
        float weight = value * Scales[group]; if (hasOffset != 0) weight += Offsets[group];
        sum += weight * X[token * cols + i];
    }
    partial[lane] = sum; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] += partial[lane + s]; __syncthreads(); }
    if (lane == 0 && row < rows && token < tokens) Y[token * outputStride + outputOffset + row] = partial[0];
}

// mode 0: weighted RMSNorm; mode 1: L2 norm. weightRows=1 shares gamma across heads.
__global__ void strata_norm(const float* X, const float* W, float* Y,
    int cols, int rows, int weightRows, int mode, float epsilon, float scale) {
    __shared__ float partial[64];
    int lane = (int)threadIdx.x; int row = (int)blockIdx.x;
    float ss = 0.0f;
    for (int i = lane; i < cols; i += 64) { float v = X[row * cols + i]; ss += v * v; }
    partial[lane] = ss; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] += partial[lane + s]; __syncthreads(); }
    float inv = scale / sqrtf((mode == 0 ? partial[0] / (float)cols : partial[0]) + epsilon);
    for (int i = lane; i < cols; i += 64) {
        float w = mode == 0 ? W[(row % weightRows) * cols + i] : 1.0f;
        if (row < rows) Y[row * cols + i] = X[row * cols + i] * inv * w;
    }
}

// 0 copy/scale; 1 SiLU; 2 sigmoid; 3 SiLU(A)*B; 4 A*sigmoid(B); 5 add.
__global__ void strata_elementwise(const float* A, const float* B, float* Y, int n, int mode, float scale) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i >= n) return;
    float a = A[i] * scale; float v = a;
    if (mode == 1) v = strata_silu(a); if (mode == 2) v = strata_sigmoid(a);
    if (mode == 3) v = strata_silu(a) * B[i]; if (mode == 4) v = a * strata_sigmoid(B[i]);
    if (mode == 5) v = a + B[i]; Y[i] = v;
}

// Stable softmax over ALL experts, stable top-k, then clamped renormalization.
// At most 512 experts and 32 selected. Ties prefer the lower expert ID.
__global__ void strata_router(const float* Logits, int* Ids, float* Weights, int experts, int topK) {
    __shared__ float prob[512]; __shared__ float reduce[64];
    __shared__ int winner[64]; __shared__ float selected[32];
    int lane = (int)threadIdx.x;
    float mx = -3.402823e38f;
    for (int e = lane; e < experts; e += 64) mx = fmaxf(mx, Logits[e]);
    reduce[lane] = mx; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) reduce[lane] = fmaxf(reduce[lane], reduce[lane + s]); __syncthreads(); }
    mx = reduce[0]; __syncthreads(); float sum = 0.0f;
    for (int e = lane; e < experts; e += 64) { prob[e] = expf(Logits[e] - mx); sum += prob[e]; }
    reduce[lane] = sum; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) reduce[lane] += reduce[lane + s]; __syncthreads(); }
    float inv = 1.0f / reduce[0]; __syncthreads();
    for (int e = lane; e < experts; e += 64) prob[e] *= inv;
    __syncthreads();
    for (int k = 0; k < topK; ++k) {
        float best = -1.0f; int id = experts;
        for (int e = lane; e < experts; e += 64) if (prob[e] > best || (prob[e] == best && e < id)) { best = prob[e]; id = e; }
        reduce[lane] = best; winner[lane] = id; __syncthreads();
        for (int s = 32; s > 0; s >>= 1) {
            if (lane < s) { float v = reduce[lane + s]; int j = winner[lane + s];
                if (v > reduce[lane] || (v == reduce[lane] && j < winner[lane])) { reduce[lane] = v; winner[lane] = j; }
            } __syncthreads();
        }
        if (lane == 0) { Ids[k] = winner[0]; selected[k] = reduce[0]; prob[winner[0]] = -1.0f; }
        __syncthreads();
    }
    if (lane == 0) { float denom = 0.0f; for (int k = 0; k < topK; ++k) denom += selected[k];
        denom = fmaxf(denom, 0.00006103515625f); for (int k = 0; k < topK; ++k) Weights[k] = selected[k] / denom;
    }
}

// HC collapse and write-back are separate passes; no read/write race across streams.
__global__ void strata_hc_mix(const float* Norm, const float* Gate, float* Mixed, int width, int streams) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i >= width) return;
    float v = 0.0f; for (int c = 0; c < streams; ++c) v += Norm[c * width + i] * strata_sigmoid(Gate[c * width + i]);
    Mixed[i] = v / (float)streams;
}
__global__ void strata_hc_write(float* Residual, const float* Output, const float* Inject, int width, int streams) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i >= width * streams) return;
    Residual[i] += Output[i % width] * (2.0f * strata_sigmoid(Inject[i / width] / (float)streams));
}
__global__ void strata_embed(const float* Row, float* Residual, int width, int streams) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < width * streams) Residual[i] = Row[i % width];
}

// Causal convolution with channel-major history and kernel[0] = oldest tap.
__global__ void strata_conv(float* History, const float* X, const float* W, float* Y,
    int channels, int taps, int dilation, int activation) {
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (c >= channels) return;
    int history = (taps - 1) * dilation; float sum = 0.0f;
    for (int t = 0; t < taps - 1; ++t) sum += History[c * history + t * dilation] * W[c * taps + t];
    sum += X[c] * W[c * taps + taps - 1]; Y[c] = activation == 1 ? strata_silu(sum) : sum;
    for (int t = 0; t < history - 1; ++t) History[c * history + t] = History[c * history + t + 1];
    if (history > 0) History[c * history + history - 1] = X[c];
}
__global__ void strata_gdn_gates(const float* Alpha, const float* Beta, const float* A, const float* Dt,
    float* Decay, float* Strength, int heads) {
    int h = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (h >= heads) return;
    float a = Alpha[h] + Dt[h]; float sp = a > 20.0f ? a : logf(1.0f + expf(a));
    Decay[h] = expf(A[h] * sp); Strength[h] = strata_sigmoid(Beta[h]);
}
// State shape [keyDimension, valueHeads, valueDimension], exactly Strata's layout.
// Value head h pairs with key head h % keyHeads (NOT h / replication).
__global__ void strata_gdn_step(float* State, const float* Q, const float* K, const float* V,
    const float* Decay, const float* Beta, float* Y, int dim, int keyHeads, int valueHeads) {
    int jh = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (jh >= dim * valueHeads) return;
    int h = jh / dim; int kh = h % keyHeads; int stride = valueHeads * dim;
    float prediction = 0.0f;
    for (int i = 0; i < dim; ++i) { int ix = i * stride + jh; State[ix] *= Decay[h]; prediction += State[ix] * K[kh * dim + i]; }
    float delta = (V[jh] - prediction) * Beta[h]; float result = 0.0f;
    for (int i = 0; i < dim; ++i) { int ix = i * stride + jh; State[ix] += K[kh * dim + i] * delta; result += State[ix] * Q[kh * dim + i]; }
    Y[jh] = result;
}
__global__ void strata_slice(const float* X, float* Y, int width, int rows, int stride, int offset) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < width * rows) Y[i] = X[(i / width) * stride + offset + i % width];
}

// NeoX half-split rotary embedding; cos/sin supplied by model's position policy.
__global__ void strata_rope(const float* X, const float* Cos, const float* Sin, float* Y,
    int dim, int heads, int rotary) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i >= dim * heads) return;
    int d = i % dim; int base = i - d;
    if (d >= rotary) { Y[i] = X[i]; return; }
    int half = rotary / 2; int pair = d % half;
    float a = X[base + pair]; float b = X[base + pair + half];
    Y[i] = d < half ? a * Cos[pair] - b * Sin[pair] : b * Cos[pair] + a * Sin[pair];
}
__global__ void strata_kv_append(const float* K, const float* V, float* Keys, float* Values, int width, int position) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < width) { Keys[position * width + i] = K[i]; Values[position * width + i] = V[i]; }
}
// One workgroup per head and 64-component output tile. Scores are shared across
// the tile, avoiding a repeated query/key dot product for each output component.
// The host caps the selected cell count at 2048 (8 KiB of shared score storage).
__global__ void strata_attention(const float* Q, const float* Keys, const float* Values,
    const int* Cells, float* Y, int dim, int heads, int kvHeads, int count, float scale) {
    __shared__ float scores[2048]; __shared__ float partial[64];
    int lane = (int)threadIdx.x; int h = (int)blockIdx.x;
    int d = (int)blockIdx.y * 64 + lane; int kh = h / (heads / kvHeads);
    float mx = -3.402823e38f;
    for (int t = lane; t < count; t += 64) {
        int base = (Cells[t] * kvHeads + kh) * dim; float score = 0.0f;
        for (int j = 0; j < dim; ++j) score += Q[h * dim + j] * Keys[base + j];
        score *= scale; scores[t] = score; mx = fmaxf(mx, score);
    }
    partial[lane] = mx; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] = fmaxf(partial[lane], partial[lane + s]); __syncthreads(); }
    mx = partial[0]; __syncthreads(); float sum = 0.0f;
    for (int t = lane; t < count; t += 64) { float p = expf(scores[t] - mx); scores[t] = p; sum += p; }
    partial[lane] = sum; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] += partial[lane + s]; __syncthreads(); }
    if (h < heads && d < dim) {
        float value = 0.0f;
        for (int t = 0; t < count; ++t) value += scores[t] * Values[(Cells[t] * kvHeads + kh) * dim + d];
        Y[h * dim + d] = value / partial[0];
    }
}
__global__ void strata_attention_gate(const float* Attention, const float* FullQ, float* Y, int dim, int heads) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < dim * heads) Y[i] = Attention[i] * strata_sigmoid(FullQ[(i / dim) * dim * 2 + dim + i % dim]);
}
__global__ void strata_moe_combine(const float* Parts, const float* Weights, const float* Shared,
    const float* SharedGate, float* Y, int width, int count) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i >= width) return;
    float value = Shared[i] * strata_sigmoid(SharedGate[0]);
    for (int k = 0; k < count; ++k) value += Parts[k * width + i] * Weights[k]; Y[i] = value;
}

__global__ void strata_ple_gate(const float* Key, const float* Query, float* Gate, int width, int streams) {
    __shared__ float partial[64];
    int lane = (int)threadIdx.x; int c = (int)blockIdx.x; float dot = 0.0f;
    for (int d = lane; d < width; d += 64) dot += Key[c * width + d] * Query[c * width + d];
    partial[lane] = dot; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] += partial[lane + s]; __syncthreads(); }
    if (lane == 0 && c < streams) {
        float v = partial[0] / sqrtf((float)width); float sign = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
        Gate[c] = strata_sigmoid(sign * sqrtf(fmaxf(fabsf(v), 0.000001f)));
    }
}
__global__ void strata_ple_broadcast(const float* Value, const float* Gate, float* Y, int width, int streams) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < width * streams) Y[i] = Value[i % width] * Gate[i / width];
}
__global__ void strata_ple_add(float* Residual, const float* Gated, const float* Conv, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < n) Residual[i] += Gated[i] + Conv[i];
}

// Deterministic greedy decoding. Lowest token index wins a tie.
__global__ void strata_argmax(const float* Logits, int* Token, int count) {
    __shared__ float values[64]; __shared__ int ids[64];
    int lane = (int)threadIdx.x; float best = -3.402823e38f; int id = count;
    for (int i = lane; i < count; i += 64) if (Logits[i] > best || (Logits[i] == best && i < id)) { best = Logits[i]; id = i; }
    values[lane] = best; ids[lane] = id; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) {
        if (values[lane + s] > values[lane] || (values[lane + s] == values[lane] && ids[lane + s] < ids[lane])) { values[lane] = values[lane + s]; ids[lane] = ids[lane + s]; }
    } __syncthreads(); }
    if (lane == 0) Token[0] = ids[0];
}

// Prompt kernels keep the token dimension explicit. Recurrent state is updated
// in chronological order by the same owner thread, including across chunks.
__global__ void strata_embed_batch(const float* Rows, float* Residual, int width, int streams, int tokens) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); int hc = width * streams;
    if (i < hc * tokens) Residual[i] = Rows[(i / hc) * width + i % width];
}
__global__ void strata_hc_mix_batch(const float* Norm, const float* Gate, float* Mixed, int width, int streams, int tokens) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i >= width * tokens) return;
    int token = i / width; int d = i % width; float v = 0.0f;
    for (int c = 0; c < streams; ++c) { int p = (token * streams + c) * width + d; v += Norm[p] * strata_sigmoid(Gate[p]); }
    Mixed[i] = v / (float)streams;
}
__global__ void strata_hc_write_batch(float* Residual, const float* Output, const float* Inject, int width, int streams, int tokens) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); int hc = width * streams; if (i >= hc * tokens) return;
    Residual[i] += Output[(i / hc) * width + i % width] * (2.0f * strata_sigmoid(Inject[i / width] / (float)streams));
}
__global__ void strata_conv_batch(float* History, const float* X, const float* W, float* Y,
    int channels, int taps, int dilation, int activation, int tokens) {
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (c >= channels) return;
    int history = (taps - 1) * dilation;
    for (int token = 0; token < tokens; ++token) {
        float sum = 0.0f;
        for (int t = 0; t < taps - 1; ++t) sum += History[c * history + t * dilation] * W[c * taps + t];
        sum += X[token * channels + c] * W[c * taps + taps - 1]; Y[token * channels + c] = activation == 1 ? strata_silu(sum) : sum;
        for (int t = 0; t < history - 1; ++t) History[c * history + t] = History[c * history + t + 1];
        History[c * history + history - 1] = X[token * channels + c];
    }
}
__global__ void strata_gdn_gates_batch(const float* Alpha, const float* Beta, const float* A, const float* Dt,
    float* Decay, float* Strength, int heads, int tokens) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i >= heads * tokens) return;
    float x = Alpha[i] + Dt[i % heads];
    float sp = x > 20.0f ? x : logf(1.0f + expf(x));
    Decay[i] = expf(A[i % heads] * sp); Strength[i] = strata_sigmoid(Beta[i]);
}
__global__ void strata_gdn_batch(float* State, const float* Q, const float* K, const float* V,
    const float* Decay, const float* Beta, float* Y, int dim, int keyHeads, int valueHeads, int tokens) {
    int jh = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (jh >= dim * valueHeads) return;
    int h = jh / dim; int kh = h % keyHeads; int stride = valueHeads * dim;
    for (int token = 0; token < tokens; ++token) {
        int qbase = (token * keyHeads + kh) * dim; int gh = token * valueHeads + h;
        float prediction = 0.0f;
        for (int i = 0; i < dim; ++i) { int ix = i * stride + jh; State[ix] *= Decay[gh]; prediction += State[ix] * K[qbase + i]; }
        float delta = (V[token * stride + jh] - prediction) * Beta[gh]; float result = 0.0f;
        for (int i = 0; i < dim; ++i) { int ix = i * stride + jh; State[ix] += K[qbase + i] * delta; result += State[ix] * Q[qbase + i]; }
        Y[token * stride + jh] = result;
    }
}
__global__ void strata_rope_batch(const float* X, const float* Cos, const float* Sin, float* Y,
    int dim, int heads, int rotary, int tokens) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i >= dim * heads * tokens) return;
    int d = i % dim; if (d >= rotary) { Y[i] = X[i]; return; }
    int half = rotary / 2; int pair = d % half; int base = i - d; int angle = (i / (dim * heads)) * half + pair;
    float a = X[base + pair]; float b = X[base + pair + half];
    Y[i] = d < half ? a * Cos[angle] - b * Sin[angle] : b * Cos[angle] + a * Sin[angle];
}
__global__ void strata_kv_append_batch(const float* K, const float* V, float* Keys, float* Values,
    int width, int position, int tokens) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < width * tokens) { Keys[position * width + i] = K[i]; Values[position * width + i] = V[i]; }
}
__global__ void strata_attention_gate_batch(const float* Attention, const float* FullQ, float* Y, int dim, int heads, int tokens) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < dim * heads * tokens) Y[i] = Attention[i] * strata_sigmoid(FullQ[(i / dim) * dim * 2 + dim + i % dim]);
}
__global__ void strata_ple_broadcast_batch(const float* Value, const float* Gate, float* Y, int width, int streams, int tokens) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); int hc = width * streams;
    if (i < hc * tokens) Y[i] = Value[(i / hc) * width + i % width] * Gate[i / width];
}
__global__ void strata_gather_rows(const float* X, const int* Indices, float* Y, int width, int rows) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < width * rows) Y[i] = X[Indices[i / width] * width + i % width];
}
__global__ void strata_scatter_rows(const float* X, const int* Indices, float* Y, int width, int rows) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i < width * rows) Y[Indices[i / width] * width + i % width] = X[i];
}
__global__ void strata_moe_combine_batch(const float* Parts, const float* Weights, const float* Shared,
    const float* SharedGate, float* Y, int width, int count, int tokens) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i >= width * tokens) return;
    int token = i / width; int d = i % width;
    float value = Shared[i] * strata_sigmoid(SharedGate[token]);
    for (int k = 0; k < count; ++k) value += Parts[(token * count + k) * width + d] * Weights[token * count + k]; Y[i] = value;
}

// Each prompt row routes independently, using the decode reduction order.
__global__ void strata_router_batch(const float* Logits, int* Ids, float* Weights, int experts, int topK) {
    __shared__ float prob[512]; __shared__ float reduce[64];
    __shared__ int winner[64]; __shared__ float selected[32];
    int token = (int)blockIdx.x; int lane = (int)threadIdx.x;
    float mx = -3.402823e38f;
    for (int e = lane; e < experts; e += 64) mx = fmaxf(mx, Logits[token * experts + e]);
    reduce[lane] = mx; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) reduce[lane] = fmaxf(reduce[lane], reduce[lane + s]); __syncthreads(); }
    mx = reduce[0]; __syncthreads(); float sum = 0.0f;
    for (int e = lane; e < experts; e += 64) { prob[e] = expf(Logits[token * experts + e] - mx); sum += prob[e]; }
    reduce[lane] = sum; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) reduce[lane] += reduce[lane + s]; __syncthreads(); }
    float inv = 1.0f / reduce[0]; __syncthreads();
    for (int e = lane; e < experts; e += 64) prob[e] *= inv;
    __syncthreads();
    for (int k = 0; k < topK; ++k) {
        float best = -1.0f; int id = experts;
        for (int e = lane; e < experts; e += 64) if (prob[e] > best || (prob[e] == best && e < id)) { best = prob[e]; id = e; }
        reduce[lane] = best; winner[lane] = id; __syncthreads();
        for (int s = 32; s > 0; s >>= 1) {
            if (lane < s) { float v = reduce[lane + s]; int j = winner[lane + s];
                if (v > reduce[lane] || (v == reduce[lane] && j < winner[lane])) { reduce[lane] = v; winner[lane] = j; }
            } __syncthreads();
        }
        if (lane == 0) { Ids[token * topK + k] = winner[0]; selected[k] = reduce[0]; prob[winner[0]] = -1.0f; }
        __syncthreads();
    }
    if (lane == 0) { float denom = 0.0f; for (int k = 0; k < topK; ++k) denom += selected[k];
        denom = fmaxf(denom, 0.00006103515625f); for (int k = 0; k < topK; ++k) Weights[token * topK + k] = selected[k] / denom;
    }
}

// All KV rows are present, but each prompt query sees only its causal prefix.
__global__ void strata_attention_batch(const float* Q, const float* Keys, const float* Values,
    float* Y, int dim, int heads, int kvHeads, int position, float scale) {
    __shared__ float scores[2048]; __shared__ float partial[64];
    int token = (int)blockIdx.z; int count = position + token + 1; int lane = (int)threadIdx.x; int h = (int)blockIdx.x;
    int d = (int)blockIdx.y * 64 + lane; int kh = h / (heads / kvHeads);
    float mx = -3.402823e38f;
    for (int t = lane; t < count; t += 64) {
        int base = (t * kvHeads + kh) * dim; float score = 0.0f;
        for (int j = 0; j < dim; ++j) score += Q[(token * heads + h) * dim + j] * Keys[base + j];
        score *= scale; scores[t] = score; mx = fmaxf(mx, score);
    }
    partial[lane] = mx; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] = fmaxf(partial[lane], partial[lane + s]); __syncthreads(); }
    mx = partial[0]; __syncthreads(); float sum = 0.0f;
    for (int t = lane; t < count; t += 64) { float p = expf(scores[t] - mx); scores[t] = p; sum += p; }
    partial[lane] = sum; __syncthreads();
    for (int s = 32; s > 0; s >>= 1) { if (lane < s) partial[lane] += partial[lane + s]; __syncthreads(); }
    if (h < heads && d < dim) {
        float value = 0.0f;
        for (int t = 0; t < count; ++t) value += scores[t] * Values[(t * kvHeads + kh) * dim + d];
        Y[(token * heads + h) * dim + d] = value / partial[0];
    }
}
