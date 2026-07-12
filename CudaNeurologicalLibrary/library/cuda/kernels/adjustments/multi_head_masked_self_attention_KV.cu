#include "../../adjustments.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernel — process_KV_gradients
//
// Fused computation of both gradients via two transposed matmuls that share
// the same contracted dimension (q) and the same output space (k_pos, head, d):
//
//   values_grad[k][d] = Σ_q  softmax[q][k]                  · output_grad[q][d]
//   keys_grad  [k][d] = Σ_q  score_grad[q][k] · scores_scale · queries[q][d]
//
// Splitting into two kernels would double the q-tile iterations. Keeping them
// fused halves the global reads and tile iterations with only 1 KB of shared
// memory (4 × TILE²).
//
// Grid  : (ceil(headOutputSize / TILE),  ceil(length / TILE),  headsAmount)
// Block : (TILE, TILE)   tx = k_local during weight loading / d_local during
//                             source loading and accumulation   (dual role)
//                        ty = q_local during loading / k_local during
//                             accumulation                      (dual role)
//
// Shared-memory layout
// ────────────────────
// All four arrays are indexed [ty][tx] during loading — ty=q_local, while tx
// serves as k_local for the weight arrays (shS, shSG) and d_local for the
// source arrays (shOG, shQ).  During accumulation ty becomes k_local:
//
//   shS [TILE][TILE]   shS [ty=q_local][tx=k_local] = softmax[q][k]
//   shSG[TILE][TILE]   shSG[ty=q_local][tx=k_local] = score_grad[q][k] · scale
//   shOG[TILE][TILE]   shOG[ty=q_local][tx=d_local] = output_grad[q][d]
//   shQ [TILE][TILE]   shQ [ty=q_local][tx=d_local] = queries[q][d]
//
//   accumulation:  val_grad += shS [i][ty] · shOG[i][tx]
//                 key_grad  += shSG[i][ty] · shQ [i][tx]
//
// All global loads are coalesced:
//   shS / shSG — tx indexes k (stride 1), not q (stride H·L as in old code)
//   shOG / shQ — tx indexes d (stride 1)
//
// No bank conflicts during accumulation (column access with consecutive banks)
// and no padding is needed.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void process_KV_gradients_kernel(
    const kernel_float* __restrict__ final_output_gradient,
    const kernel_float* __restrict__ queries,
    const kernel_float* __restrict__ cache_softmax,
    const kernel_float* __restrict__ cache_scores_gradient,
    kernel_float* __restrict__ keys_gradient,
    kernel_float* __restrict__ values_gradient,
    const int length,
    const int scoreStart,
    const int headsAmount,
    const int headOutputSize,
    const int sample_start,
    const kernel_float scores_scale)
{
    __shared__ kernel_float shS [TILE][TILE]; // softmax           [q_local][k_local]
    __shared__ kernel_float shSG[TILE][TILE]; // score_grad·scale  [q_local][k_local]
    __shared__ kernel_float shOG[TILE][TILE]; // output_grad       [q_local][d_local]
    __shared__ kernel_float shQ [TILE][TILE]; // queries           [q_local][d_local]

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int d       = blockIdx.x * TILE + tx;   // output d position
    const int k_pos   = blockIdx.y * TILE + ty;   // output k position
    const int k_block = blockIdx.y * TILE;
    const int head    = blockIdx.z;

    kernel_float val_grad = 0.f;
    kernel_float key_grad = 0.f;

    for (int q_tile = 0; q_tile < length; q_tile += TILE) {

        // ── Weight arrays: ty = q_local, tx = k_local (coalesced in k) ────────
        const int  q_w    = q_tile + ty;
        const int  k_w    = k_block + tx;
        const bool w_load = (q_w < length && k_w < length && k_w <= q_w); // causal

        shS[ty][tx] = w_load
            ? cache_softmax[scoreStart + q_w * headsAmount * length
                            + head * length + k_w]
            : 0.f;
        shSG[ty][tx] = w_load
            ? cache_scores_gradient[scoreStart + q_w * headsAmount * length
                                    + head * length + k_w] * scores_scale
            : 0.f;

        // ── Source arrays: ty = q_local, tx = d_local (coalesced in d) ────────
        const int  q_s    = q_tile + ty;
        const bool s_load = (q_s < length && d < headOutputSize);

        shOG[ty][tx] = s_load
            ? final_output_gradient[(sample_start + q_s) * headsAmount * headOutputSize
                                    + head * headOutputSize + d]
            : 0.f;
        shQ[ty][tx] = s_load
            ? queries[(sample_start + q_s) * headsAmount * headOutputSize
                      + head * headOutputSize + d]
            : 0.f;

        __syncthreads();

        // ── Accumulate (ty = k_local, tx = d_local) ───────────────────────────
        // shS [i][ty] = softmax    [q_tile+i, head, k_block+ty] = softmaxᵀ   [k_pos][q_frag]
        // shSG[i][ty] = score_grad·scale [q_tile+i, head, k_block+ty] = idem
        // shOG[i][tx] = output_grad[q_tile+i, head, d_block+tx]
        // shQ [i][tx] = queries    [q_tile+i, head, d_block+tx]
        for (int i = 0; i < TILE; ++i) {
            val_grad += shS[i][ty]  * shOG[i][tx];
            key_grad += shSG[i][ty] * shQ[i][tx];
        }

        __syncthreads();
    }

    if (k_pos < length && d < headOutputSize) {
        values_gradient[(sample_start + k_pos) * headsAmount * headOutputSize
                        + head * headOutputSize + d] = val_grad;
        keys_gradient[(sample_start + k_pos) * headsAmount * headOutputSize
                      + head * headOutputSize + d] = key_grad;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host entry point
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void multi_head_masked_self_attention_KV__gradient(
    const kernel_float* final_output_gradient,
    const kernel_float* queries,
    const kernel_float* cache_softmax,
    const kernel_float* cache_scores_gradient,
    kernel_float* keys_gradient,
    kernel_float* values_gradient,
    const int& batchSize,
    const int* seriesLengths,
    const int& headsAmount,
    const int& inputSize,
    const int& headOutputSize)
{
    const kernel_float scores_scale = rsqrtf((kernel_float)headOutputSize);

    int sample_start = 0;
    int score_start  = 0;

    for (int b = 0; b < batchSize; ++b) {
        const int l = seriesLengths[b];

        dim3 threads(TILE, TILE);
        dim3 blocks(
            (headOutputSize + TILE - 1) / TILE,
            (l             + TILE - 1) / TILE,
            headsAmount);

        process_KV_gradients_kernel<<<blocks, threads>>>(
            final_output_gradient,
            queries,
            cache_softmax,
            cache_scores_gradient,
            keys_gradient,
            values_gradient,
            l, score_start,
            headsAmount, headOutputSize,
            sample_start,
            scores_scale);

        sample_start += l;
        score_start  += l * headsAmount * l;
    }
}
