#include "../../adjustments.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 — process_score_gradients
//
// Computes three things for each (q_pos, head):
//
//   (a) raw_grad[q][k]   = Σ_d output_grad[q][d] · V[k][d]   (output_grad · Vᵀ)
//       Written to cache_scores_gradient as an intermediate.
//
//   (b) softmax_dot[q]   = Σ_k raw_grad[q][k] · softmax[q][k]
//       Fused into the k_tile loop so no extra pass is needed.
//
//   (c) score_grad[q][k] = softmax[q][k] · (raw_grad[q][k] − softmax_dot[q])
//       Overwrites cache_scores_gradient (unscaled — KV kernel and kernel 2
//       both apply scores_scale on read, consistent with existing convention).
//
// Grid  : (ceil(length / TILE),  headsAmount)
// Block : (TILE, TILE)   tx = d_frag during loading / k_local during accumulation
//                        ty = q_local
//
// Shared-memory layout mirrors process_scores_kernel in the activation:
//   shOutGrad[TILE][TILE]     [q_local][d_frag]
//   shV      [TILE][TILE+1]   [k_local][d_frag], +1 padding removes bank conflicts
//   shReduce [TILE][TILE]     reused for the softmax_dot reduction
// ─────────────────────────────────────────────────────────────────────────────
__global__ void process_score_gradients_kernel(
    const kernel_float* __restrict__ final_output_gradient,
    const kernel_float* __restrict__ values,
    const kernel_float* __restrict__ cache_softmax,
    kernel_float* __restrict__ cache_scores_gradient,
    const int length,
    const int scoreStart,
    const int headsAmount,
    const int headOutputSize,
    const int sample_start)
{
    __shared__ kernel_float shOutGrad[TILE][TILE];
    __shared__ kernel_float shV[TILE][TILE + 1];   // +1 padding avoids bank conflicts
    __shared__ kernel_float shReduce[TILE][TILE];

    const int tx    = threadIdx.x;
    const int ty    = threadIdx.y;
    const int q_pos = blockIdx.x * TILE + ty;
    const int head  = blockIdx.y;

    const bool q_valid = (q_pos < length);

    // ── Phase 1: tiled output_grad · Vᵀ + fused softmax_dot accumulation ────
    kernel_float softmax_dot_partial = 0.f;

    for (int k_tile = 0; k_tile < length; k_tile += TILE) {
        kernel_float raw_grad = 0.f;

        for (int d_tile = 0; d_tile < headOutputSize; d_tile += TILE) {
            shOutGrad[ty][tx] = (q_valid && d_tile + tx < headOutputSize)
                ? final_output_gradient[(sample_start + q_pos) * headsAmount * headOutputSize
                                        + head * headOutputSize + d_tile + tx]
                : 0.f;

            // ty acts as k_local during loading (same dual-role as activation kernel)
            shV[ty][tx] = (k_tile + ty < length && d_tile + tx < headOutputSize)
                ? values[(sample_start + k_tile + ty) * headsAmount * headOutputSize
                         + head * headOutputSize + d_tile + tx]
                : 0.f;

            __syncthreads();

            // tx now acts as k_local during accumulation
            for (int i = 0; i < TILE; ++i)
                raw_grad += shOutGrad[ty][i] * shV[tx][i];

            __syncthreads();
        }

        const int  k       = k_tile + tx;
        const bool k_valid = (k < length);
        const bool causal  = (k <= q_pos);

        raw_grad = (q_valid && k_valid && causal) ? raw_grad : 0.f;

        if (q_valid && k_valid)
            cache_scores_gradient[scoreStart + q_pos * headsAmount * length
                                  + head * length + k] = raw_grad;

        // Fuse softmax_dot: read softmax[q][k] and accumulate raw_grad · softmax
        const kernel_float s = (q_valid && k_valid && causal)
            ? cache_softmax[scoreStart + q_pos * headsAmount * length + head * length + k]
            : 0.f;
        softmax_dot_partial += raw_grad * s;
    }

    // Reduce softmax_dot_partial across tx for each ty row
    shReduce[ty][tx] = softmax_dot_partial;
    __syncthreads();
    for (int stride = TILE / 2; stride >= 1; stride >>= 1) {
        if (tx < stride)
            shReduce[ty][tx] += shReduce[ty][tx + stride];
        __syncthreads();
    }
    // shReduce[ty][0] = softmax_dot for q_pos = blockIdx.x*TILE + ty

    // ── Phase 2: apply softmax backward, overwrite cache_scores_gradient ─────
    for (int k = tx; k < length; k += TILE) {
        if (q_valid && k <= q_pos) {
            const kernel_float raw = cache_scores_gradient[scoreStart + q_pos * headsAmount * length
                                                           + head * length + k];
            const kernel_float sv  = cache_softmax[scoreStart + q_pos * headsAmount * length
                                                   + head * length + k];
            cache_scores_gradient[scoreStart + q_pos * headsAmount * length
                                  + head * length + k] = sv * (raw - shReduce[ty][0]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 — process_queries_gradient
//
// Computes ∇Q = score_grad_scaled · K via tiled matmul:
//   queries_gradient[q][d] = Σ_k  cache_scores_gradient[q][k] · scores_scale · K[k][d]
//
// cache_scores_gradient holds the UNSCALED softmax-backward result; scores_scale
// is applied here (matching the convention used by the KV kernel).
//
// Grid  : (ceil(headOutputSize / TILE),  ceil(length / TILE),  headsAmount)
// Block : (TILE, TILE)   tx = d_local   ty = q_local   head = blockIdx.z
// ─────────────────────────────────────────────────────────────────────────────
__global__ void process_queries_gradient_kernel(
    const kernel_float* __restrict__ keys,
    const kernel_float* __restrict__ cache_scores_gradient,
    kernel_float* __restrict__ queries_gradient,
    const int length,
    const int scoreStart,
    const int headsAmount,
    const int headOutputSize,
    const int sample_start,
    const kernel_float scores_scale)
{
    __shared__ kernel_float shScoreGrad[TILE][TILE];
    __shared__ kernel_float shK[TILE][TILE + 1];   // +1 padding for consistency

    const int tx    = threadIdx.x;
    const int ty    = threadIdx.y;
    const int d     = blockIdx.x * TILE + tx;
    const int q_pos = blockIdx.y * TILE + ty;
    const int head  = blockIdx.z;

    kernel_float result = 0.f;

    for (int k_tile = 0; k_tile < length; k_tile += TILE) {
        // ty = q_local, tx = k_local; apply scores_scale and causal mask here
        const int k_s         = k_tile + tx;
        shScoreGrad[ty][tx] = (q_pos < length && k_s < length && k_s <= q_pos)
            ? cache_scores_gradient[scoreStart + q_pos * headsAmount * length
                                    + head * length + k_s] * scores_scale
            : 0.f;

        // ty = k_local, tx = d_local
        const int k_v  = k_tile + ty;
        shK[ty][tx] = (k_v < length && d < headOutputSize)
            ? keys[(sample_start + k_v) * headsAmount * headOutputSize
                   + head * headOutputSize + d]
            : 0.f;

        __syncthreads();

        for (int i = 0; i < TILE; ++i)
            result += shScoreGrad[ty][i] * shK[i][tx];

        __syncthreads();
    }

    if (q_pos < length && d < headOutputSize)
        queries_gradient[(sample_start + q_pos) * headsAmount * headOutputSize
                         + head * headOutputSize + d] = result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host entry point
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void multi_head_masked_self_attention_queries__gradient(
    const kernel_float* final_output_gradient,
    const kernel_float* keys,
    const kernel_float* values,
    const kernel_float* cache_softmax,
    kernel_float* cache_scores_gradient,
    kernel_float* queries_gradient,
    const int& batchSize,
    const int* seriesLengths,
    const int& headsAmount,
    const int& inputSize,
    const int& headOutputSize)
{
    const kernel_float scores_scale = rsqrtf((kernel_float)headOutputSize);

    {
        int sample_start = 0;
        int score_start  = 0;

        for (int b = 0; b < batchSize; ++b) {
            const int l = seriesLengths[b];

            dim3 threads(TILE, TILE);
            dim3 blocks((l + TILE - 1) / TILE, headsAmount);
            process_score_gradients_kernel<<<blocks, threads>>>(
                final_output_gradient,
                values,
                cache_softmax,
                cache_scores_gradient,
                l, score_start,
                headsAmount, headOutputSize,
                sample_start);
            
            sample_start += l;
            score_start  += l * headsAmount * l;
        }
    }
    cudaDeviceSynchronize();
    {
        int sample_start = 0;
        int score_start  = 0;

        for (int b = 0; b < batchSize; ++b) {
            const int l = seriesLengths[b];

            dim3 threads(TILE, TILE);
            dim3 blocks(
                (headOutputSize + TILE - 1) / TILE,
                (l             + TILE - 1) / TILE,
                headsAmount);
            process_queries_gradient_kernel<<<blocks, threads>>>(
                keys,
                cache_scores_gradient,
                queries_gradient,
                l, score_start,
                headsAmount, headOutputSize,
                sample_start,
                scores_scale);

            sample_start += l;
            score_start  += l * headsAmount * l;
        }
    }
}
