#include "../../activations.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 — process_scores
//
// Computes the full scaled-dot-product score matrix for one batch sample,
// applies the causal mask, runs numerically-stable softmax, and writes both
// the raw softmax weights (cache_softmax) and the intermediate exp values
// (cache_scores, overwritten in-place during the softmax passes).
//
// Grid  : (ceil(length / TILE),  headsAmount)
// Block : (TILE, TILE)   tx = k_local / d_frag (dual role, see below)
//                        ty = q_local
//
// Shared-memory dual-role trick
// ─────────────────────────────
// During the tiled dot-product, each thread (ty, tx) loads:
//   shQ[ty][tx] = Q[q_pos,  head, d_tile + tx]   (ty = q_local, tx = d_frag)
//   shK[ty][tx] = K[k_tile + ty, head, d_tile + tx]   (ty = k_local, tx = d_frag)
// Then accumulates:
//   score += shQ[ty][i] * shK[tx][i]    (ty = q_local, tx = k_local)
// shK[tx][i] accesses row tx (= k_local), column i (= d_frag), which maps
// correctly because the loading used ty as the k-index for rows.
//
// shK is padded by one column to eliminate the stride-TILE bank-conflict
// pattern that arises when multiple threads read different rows of shK at the
// same column offset.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void process_scores_kernel(
    const kernel_float* __restrict__ keys,
    const kernel_float* __restrict__ queries,
    kernel_float* __restrict__ cache_scores,
    kernel_float* __restrict__ cache_softmax,
    const int length,
    const int scoreStart,
    const int headsAmount,
    const int headOutputSize,
    const int sample_start,
    const kernel_float scores_scale)
{
    __shared__ kernel_float shQ[TILE][TILE];
    __shared__ kernel_float shK[TILE][TILE + 1]; // +1 col padding avoids bank conflicts
    __shared__ kernel_float shReduce[TILE][TILE]; // reused for max and sum reductions
    __shared__ kernel_float shRunMax[TILE];        // running per-row maximum

    const int tx    = threadIdx.x;
    const int ty    = threadIdx.y;
    const int q_pos = blockIdx.x * TILE + ty;
    const int head  = blockIdx.y;

    const bool q_valid = (q_pos < length);

    if (tx == 0)
        shRunMax[ty] = -1e38f;
    __syncthreads();

    // ── Phase 1: tiled Q·Kᵀ, write scores, track running row-maximum ─────────
    for (int k_tile = 0; k_tile < length; k_tile += TILE) {
        kernel_float score = 0.f;

        for (int d_tile = 0; d_tile < headOutputSize; d_tile += TILE) {
            shQ[ty][tx] = (q_valid && d_tile + tx < headOutputSize)
                ? queries[(sample_start + q_pos) * headsAmount * headOutputSize
                          + head * headOutputSize + d_tile + tx]
                : 0.f;

            // ty acts as k_local during loading
            shK[ty][tx] = (k_tile + ty < length && d_tile + tx < headOutputSize)
                ? keys[(sample_start + k_tile + ty) * headsAmount * headOutputSize
                       + head * headOutputSize + d_tile + tx]
                : 0.f;

            __syncthreads();

            // tx now acts as k_local during accumulation
            for (int i = 0; i < TILE; ++i)
                score += shQ[ty][i] * shK[tx][i];

            __syncthreads();
        }

        const int k            = k_tile + tx;
        const bool k_valid     = (k < length);
        const bool causal      = (k <= q_pos);
        const kernel_float val = (q_valid && k_valid && causal)
                                 ? score * scores_scale
                                 : -1e38f;

        if (q_valid && k_valid)
            cache_scores[scoreStart + q_pos * headsAmount * length + head * length + k] = val;

        // Reduce tile maximum across tx for each ty row
        shReduce[ty][tx] = val;
        __syncthreads();
        for (int stride = TILE / 2; stride >= 1; stride >>= 1) {
            if (tx < stride)
                shReduce[ty][tx] = max(shReduce[ty][tx], shReduce[ty][tx + stride]);
            __syncthreads();
        }
        if (tx == 0)
            shRunMax[ty] = max(shRunMax[ty], shReduce[ty][0]);
        __syncthreads();
    }

    // ── Phase 2: exp(score − max), accumulate sum ─────────────────────────────
    // tx strides over k, giving TILE-way parallelism per q_pos row
    kernel_float thread_sum = 0.f;
    for (int k = tx; k < length; k += TILE) {
        if (q_valid && k <= q_pos) {
            const kernel_float e = expf(
                cache_scores[scoreStart + q_pos * headsAmount * length + head * length + k]
                - shRunMax[ty]);
            cache_scores[scoreStart + q_pos * headsAmount * length + head * length + k] = e;
            thread_sum += e;
        }
    }

    shReduce[ty][tx] = thread_sum;
    __syncthreads();
    for (int stride = TILE / 2; stride >= 1; stride >>= 1) {
        if (tx < stride)
            shReduce[ty][tx] += shReduce[ty][tx + stride];
        __syncthreads();
    }

    // ── Phase 3: normalize, write softmax weights ─────────────────────────────
    const kernel_float inv_sum = (shReduce[ty][0] > 0.f) ? 1.f / shReduce[ty][0] : 0.f;
    for (int k = tx; k < length; k += TILE) {
        if (q_valid && k <= q_pos) {
            cache_softmax[scoreStart + q_pos * headsAmount * length + head * length + k] =
                cache_scores[scoreStart + q_pos * headsAmount * length + head * length + k]
                * inv_sum;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 — process_output
//
// Computes output = softmax_weights · V for one batch sample using a standard
// tiled matmul.  Masked positions (k > q_pos) are already 0 in cache_softmax,
// so no explicit mask branch is needed during accumulation.
//
// Grid  : (ceil(headOutputSize / TILE),  ceil(length / TILE),  headsAmount)
// Block : (TILE, TILE)   tx = d_local  (output-head dimension)
//                        ty = q_local
// ─────────────────────────────────────────────────────────────────────────────
__global__ void process_output_kernel(
    const kernel_float* __restrict__ values,
    const kernel_float* __restrict__ cache_softmax,
    kernel_float* __restrict__ final_output,
    const int length,
    const int scoreStart,
    const int headsAmount,
    const int headOutputSize,
    const int sample_start)
{
    __shared__ kernel_float shSoftmax[TILE][TILE];
    __shared__ kernel_float shV[TILE][TILE + 1]; // +1 col padding for consistency

    const int tx    = threadIdx.x;
    const int ty    = threadIdx.y;
    const int d     = blockIdx.x * TILE + tx;
    const int q_pos = blockIdx.y * TILE + ty;
    const int head  = blockIdx.z;

    kernel_float result = 0.f;

    for (int k_tile = 0; k_tile < length; k_tile += TILE) {
        // ty = q_local, tx = k_local within tile
        const int k_s     = k_tile + tx;
        shSoftmax[ty][tx] = (q_pos < length && k_s < length && k_s <= q_pos)
            ? cache_softmax[scoreStart + q_pos * headsAmount * length + head * length + k_s]
            : 0.f;

        // ty = k_local within tile, tx = d_local
        const int k_v  = k_tile + ty;
        shV[ty][tx] = (k_v < length && d < headOutputSize)
            ? values[(sample_start + k_v) * headsAmount * headOutputSize
                     + head * headOutputSize + d]
            : 0.f;

        __syncthreads();

        for (int i = 0; i < TILE; ++i)
            result += shSoftmax[ty][i] * shV[i][tx];

        __syncthreads();
    }

    if (q_pos < length && d < headOutputSize)
        final_output[(sample_start + q_pos) * headsAmount * headOutputSize
                     + head * headOutputSize + d] = result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host entry point
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void multi_head_masked_self_attention__activation(
    const kernel_float* keys,
    const kernel_float* values,
    const kernel_float* queries,
    kernel_float* final_output,
    kernel_float* cache_scores,
    kernel_float* cache_softmax,
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
            process_scores_kernel<<<blocks, threads>>>(
                keys, queries,
                cache_scores, cache_softmax,
                l, score_start,
                headsAmount, headOutputSize,
                sample_start, scores_scale);
            
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
                (l         + TILE - 1) / TILE,
                headsAmount);
            process_output_kernel<<<blocks, threads>>>(
                values, cache_softmax,
                final_output,
                l, score_start,
                headsAmount, headOutputSize,
                sample_start);

            sample_start += l;
            score_start  += l * headsAmount * l;
        }
    }
}
