#include "../../activations.cuh"

#include <iostream>

// important notes because we're just tweaking out at this point -_-
// first: we forgot the ELSE KETWORD!! how could I forget that?
// second: this is actually very important to remember, when we process the output with sequenceTiling.. I was trying to "mask" the for loop itself with the index z, that however, doesn't work.. we need to make all threads able to reach each __syncthread() line so just let it go properly through every sample because the inner masking is already enough.
// claude code was what caught these errors btw o-o

__device__ void atomicMaxFloat(float* addr, float val) {
    unsigned int* uaddr = (unsigned int*)addr;
    unsigned int val_bits = __float_as_uint(val);
    // mask = 0x80000000 for positive, 0xFFFFFFFF for negative
    unsigned int val_key = val_bits ^ (-(val_bits >> 31) | 0x80000000u);

    unsigned int old = *uaddr;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned int old_key = assumed ^ (-(assumed >> 31) | 0x80000000u);
        if (val_key <= old_key) break;  // already have a larger value stored
        old = atomicCAS(uaddr, assumed, val_bits);
    } while (old != assumed);
}
// __device__ float atomicMaxFloat(float* address, float val) {
//     int* address_as_int = (int*)address;
//     int old = *address_as_int, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_int, assumed,
//                         __float_as_int(fmaxf(val, __int_as_float(assumed))));
//     } while (assumed != old);
//     return __int_as_float(old);
// }

// score caches and its "family" will be reset every inference step!
// so access it directly through the sample index without adding pre length

// GRID: [head, sample]
__global__ void initialize_score_max(kernel_float* __restrict__ score_max
    , const int length, const int headsAmount
    , const int pre_length)
{
    const int x = blockIdx.x*TILE + threadIdx.x; // HEAD
    const int y = blockIdx.y*TILE + threadIdx.y; // SAMPLE

    if (y < length && x < headsAmount) {
        score_max[(y)*headsAmount + x] = -1e38f;
    }
}

// GRID: [masked, head, sample]
__global__ void compute_scores(const kernel_float* __restrict__ queries, const kernel_float* __restrict__ keys
    , kernel_float* __restrict__ scores, kernel_float* __restrict__ score_max
    , const int tensorSize, const int length, const int headsAmount
    , const kernel_float scores_scale
    , const int pre_length)
{
    __shared__ kernel_float tiledQueries[TILE][TILE][TILE];
    __shared__ kernel_float tiledKeys[TILE][TILE][TILE];

    const int x = blockIdx.x*TILE + threadIdx.x; // MASKED
    const int y = blockIdx.y*TILE + threadIdx.y; // HEAD
    const int z = blockIdx.z*TILE + threadIdx.z; // SAMPLE

    const int threadX = threadIdx.x;
    const int threadY = threadIdx.y;
    const int threadZ = threadIdx.z;

    kernel_float score = 0.0f;

    for (int tensorTiling = 0; tensorTiling < tensorSize; tensorTiling += TILE) {
        if (x < pre_length+length && y < headsAmount && tensorTiling+threadZ < tensorSize) {
            tiledKeys[threadX][threadY][threadZ] = keys[x*headsAmount*tensorSize + y*tensorSize + tensorTiling+threadZ];
        } else tiledKeys[threadX][threadY][threadZ] = 0.0f;
        if (z < length && y < headsAmount && tensorTiling+threadX < tensorSize) {
            tiledQueries[threadX][threadY][threadZ] = queries[(z)*headsAmount*tensorSize + y*tensorSize + tensorTiling+threadX];
        } else tiledQueries[threadX][threadY][threadZ] = 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; ++i) {
            score += tiledKeys[threadX][threadY][i]*tiledQueries[i][threadY][threadZ];
        }
        __syncthreads();
    }

    if (z < length && y < headsAmount && x <= pre_length+z) {
        score *= scores_scale;
        scores[(z)*headsAmount*(pre_length+length) + y*(pre_length+length) + x] = score;
        // printf("score (%i, %i): %f\n", y,z, score);
        atomicMaxFloat(&score_max[(z)*headsAmount + y], score);
    }
}
// GRID: [masked, head, sample]
__global__ void compute_exponents(kernel_float* __restrict__ scores, const kernel_float* __restrict__ score_max
    , kernel_float* __restrict__ exponent_score_sums
    , const int length, const int headsAmount
    , const int pre_length)
{
    __shared__ kernel_float tiledMax_scores[TILE][TILE];

    const int x = blockIdx.x*TILE + threadIdx.x; // MASKED
    const int y = blockIdx.y*TILE + threadIdx.y; // HEAD
    const int z = blockIdx.z*TILE + threadIdx.z; // SAMPLE

    const int threadX = threadIdx.x;
    const int threadY = threadIdx.y;
    const int threadZ = threadIdx.z;

    if (threadX == 0) {
        if (z < length && y < headsAmount) {
            // printf("score max (%i, %i): %f\n", y,z, score_max[(z)*headsAmount*(pre_length+length) + y*(pre_length+length)]);
            tiledMax_scores[threadY][threadZ] = score_max[(z)*headsAmount + y];
        } else tiledMax_scores[threadY][threadZ] = 0.0f;
    }
    __syncthreads();

    if (z < length && y < headsAmount && x <= pre_length+z) {
        kernel_float exponent = expf(scores[(z)*headsAmount*(pre_length+length) + y*(pre_length+length) + x]-tiledMax_scores[threadY][threadZ]);
        scores[(z)*headsAmount*(pre_length+length) + y*(pre_length+length) + x] = exponent;
        atomicAdd(&exponent_score_sums[(z)*headsAmount + y], exponent);
    }
}
// GRID: [masked, head, sample]
__global__ void compute_softmax(const kernel_float* __restrict__ exponent_scores, const kernel_float* __restrict__ exponent_score_sums
    , kernel_float* __restrict__ softmax
    , const int length, const int headsAmount
    , const int pre_length)
{
    __shared__ kernel_float tiledExponent_sum_multipliers[TILE][TILE];

    const int x = blockIdx.x*TILE + threadIdx.x; // MASKED
    const int y = blockIdx.y*TILE + threadIdx.y; // HEAD
    const int z = blockIdx.z*TILE + threadIdx.z; // SAMPLE

    const int threadX = threadIdx.x;
    const int threadY = threadIdx.y;
    const int threadZ = threadIdx.z;

    if (threadX == 0) {
        if (z < length && y < headsAmount) {
            tiledExponent_sum_multipliers[threadY][threadZ] = 1.0f / exponent_score_sums[(z)*headsAmount + y];
        } else tiledExponent_sum_multipliers[threadY][threadZ] = 0.0f;
    }
    __syncthreads();

    if (z < length && y < headsAmount && x <= pre_length+z) {
        kernel_float softmax_value = exponent_scores[(z)*headsAmount*(pre_length+length) + y*(pre_length+length) + x]*tiledExponent_sum_multipliers[threadY][threadZ];
        softmax[(z)*headsAmount*(pre_length+length) + y*(pre_length+length) + x] = softmax_value;
    }
}

// GRID: [value, head, sample]
__global__ void compute_output(kernel_float* __restrict__ output
    , const kernel_float* __restrict__ values, const kernel_float* __restrict__ softmax
    , const int tensorSize, const int length, const int headsAmount
    , const int pre_length)
{
    __shared__ kernel_float tiledValues[TILE][TILE][TILE];
    __shared__ kernel_float tiledSoftmax[TILE][TILE][TILE];

    const int x = blockIdx.x*TILE + threadIdx.x; // VALUE
    const int y = blockIdx.y*TILE + threadIdx.y; // HEAD
    const int z = blockIdx.z*TILE + threadIdx.z; // SAMPLE

    const int threadX = threadIdx.x;
    const int threadY = threadIdx.y;
    const int threadZ = threadIdx.z;

    kernel_float result = 0.0;

    for (int sequenceTiling = 0; sequenceTiling < pre_length+length; sequenceTiling += TILE) {
        if (z < length && y < headsAmount && sequenceTiling+threadX < pre_length+length) {
            tiledSoftmax[threadZ][threadY][threadX] = softmax[(z)*headsAmount*(pre_length+length) + y*(pre_length+length) + sequenceTiling+threadX];
        } else tiledSoftmax[threadZ][threadY][threadX] = 0.0f;
        if (sequenceTiling+threadZ < pre_length+length && y < headsAmount && x < tensorSize) {
            tiledValues[threadZ][threadY][threadX] = values[(sequenceTiling+threadZ)*headsAmount*tensorSize + y*tensorSize + x];
        } else tiledValues[threadZ][threadY][threadX] = 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; ++i) {
            if (sequenceTiling+i <= pre_length+z) {
                result += tiledSoftmax[threadZ][threadY][i]*tiledValues[i][threadY][threadX];
            }
        }
        __syncthreads();
    }

    if (z < length && y < headsAmount && x < tensorSize) {
        output[z*headsAmount*tensorSize + y*tensorSize + x] = result;
    }
}

// LAUNCHER
extern "C" void MH_masked_self_attention_inference__activation(
    const kernel_float* keys,
    const kernel_float* values,
    const kernel_float* queries,
    kernel_float* final_output,
    kernel_float* cache_scores,
    kernel_float* cache_score_max,
    kernel_float* cache_exponent_sums,
    kernel_float* cache_softmax,
    const int& pre_length,
    const int& length,
    const int& headsAmount,
    const int& inputSize,
    const int& headOutputSize)
{
    const kernel_float scores_scale = rsqrtf((kernel_float)headOutputSize);

    {
        dim3 threads(TILE, TILE);
        dim3 blocks((headsAmount + TILE - 1) / TILE, (length + TILE - 1) / TILE);
        initialize_score_max<<<blocks, threads>>>(
            cache_score_max,
            length,
            headsAmount,
            pre_length
        );
    }
    cudaDeviceSynchronize();
    {
        dim3 threads(TILE, TILE, TILE);
        dim3 blocks((pre_length+length + TILE - 1) / TILE, (headsAmount + TILE - 1) / TILE, (length + TILE - 1) / TILE);
        compute_scores<<<blocks, threads>>>(
            queries,
            keys,
            cache_scores,
            cache_score_max,
            headOutputSize,
            length,
            headsAmount,
            scores_scale,
            pre_length
        );
    }
    cudaDeviceSynchronize();
    {
        dim3 threads(TILE, TILE, TILE);
        dim3 blocks((pre_length+length + TILE - 1) / TILE, (headsAmount + TILE - 1) / TILE, (length + TILE - 1) / TILE);
        compute_exponents<<<blocks, threads>>>(
            cache_scores,
            cache_score_max,
            cache_exponent_sums,
            length,
            headsAmount,
            pre_length
        );
    }
    cudaDeviceSynchronize();
    {
        dim3 threads(TILE, TILE, TILE);
        dim3 blocks((pre_length+length + TILE - 1) / TILE, (headsAmount + TILE - 1) / TILE, (length + TILE - 1) / TILE);
        compute_softmax<<<blocks, threads>>>(
            cache_scores,
            cache_exponent_sums,
            cache_softmax,
            length,
            headsAmount,
            pre_length
        );
    }
    cudaDeviceSynchronize();
    {
        dim3 threads(TILE, TILE, TILE);
        dim3 blocks((headOutputSize + TILE - 1) / TILE, (headsAmount + TILE - 1) / TILE, (length + TILE - 1) / TILE);
        compute_output<<<blocks, threads>>>(
            final_output,
            values,
            cache_softmax,
            headOutputSize,
            length,
            headsAmount,
            pre_length
        );
    }
    cudaDeviceSynchronize();
}

// for now, I believe we finished the inference kernel!
// now we need to start on rebuilding the training kernels but this time make sure they support batching properly