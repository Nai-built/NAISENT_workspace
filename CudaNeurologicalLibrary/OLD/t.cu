
__global__ void multi_head_masked_self_attention__activationKernel_OLD(
    const kernel_float* __restrict__ keys,
    const kernel_float* __restrict__ values,
    const kernel_float* __restrict__ queries,
    kernel_float* __restrict__ final_output,
    kernel_float* __restrict__ cache_scores,
    kernel_float* __restrict__ cache_softmax,
    const int batchSize, const int* __restrict__ seriesLengths, const int totalSamplesAmount,
    const int headsAmount, const int inputSize, const int headOutputSize)
{
    int token_idx = blockIdx.y * TILE + threadIdx.y;
    int out_idx   = blockIdx.x * TILE + threadIdx.x;

    if (token_idx >= totalSamplesAmount || out_idx >= headOutputSize * headsAmount) return;

    int head     = out_idx / headOutputSize;
    int head_dim = out_idx % headOutputSize;

    int batch_start = 0;
    int length = 0;
    for (int b = 0; b < batchSize; ++b) {
        int l = seriesLengths[b];
        if (batch_start + l > token_idx) { length = l; break; }
        batch_start += l;
    }
    int pos = token_idx - batch_start;

    const kernel_float scale = rsqrtf((kernel_float)headOutputSize);
    const kernel_float* __restrict__ q = queries + token_idx * headsAmount * headOutputSize + head * headOutputSize;

    const int ptr_indexing = token_idx * headsAmount * totalSamplesAmount + head * totalSamplesAmount + batch_start;
    kernel_float* __restrict__ scores_row  = cache_scores  + ptr_indexing;
    kernel_float* __restrict__ softmax_row = cache_softmax + ptr_indexing;

    // Pass 1: compute scores, write to cache_scores, find max
    kernel_float maxScore = -1e30f;
    for (int masked = 0; masked <= pos; ++masked) {
        const kernel_float* __restrict__ k = keys + (batch_start + masked) * headsAmount * headOutputSize + head * headOutputSize;
        kernel_float score = 0.0f;
        for (int e = 0; e < headOutputSize; ++e)
            score += q[e] * k[e];
        score *= scale;
        scores_row[masked] = score;
        maxScore = fmaxf(maxScore, score);
    }

    // Pass 2: compute expSum
    kernel_float expSum = 0.0f;
    for (int masked = 0; masked <= pos; ++masked) {
        scores_row[masked] = expf(scores_row[masked] - maxScore);
        expSum += scores_row[masked];
    }

    const kernel_float invExpSum = 1.0f / expSum;

    // Pass 3: normalize, write cache_softmax, accumulate weighted values
    kernel_float result = 0.0f;
    for (int masked = 0; masked <= pos; ++masked) {
        kernel_float sm_weight = scores_row[masked] * invExpSum;
        softmax_row[masked] = sm_weight;

        const kernel_float* __restrict__ v = values + (batch_start + masked) * headsAmount * headOutputSize + head * headOutputSize;
        result += sm_weight * v[head_dim];
    }

    final_output[token_idx * headsAmount * headOutputSize + out_idx] = result;
}

// OLD
// __global__ void multi_head_masked_self_attention__activationKernel(
//     const kernel_float* __restrict__ keys,
//     const kernel_float* __restrict__ values,
//     const kernel_float* __restrict__ queries,
//     kernel_float* __restrict__ final_output,
//     kernel_float* __restrict__ cache_scores,
//     kernel_float* __restrict__ cache_softmax,
    
//     const int batchSize, const int* __restrict__ seriesLengths, const int totalSamplesAmount,
//     const int headsAmount, const int inputSize, const int headOutputSize)
// {
//     int token_idx = blockIdx.y * TILE + threadIdx.y;
//     int out_idx   = blockIdx.x * TILE + threadIdx.x;

//     if (token_idx >= totalSamplesAmount || out_idx >= headOutputSize * headsAmount) return;

//     int head     = out_idx / headOutputSize;
//     int head_dim = out_idx % headOutputSize;

//     // find which batch this token belongs to, and its local position within that sequence
//     int batch_start = 0;
//     int length = 0;
//     for (int b = 0; b < batchSize; ++b) {
//         int l = seriesLengths[b];
//         if (batch_start + l > token_idx) {
//             length = l;
//             break;
//         }
//         batch_start += l;
//     }
//     int pos = token_idx - batch_start;

//     const kernel_float scale = rsqrtf((kernel_float)headOutputSize);

//     // query for this token+head: layout [token][head][dim]
//     const kernel_float* __restrict__ q = queries + token_idx * headsAmount * headOutputSize + head * headOutputSize;

//     // online softmax + weighted value accumulation (no temp buffer needed)
//     kernel_float running_max   = -1e30f;
//     kernel_float running_denom = 0.0f;
//     kernel_float result        = 0.0f;

//     for (int masked = 0; masked <= pos; ++masked) {
//         int masked_token = batch_start + masked;

//         const kernel_float* __restrict__ k = keys + masked_token * headsAmount * headOutputSize + head * headOutputSize;

//         kernel_float score = 0.0f;
//         for (int e = 0; e < headOutputSize; ++e)
//             score += q[e] * k[e];
//         score *= scale;

//         kernel_float new_max   = fmaxf(running_max, score);
//         kernel_float exp_shift = expf(running_max - new_max);  // rescale previous accumulation
//         kernel_float exp_score = expf(score - new_max);

//         const kernel_float* __restrict__ v = values + masked_token * headsAmount * headOutputSize + head * headOutputSize;

//         result        = result * exp_shift + exp_score * v[head_dim];
//         running_denom = running_denom * exp_shift + exp_score;
//         running_max   = new_max;
//     }

//     // layout [token][head][dim] = [token][out_idx]
//     final_output[token_idx * headsAmount * headOutputSize + out_idx] = result / running_denom;
// }
