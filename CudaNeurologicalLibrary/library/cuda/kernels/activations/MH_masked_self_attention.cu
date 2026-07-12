// #include "../../activations.cuh"

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

// // GRID: [masked, head, sample]
// __global__ void compute_scores(const kernel_float* __restrict__ queries, const kernel_float* __restrict__ keys
//     , kernel_float* __restrict__ scores, kernel_float* __restrict__ score_max
//     , const int tensorSize, const int headsAmount
//     , const int* lengths, const int batchSize, const int totalSamples)
// {
//     __shared__ kernel_float tiledQueries[TILE][TILE][TILE];
//     __shared__ kernel_float tiledKeys[TILE][TILE][TILE];

//     const int x = blockIdx.x*TILE + threadIdx.x; // MASKED
//     const int y = blockIdx.y*TILE + threadIdx.y; // HEAD
//     const int z = blockIdx.z*TILE + threadIdx.z; // SAMPLE

//     const int threadX = threadIdx.x;
//     const int threadY = threadIdx.y;
//     const int threadZ = threadIdx.z;

//     // FIGURE OUT CORRECT ARGUEMENTS
//     int length = 0;
//     int batch_start = 0;
//     int score_start = 0;
//     int max_score_start = 0;
//     for (int _ = 0; _ < batchSize; ++_) {
//         const int _l = lengths[_];
//         if (batch_start+_l > z) {
//             length = _l;
//             break;
//         }
//         batch_start += _l;
//         score_start += _l*headsAmount+_l;
//         max_score_start += _l*headsAmount;
//     }
//     const int local_x = x-batch_start;
//     const int local_z = z-batch_start;

//     kernel_float score = 0.0f;

//     for (int tensorTiling = 0; tensorTiling < tensorSize; tensorTiling += TILE) {
//         if (local_x < length && y < headsAmount && tensorTiling+threadZ < tensorSize) {
//             tiledKeys[threadX][threadY][threadZ] = keys[x*headsAmount*tensorSize + y*tensorSize + tensorTiling+threadZ];
//         } else tiledKeys[threadX][threadY][threadZ] = 0.0f;
//         if (local_z < length && y < headsAmount && tensorTiling+threadX < tensorSize) {
//             tiledQueries[threadX][threadY][threadZ] = queries[(z)*headsAmount*tensorSize + y*tensorSize + tensorTiling+threadX];
//         } else tiledQueries[threadX][threadY][threadZ] = 0.0f;
//         __syncthreads();

//         if (local_x <= local_z) {
//             for (int i = 0; i < TILE; ++i) {
//                 score += tiledKeys[threadX][threadY][i]*tiledQueries[i][threadY][threadZ];
//             }
//         }
//         __syncthreads();
//     }

//     if (z < length && y < headsAmount && local_x <= local_z) {
//         scores[score_start + (local_z)*headsAmount*(length) + y*(length) + local_x] = score;
//         atomicMaxFloat(&score_max[max_score_start + (local_z)*headsAmount*(length) + y*(length)], score);
//     }
// }
// // GRID: [masked, head, sample]
// __global__ void compute_exponents(kernel_float* __restrict__ scores, const kernel_float* __restrict__ score_max
//     , kernel_float* __restrict__ exponent_score_sums
//     , const int tensorSize, const int headsAmount
//     , const int* lengths, const int batchSize, const int totalSamples)
// {
//     __shared__ kernel_float tiledMax_scores[TILE][TILE];

//     const int x = blockIdx.x*TILE + threadIdx.x; // MASKED
//     const int y = blockIdx.y*TILE + threadIdx.y; // HEAD
//     const int z = blockIdx.z*TILE + threadIdx.z; // SAMPLE

//     const int threadX = threadIdx.x;
//     const int threadY = threadIdx.y;
//     const int threadZ = threadIdx.z;

//     // FIGURE OUT CORRECT ARGUEMENTS
//     int length = 0;
//     int batch_start = 0;
//     int score_start = 0;
//     int max_score_start = 0;
//     for (int _ = 0; _ < batchSize; ++_) {
//         const int _l = lengths[_];
//         if (batch_start+_l > z) {
//             length = _l;
//             break;
//         }
//         batch_start += _l;
//         score_start += _l*headsAmount+_l;
//         max_score_start += _l*headsAmount;
//     }
//     const int local_x = x-batch_start;
//     const int local_z = z-batch_start;

//     if (threadX == 0) {
//         if (local_z < length && y < headsAmount) {
//             tiledMax_scores[threadY][threadZ] = score_max[max_score_start + (local_z)*headsAmount*(length) + y*(length)];
//         } tiledMax_scores[threadY][threadZ] = 0.0f;
//     }
//     __syncthreads();

//     if (local_z < length && y < headsAmount && local_x <= local_z) {
//         kernel_float exponent = exp(scores[score_start + (local_z)*headsAmount*(length) + y*(length) + local_x]-tiledMax_scores[threadY][threadZ]);
//         scores[score_start + (local_z)*headsAmount*(length) + y*(length) + local_x] = exponent;
//         atomicAdd(&exponent_score_sums[max_score_start + (local_z)*headsAmount*(length) + y*(length)], exponent);
//     }
// }
// // GRID: [masked, head, sample]
// __global__ void compute_softmax(const kernel_float* __restrict__ exponent_scores, const kernel_float* __restrict__ exponent_score_sums
//     , kernel_float* __restrict__ softmax
//     , const int tensorSize, const int headsAmount
//     , const int* lengths, const int batchSize, const int totalSamples)
// {
//     __shared__ kernel_float tiledExponent_sum_multipliers[TILE][TILE];

//     const int x = blockIdx.x*TILE + threadIdx.x; // MASKED
//     const int y = blockIdx.y*TILE + threadIdx.y; // HEAD
//     const int z = blockIdx.z*TILE + threadIdx.z; // SAMPLE

//     const int threadX = threadIdx.x;
//     const int threadY = threadIdx.y;
//     const int threadZ = threadIdx.z;

//     // FIGURE OUT CORRECT ARGUEMENTS
//     int length = 0;
//     int batch_start = 0;
//     int score_start = 0;
//     int max_score_start = 0;
//     for (int _ = 0; _ < batchSize; ++_) {
//         const int _l = lengths[_];
//         if (batch_start+_l > z) {
//             length = _l;
//             break;
//         }
//         batch_start += _l;
//         score_start += _l*headsAmount+_l;
//         max_score_start += _l*headsAmount;
//     }
//     const int local_x = x-batch_start;
//     const int local_z = z-batch_start;

//     if (threadX == 0) {
//         if (z < length && y < headsAmount) {
//             tiledExponent_sum_multipliers[threadY][threadZ] = 1.0f / exponent_score_sums[(z)*headsAmount*(pre_length+length) + y*(pre_length+length)];
//         } tiledExponent_sum_multipliers[threadY][threadZ] = 0.0f;
//     }
//     __syncthreads();

//     if (z < length && y < headsAmount && x <= pre_length+z) {
//         kernel_float softmax_value = exponent_scores[(z)*headsAmount*(pre_length+length) + y*(pre_length+length) + x]*tiledExponent_sum_multipliers[threadY][threadZ];
//         softmax[(z)*headsAmount*(pre_length+length) + y*(pre_length+length) + x] = softmax_value;
//     }
// }