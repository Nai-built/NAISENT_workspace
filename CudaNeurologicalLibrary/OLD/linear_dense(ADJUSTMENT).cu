#include "../../adjustments.cuh"

__global__ void denseBackwardKernel(
    const kernel_float* output_gradient,   // dL/dOutput [samples x outputSize]
    const kernel_float* cached_input,      // input from forward pass [samples x inputSize]
    kernel_float* input_gradient,          // dL/dInput [samples x inputSize]
    const kernel_float* weights,           // weights [inputSize x outputSize]
    kernel_float* weights_gradient,        // dL/dW [inputSize x outputSize]
    kernel_float* biases_gradient,         // dL/db [outputSize]
    const int samplesAmount,
    const int inputSize,
    const int outputSize
)
{
    int threadX = threadIdx.x;  // output neuron index within block
    int threadY = threadIdx.y;  // sample index within block

    int outIdx = blockIdx.x * TILE + threadX;
    int sampleIdx = blockIdx.y * TILE + threadY;

    // -------------------------
    // 1️⃣ Bias gradient (atomicAdd per sample)
    // -------------------------
    if (outIdx < outputSize && sampleIdx < samplesAmount)
    {
        kernel_float g = output_gradient[sampleIdx * outputSize + outIdx];
        atomicAdd(&biases_gradient[outIdx], g);
    }

    __syncthreads();

    // -------------------------
    // 2️⃣ Loop over input tiles for weights and input gradients
    // -------------------------
    for (int inputTileStart = 0; inputTileStart < inputSize; inputTileStart += TILE)
    {
        __shared__ kernel_float tileInput[TILE][TILE];
        // __shared__ kernel_float tileWeights[TILE][TILE];

        // Load input tile safely
        if (sampleIdx < samplesAmount && (inputTileStart + threadX) < inputSize)
            tileInput[threadY][threadX] = cached_input[sampleIdx * inputSize + (inputTileStart + threadX)];
        else
            tileInput[threadY][threadX] = 0.0f;
        // Load weights tile safely
        // if (outIdx < outputSize && (inputTileStart + threadY) < inputSize)
        //     tileWeights[threadY][threadX] = weights[(inputTileStart + threadY) * outputSize + outIdx];
        // else
        //     tileWeights[threadY][threadX] = 0.0f;

        __syncthreads();

        if (sampleIdx < samplesAmount && outIdx < outputSize)
        {
            kernel_float g = output_gradient[sampleIdx * outputSize + outIdx];

            // Compute weight gradient for each input in the tile
            if (inputTileStart + threadY < inputSize)
            {
                kernel_float grad = tileInput[threadY][threadX] * g;
                atomicAdd(&weights_gradient[(inputTileStart + threadY) * outputSize + outIdx], grad);
            }
            // Compute input gradient for each weight in the tile
            // if (inputTileStart + threadX < inputSize)
            // {
            //     kernel_float inputGrad = tileWeights[threadY][threadX] * g;
            //     atomicAdd(&input_gradient[sampleIdx * inputSize + (inputTileStart + threadX)], inputGrad);
            // }

            // Compute input gradient
            for (int i = 0; i < TILE; ++i)
            {
                int input_i = inputTileStart + i;
                if (input_i < inputSize)
                {
                    kernel_float w = weights[input_i * outputSize + outIdx];
                    atomicAdd(&input_gradient[sampleIdx * inputSize + input_i], w * g);
                }
            }
        }

        __syncthreads();
    }
}

__global__ void linear_dense__adjustmentKernel(const kernel_float* output_gradient
    , const kernel_float* cached_input, kernel_float* input_gradient
    , const kernel_float* weights, kernel_float* weights_gradient
    , kernel_float* biases_gradient

    , const int samplesAmount, const int inputSize, const int outputSize)
{
    __shared__ kernel_float tiledCached_input [TILE][TILE];

    __shared__ kernel_float tiledWeights [TILE][TILE];

    __shared__ kernel_float tiledBiases_gradient [TILE];

    __shared__ kernel_float tiledInput_gradient [TILE][TILE];
    __shared__ kernel_float tiledWeights_gradient [TILE][TILE];

    int y = blockIdx.y*TILE + threadIdx.y; // current sample unit
    int x = blockIdx.x*TILE + threadIdx.x; // current sample value

    int threadX = threadIdx.x; // does NOT necessarily represent a specific "neuron" location
    int threadY = threadIdx.y; // does NOT necessarily represent batching or a specific sample index

    // initialize tiled biases gradient
    if (threadY == 0) {
        tiledBiases_gradient[threadX] = 0.0f;
    }
    __syncthreads();

    // get output gradient and assign it to biases
    const kernel_float g = (x < outputSize && y < samplesAmount) ? output_gradient[y*outputSize + x] : 0.0f;
    atomicAdd(&tiledBiases_gradient[threadX], g);
    __syncthreads();
    
    // update biases gradient to main memory
    if (threadY == 0) {
        if (x < outputSize) {
            atomicAdd(&biases_gradient[x], tiledBiases_gradient[threadX]);
        }
    }

    for (int inputTiling = 0; inputTiling < inputSize; inputTiling += TILE) {
        // get cached input and weights from main memory to block shared memory
        if (y < samplesAmount && inputTiling+threadX < inputSize) {
            tiledCached_input[threadY][threadX] = cached_input[y*inputSize + inputTiling+threadX];
        } else tiledCached_input[threadY][threadX] = 0.0f;
        if (x < outputSize && inputTiling+threadY < inputSize) {
            tiledWeights[threadY][threadX] = weights[(inputTiling+threadY)*outputSize + x];
        } else tiledWeights[threadY][threadX] = 0.0f;

        // initialize tiled input and weights gradients
        tiledInput_gradient[threadY][threadX] = 0.0f;
        tiledWeights_gradient[threadY][threadX] = 0.0f;
        __syncthreads();

        // compute weights and input gradients
        for (int i = 0; i < TILE; ++i)
        {
            atomicAdd(&tiledWeights_gradient[i][threadX], tiledCached_input[threadY][i] * g);
            atomicAdd(&tiledInput_gradient[threadY][i], tiledWeights[i][threadX] * g);
        }
        __syncthreads();
        
        // update weights and input gradients to main memory
        if (threadY == 0) {
            if (x < outputSize && inputTiling+threadY < inputSize) {
                atomicAdd(&weights_gradient[(inputTiling+threadY)*outputSize + x],
                            tiledWeights_gradient[threadY][threadX]);
            }
        }
        if (threadX == 0) {
            if (y < samplesAmount && inputTiling+threadX < inputSize) {
                atomicAdd(&input_gradient[y*inputSize + inputTiling+threadX],
                            tiledInput_gradient[threadY][threadX]);
            }
        }
    }
}

extern "C" void linear_dense__adjustment(const kernel_float* output_gradient
    , const kernel_float* cached_input, kernel_float* input_gradient
    , const kernel_float* weights, kernel_float* weights_gradient
    , kernel_float* biases_gradient
    , const int& samplesAmount, const int& inputSize, const int& outputSize)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(
        (outputSize + TILE - 1) / TILE,
        (samplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    denseBackwardKernel<<<blocks, threads>>>(
        output_gradient,
        cached_input, input_gradient,
        weights, weights_gradient,
        biases_gradient,
        samplesAmount, inputSize, outputSize
    );
    cudaDeviceSynchronize();
}