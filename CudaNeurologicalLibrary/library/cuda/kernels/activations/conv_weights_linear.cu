#include "../../activations.cuh"

#include <iostream>

// this is a super naive convolutional activation kernel that computes both weights and biases
// next step is to turn it into a much more optimized and performance friendly kernel using techs like shared memory
// also don't forget to split it into 2 kernels (one for weights and another for biases)
// why? because processing biases with convolutional weights is unoptimal, considering each value in a single channel gets the exact same bias parameter. unlike dense layers of course.
// nevermind, after deeper thinking.. we need merge the activation with the weights kernel for better performance
// and so, we will still make it into 2 kernels but a bit different (first one: applying biases values, and the second one: weights computation AND activation functions)

// __global__ void convolutional_filters_linear__activationKernelNaive(const kernel_float* __restrict__ input
//     , kernel_float* __restrict__ output
//     , const kernel_float* __restrict__ weights
//     , const kernel_float* __restrict__ biases

//     , const int samplesAmount, const int inputChannels, const int outputChannels

//     , const int inputWidth, const int outputWidth
//     , const int inputHeight, const int outputHeight

//     , const int slideWidth, const int slideHeight

//     , const int stride, const int padding)
// {
//     const int gridX = blockIdx.x*TILE + threadIdx.x;
//     const int gridY = blockIdx.y*TILE + threadIdx.y;

//     // FIGURING OUT LOCATIONS
//     const int channel = gridX/(outputWidth*outputHeight);
//     const int channelFlatPoint = gridX-(channel*outputWidth*outputHeight);

//     const int y = channelFlatPoint/outputWidth;
//     const int x = channelFlatPoint-(y*outputWidth);

//     kernel_float result = 0.0f;

//     if (gridY < samplesAmount && channel < outputChannels && y < outputHeight && x < outputWidth) {
//         // NAIVE ACCOMULATION LOOP
//         for (int i = 0; i < inputChannels; ++i) {
//             for (int sY = 0; sY < slideHeight; ++sY) {
//                 int iY = (y*stride - padding)+sY;
//                 if (iY >= inputHeight || iY < 0) continue;
//                 for (int sX = 0; sX < slideWidth; ++sX) {
//                     int iX = (x*stride - padding)+sX;
//                     if (iX >= inputWidth || iX < 0) continue;

//                     result +=
//                         weights[i*outputChannels*slideWidth*slideHeight +
//                             channel*slideWidth*slideHeight + sY*slideWidth + sX]
//                         * input[gridY*inputChannels*inputWidth*inputHeight +
//                             i*inputWidth*inputHeight + iY*inputWidth + iX];
//                 }
//             }
//         }

//         output[gridY*outputChannels*outputWidth*outputHeight
//             + channel*outputWidth*outputHeight + y*outputWidth + x] += result;
//     }
// }

// __global__ void convolutional_weights_linear__activationKernel(const kernel_float* __restrict__ input
//     , kernel_float* __restrict__ output
//     , const kernel_float* __restrict__ weights
//     , const kernel_float* __restrict__ biases

//     , const int samplesAmount, const int inputChannels, const int outputChannels

//     , const int inputWidth, const int outputWidth
//     , const int inputHeight, const int outputHeight

//     , const int slideWidth, const int slideHeight

//     , const int stride, const int padding)
// {
//     __shared__ kernel_float tiledInput[TILE][TILE][TILE];
//     __shared__ kernel_float tiledWeights[TILE][TILE][TILE];

//     const int gridX = blockIdx.x*TILE + threadIdx.x; // VALUE
//     const int gridY = blockIdx.y*TILE + threadIdx.y; // CHANNEL
//     const int gridZ = blockIdx.z*TILE + threadIdx.z; // SAMPLE

//     const int threadX = threadIdx.x;
//     const int threadY = threadIdx.y;
//     const int threadZ = threadIdx.z;

//     // FIGURING OUT LOCATIONS
//     const int y = gridX/outputWidth;
//     const int x = gridX-(y*outputWidth);

//     kernel_float result = 0.0f;

//     // MISSED UP!
//     // gridX is used twice to access input
//     // slide tiling is redundent in our accomulation loop
//     // I guess... ?
//     // we will come back to it later. alot of things need to be rethought properly.

//     for (int channelTiling = 0; channelTiling < inputChannels; channelTiling += TILE) {
//         for (int slideTiling = 0; slideTiling < slideWidth*slideHeight; slideTiling += TILE) {
//             const int sFlat = slideTiling+threadX;
//             const int sY = sFlat/slideWidth;
//             const int sX = sFlat - sY*slideWidth;

//             tiledWeights[threadZ][threadY][threadX] = weights[
//                 (channelTiling+threadZ)*outputChannels*slideWidth*slideHeight + // target input
//                 gridY*slideWidth*slideHeight + // target output
//                 sY*slideWidth + sX  // target position
//             ];
//             __syncthreads();
//             for (int inputTiling = 0; inputTiling < inputWidth*inputHeight; inputTiling += TILE) {

//             }
//             tiledInput[threadZ][threadY][threadX] = input[
//                 gridZ*inputChannels*inputWidth*inputHeight + // target sample
//                 (channelTiling+threadY)*inputWidth*inputHeight + // target input
//                 ((y*stride - padding)+sY)*inputWidth + ((x*stride - padding)+sX) // target position
//             ];
//             __syncthreads();

//             for (int i = 0; i < TILE; ++i) {
//                 for (int s = 0; s < TILE; ++s) {
//                     result += tiledInput[threadZ][i][s] * tiledWeights[i][threadY][s];
//                 }
//             }
//             __syncthreads();
//         }
//     }

//     if (gridY < samplesAmount && gridY < outputChannels && y < outputHeight && x < outputWidth) {
//         output[gridZ*outputChannels*outputWidth*outputHeight
//             + gridY*outputWidth*outputHeight + y*outputWidth + x] += result;
//     }
// }

__global__ void conv_weights_linear__activationKernel(const kernel_float* __restrict__ input
    , kernel_float* __restrict__ output
    , const kernel_float* __restrict__ weights

    , const int samplesAmount, const int inputChannels, const int outputChannels

    , const int inputWidth, const int outputWidth
    , const int inputHeight, const int outputHeight

    , const int slideWidth, const int slideHeight

    , const int stride, const int padding)
{
    // SHARED ACCROSS BLOCK
    __shared__ kernel_float tiledInput[TILE][TILE][TILE];
    __shared__ kernel_float tiledWeights[TILE][TILE][TILE];
    const int blockFlatX = blockIdx.x*TILE;
    const int blockY = (blockFlatX/outputWidth);
    const int blockX = blockFlatX-(blockY*outputWidth);
    const int blockInputCorner = ((blockY*stride - padding)*
    inputWidth + (blockX*stride - padding));
    const int blockEndFlatX = blockIdx.x*TILE + TILE - 1;
    const int blockEndY = (blockEndFlatX/outputWidth);
    const int blockEndX = blockEndFlatX-(blockEndY*outputWidth);
    const int blockInputEnd = ((blockEndY*stride - padding + slideHeight)*
    inputWidth + (blockEndX*stride - padding + slideWidth));

    const int gridX = blockIdx.x*TILE + threadIdx.x; // VALUE
    const int gridY = blockIdx.y*TILE + threadIdx.y; // CHANNEL
    const int gridZ = blockIdx.z*TILE + threadIdx.z; // SAMPLE

    const int threadX = threadIdx.x;
    const int threadY = threadIdx.y;
    const int threadZ = threadIdx.z;

    // FIGURING OUT LOCATIONS
    const int y = (gridX/outputWidth);
    const int x = gridX-(y*outputWidth);

    const int inputY = y*stride - padding;
    const int inputX = x*stride - padding;
    // const int inputEnd = ((y*stride - padding + slideHeight)*
    // inputWidth + (x*stride - padding + slideWidth));

    kernel_float result = 0.0f;

    // MISSED UP!
    // gridX is used twice to access input
    // slide tiling is redundent in our accomulation loop
    // I guess... ?
    // we will come back to it later. alot of things need to be rethought properly.

    // now we need to mask the input properly to the gridX field AND only extract the input thats within our slideTiling (no repeated accessing to main memory)

    // PERFECT! now all that's left is to add guard locks! <:

    // there is a problem where slide weights and input do not match!

    // it is.. done? I hope at least.

    // tile through input channels
    for (int channelTiling = 0; channelTiling < inputChannels; channelTiling += TILE) {
        // tile through slide weights
        for (int slideTiling = 0; slideTiling < slideWidth*slideHeight; slideTiling += TILE) {
            const int sFlat = slideTiling+threadX;
            const int sY = (sFlat/slideWidth);
            const int sX = sFlat - sY*slideWidth;

            const int sEndFlat = slideTiling+threadX + TILE;
            const int sEndY = sEndFlat/slideWidth;
            const int sEndX = sEndFlat - sEndY*slideWidth;
            
            if ((channelTiling+threadZ) < inputChannels && gridY < outputChannels && sY*slideWidth + sX < slideWidth*slideHeight) {
                tiledWeights[threadZ][threadY][threadX] = weights[
                    (channelTiling+threadZ)*outputChannels*slideWidth*slideHeight + // target input
                    gridY*slideWidth*slideHeight + // target output
                    sY*slideWidth + sX // target position
                ];
                // printf("weight: %f; indexing: %i\n", weights[
                //     (channelTiling+threadZ)*outputChannels*slideWidth*slideHeight + // target input
                //     gridY*slideWidth*slideHeight + // target output
                //     sY*slideWidth + sX // target position
                // ], 
                //     (channelTiling+threadZ)*outputChannels*slideWidth*slideHeight + // target input
                //     gridY*slideWidth*slideHeight + // target output
                //     sY*slideWidth + sX);
            } else {
                tiledWeights[threadZ][threadY][threadX] = 0.0f;
            }
            __syncthreads();

            // tile through block input
            for (int inputTiling = blockInputCorner; inputTiling < blockInputEnd; inputTiling += TILE) {
                const int inputTile = inputTiling+threadY;
                if (gridZ < samplesAmount && (channelTiling+threadX) < inputChannels && inputTile < inputWidth*inputHeight) {
                    // later: mask to only get the coresponding input values for our tiled weights.
                    tiledInput[threadZ][threadY][threadX] = input[
                        gridZ*inputChannels*inputWidth*inputHeight + // target sample
                        (channelTiling+threadX)*inputWidth*inputHeight + // target input
                        inputTile // target position
                    ];
                    // printf("input: %f\n", input[
                    //     gridZ*inputChannels*inputWidth*inputHeight + // target sample
                    //     (channelTiling+threadX)*inputWidth*inputHeight + // target input
                    //     inputTile // target position
                    // ]);
                } else {
                    tiledInput[threadZ][threadY][threadX] = 0.0f;
                }
                __syncthreads();

                // c = current input "c"hannel
                // i = current "i"nput value

                for (int i = 0; i < TILE; ++i) {
                    const int iFlat = inputTiling+i;
                    const int iY = (iFlat/inputWidth);
                    const int iX = iFlat - iY*inputWidth;

                    // selecting ONLY the input for this specific thread
                    if (iY >= inputY && iY <= inputY+slideHeight
                    && iX >= inputX && iX <= inputX+slideWidth) {
                        const int wX = iX-inputX;
                        const int wY = iY-inputY;
                        const int weightIndex = wY*slideWidth + wX - slideTiling;

                        // guarding shared weights indexing
                        if (weightIndex >= 0 && weightIndex < TILE) {
                            for (int c = 0; c < TILE; ++c) {
                                result += tiledInput[threadZ][i][c] * tiledWeights[c][threadY][weightIndex];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    if (gridZ < samplesAmount && gridY < outputChannels && y < outputHeight && x < outputWidth) {
        // printf("old value: %f; index: %i; weights result: %f\n", output[gridZ*outputChannels*outputWidth*outputHeight
        //     + gridY*outputWidth*outputHeight + y*outputWidth + x], gridZ*outputChannels*outputWidth*outputHeight
        //     + gridY*outputWidth*outputHeight + y*outputWidth + x, result);

        output[gridZ*outputChannels*outputWidth*outputHeight
            + gridY*outputWidth*outputHeight + y*outputWidth + x] += result;
    }
}

// __global__ void conv_weights_linear__activationKernel(const kernel_float* __restrict__ input
//     , kernel_float* __restrict__ output
//     , const kernel_float* __restrict__ weights

//     , const int samplesAmount, const int inputChannels, const int outputChannels

//     , const int inputWidth, const int outputWidth
//     , const int inputHeight, const int outputHeight

//     , const int slideWidth, const int slideHeight

//     , const int stride, const int padding)
// {
//     // SHARED ACCROSS BLOCK
//     __shared__ kernel_float tiledInput[TILE][TILE][TILE];
//     __shared__ kernel_float tiledWeights[TILE][TILE][TILE];
//     __shared__ kernel_float inputWeights[TILE][TILE][TILE];
//     const int blockFlatX = blockIdx.x*TILE;
//     const int blockY = (blockFlatX/outputWidth);
//     const int blockX = blockFlatX-(blockY*outputWidth);
//     const int blockInputCorner = ((blockY*stride - padding)*
//     inputWidth + (blockX*stride - padding));
//     const int blockEndFlatX = blockIdx.x*TILE + TILE - 1;
//     const int blockEndY = (blockEndFlatX/outputWidth);
//     const int blockEndX = blockEndFlatX-(blockEndY*outputWidth);
//     const int blockInputEnd = ((blockEndY*stride - padding + slideHeight)*
//     inputWidth + (blockEndX*stride - padding + slideWidth));

//     const int gridX = blockIdx.x*TILE + threadIdx.x; // VALUE
//     const int gridY = blockIdx.y*TILE + threadIdx.y; // CHANNEL
//     const int gridZ = blockIdx.z*TILE + threadIdx.z; // SAMPLE

//     const int threadX = threadIdx.x;
//     const int threadY = threadIdx.y;
//     const int threadZ = threadIdx.z;

//     // FIGURING OUT LOCATIONS
//     const int y = (gridX/outputWidth);
//     const int x = gridX-(y*outputWidth);

//     const int inputY = y*stride - padding;
//     const int inputX = x*stride - padding;
//     // const int inputEnd = ((y*stride - padding + slideHeight)*
//     // inputWidth + (x*stride - padding + slideWidth));

//     kernel_float result = 0.0f;

//     // MISSED UP!
//     // gridX is used twice to access input
//     // slide tiling is redundent in our accomulation loop
//     // I guess... ?
//     // we will come back to it later. alot of things need to be rethought properly.

//     // now we need to mask the input properly to the gridX field AND only extract the input thats within our slideTiling (no repeated accessing to main memory)

//     // PERFECT! now all that's left is to add guard locks! <:

//     // there is a problem where slide weights and input do not match!

//     // it is.. done? I hope at least.

//     // tile through input channels
//     for (int channelTiling = 0; channelTiling < inputChannels; channelTiling += TILE) {
//         // tile through slide weights
//         for (int slideTiling = 0; slideTiling < slideWidth*slideHeight; slideTiling += TILE) {
//             const int sFlat = slideTiling+threadX;
//             const int sY = (sFlat/slideWidth);
//             const int sX = sFlat - sY*slideWidth;

//             const int sEndFlat = slideTiling+threadX + TILE;
//             const int sEndY = sEndFlat/slideWidth;
//             const int sEndX = sEndFlat - sEndY*slideWidth;
            
//             if ((channelTiling+threadZ) < inputChannels && gridY < outputChannels && sY*slideWidth + sX < slideWidth*slideHeight) {
//                 tiledWeights[threadZ][threadY][threadX] = weights[
//                     (channelTiling+threadZ)*outputChannels*slideWidth*slideHeight + // target input
//                     gridY*slideWidth*slideHeight + // target output
//                     sY*slideWidth + sX // target position
//                 ];
//                 // printf("weight: %f; indexing: %i\n", weights[
//                 //     (channelTiling+threadZ)*outputChannels*slideWidth*slideHeight + // target input
//                 //     gridY*slideWidth*slideHeight + // target output
//                 //     sY*slideWidth + sX // target position
//                 // ], 
//                 //     (channelTiling+threadZ)*outputChannels*slideWidth*slideHeight + // target input
//                 //     gridY*slideWidth*slideHeight + // target output
//                 //     sY*slideWidth + sX);
//             } else {
//                 tiledWeights[threadZ][threadY][threadX] = 0.0f;
//             }
//             __syncthreads();

//             // tile through block input
//             for (int inputTiling = blockInputCorner; inputTiling < blockInputEnd; inputTiling += TILE) {
//                 const int inputTile = inputTiling+threadY;
//                 if (gridZ < samplesAmount && (channelTiling+threadX) < inputChannels && inputTile < inputWidth*inputHeight) {
//                     // later: mask to only get the coresponding input values for our tiled weights.
//                     tiledInput[threadZ][threadY][threadX] = input[
//                         gridZ*inputChannels*inputWidth*inputHeight + // target sample
//                         (channelTiling+threadX)*inputWidth*inputHeight + // target input
//                         inputTile // target position
//                     ];
//                     // printf("input: %f\n", input[
//                     //     gridZ*inputChannels*inputWidth*inputHeight + // target sample
//                     //     (channelTiling+threadX)*inputWidth*inputHeight + // target input
//                     //     inputTile // target position
//                     // ]);
//                 } else {
//                     tiledInput[threadZ][threadY][threadX] = 0.0f;
//                 }
//                 __syncthreads();
//                 const int iY = (inputTile/inputWidth);
//                 const int iX = inputTile - iY*inputWidth;
//                 if (iY >= inputY && iY <= inputY+slideHeight
//                 && iX >= inputX && iX <= inputX+slideWidth) {
//                     const int wX = iX-inputX;
//                     const int wY = iY-inputY;
//                     const int weightIndex = wY*slideWidth + wX - slideTiling;

//                     if (weightIndex >= 0 && weightIndex < TILE) {
//                         inputWeights[threadZ][threadY][threadX] = tiledWeights[threadZ][threadY][weightIndex];
//                     } else {
//                         inputWeights[threadZ][threadY][threadX] = 0.0f;
//                     }
//                 } else {
//                     inputWeights[threadZ][threadY][threadX] = 0.0f;
//                 }
//                 __syncthreads();

//                 // c = current input "c"hannel
//                 // i = current "i"nput value

//                 for (int i = 0; i < TILE; ++i) {
//                     for (int c = 0; c < TILE; ++c) {
//                         result += tiledInput[threadZ][i][c] * inputWeights[c][threadY][i];
//                     }
//                 }
//                 __syncthreads();
//             }
//         }
//     }

//     if (gridZ < samplesAmount && gridY < outputChannels && y < outputHeight && x < outputWidth) {
//         // printf("old value: %f; index: %i; weights result: %f\n", output[gridZ*outputChannels*outputWidth*outputHeight
//         //     + gridY*outputWidth*outputHeight + y*outputWidth + x], gridZ*outputChannels*outputWidth*outputHeight
//         //     + gridY*outputWidth*outputHeight + y*outputWidth + x, result);

//         output[gridZ*outputChannels*outputWidth*outputHeight
//             + gridY*outputWidth*outputHeight + y*outputWidth + x] += result;
//     }
// }

extern "C" void conv_weights_linear__activation(const kernel_float* input, kernel_float* output
    , const kernel_float* weights
    , const int& samplesAmount, const int& inputChannels, const int& outputChannels

    , const int& inputWidth, const int& outputWidth
    , const int& inputHeight, const int& outputHeight

    , const int& slideWidth, const int& slideHeight

    , const int& stride, const int& padding)
{
    dim3 threads(TILE, TILE, TILE);
    dim3 blocks(
        (outputWidth*outputHeight + TILE - 1) / TILE,
        (outputChannels + TILE - 1) / TILE,
        (samplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    conv_weights_linear__activationKernel<<<blocks, threads>>>(
        input, output,
        weights,
        samplesAmount, inputChannels, outputChannels,

        inputWidth, outputWidth,
        inputHeight, outputHeight,

        slideWidth, slideHeight,
        stride, padding
    );
}