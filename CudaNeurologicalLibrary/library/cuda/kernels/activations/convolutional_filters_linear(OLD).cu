#include "../../activations.cuh"

// this a super naive convolutional activation kernel that computes both weights and biases
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

__global__ void convolutional_filters_linear__activationKernel(const kernel_float* __restrict__ input
    , kernel_float* __restrict__ output
    , const kernel_float* __restrict__ weights
    , const kernel_float* __restrict__ biases

    , const int samplesAmount, const int inputChannels, const int outputChannels

    , const int inputWidth, const int outputWidth
    , const int inputHeight, const int outputHeight

    , const int slideWidth, const int slideHeight

    , const int stride, const int padding)
{
    const int gridX = blockIdx.x*TILE + threadIdx.x;
    const int gridY = blockIdx.y*TILE + threadIdx.y;

    // FIGURING OUT LOCATIONS
    const int channel = gridX/(outputWidth*outputHeight);
    const int channelFlatPoint = gridX-(channel*outputWidth*outputHeight);

    const int y = channelFlatPoint/outputWidth;
    const int x = channelFlatPoint-(y*outputWidth);

    kernel_float result = 0.0f;

    for (int inputTiling = 0; inputTiling < inputChannels; inputTiling += TILE) {
        for (int slideTiling = 0; slideTiling < slideWidth*slideHeight; slideTiling += TILE) {
            
        }
    }

    if (gridY < samplesAmount && channel < outputChannels && y < outputHeight && x < outputWidth) {
        // NAIVE ACCOMULATION LOOP
        for (int i = 0; i < inputChannels; ++i) {
            for (int sY = 0; sY < slideHeight; ++sY) {
                int iY = (y*stride - padding)+sY;
                if (iY >= inputHeight || iY < 0) continue;
                for (int sX = 0; sX < slideWidth; ++sX) {
                    int iX = (x*stride - padding)+sX;
                    if (iX >= inputWidth || iX < 0) continue;

                    result +=
                        weights[i*outputChannels*slideWidth*slideHeight +
                            channel*slideWidth*slideHeight + sY*slideWidth + sX]
                        * input[gridY*inputChannels*inputWidth*inputHeight +
                            i*inputWidth*inputHeight + iY*inputWidth + iX];
                }
            }
        }

        output[gridY*outputChannels*outputWidth*outputHeight
            + channel*outputWidth*outputHeight + y*outputWidth + x] += result;
    }
}

extern "C" void convolutional_filters_linear__activation(const kernel_float* input, kernel_float* output
    , const kernel_float* weights, const kernel_float* biases
    , const int& samplesAmount, const int& inputChannels, const int& outputChannels

    , const int& inputWidth, const int& outputWidth
    , const int& inputHeight, const int& outputHeight

    , const int& slideWidth, const int& slideHeight

    , const int& stride, const int& padding)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(
        (outputChannels*outputWidth*outputHeight + TILE - 1) / TILE,
        (samplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    convolutional_filters_linear__activationKernel<<<blocks, threads>>>(
        input, output,
        weights, biases,
        samplesAmount, inputChannels, outputChannels,

        slideWidth, slideHeight,

        inputWidth, outputWidth,
        inputHeight, outputHeight,
        stride, padding
    );
}