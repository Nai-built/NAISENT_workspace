#include "../../optimization.cuh"

__global__ void ADAM_optimization__applyKernel(const int size
    , const kernel_float beta1, const kernel_float beta2, const kernel_float epsilon
    , const int t
    , kernel_float* __restrict__ m
    , kernel_float* __restrict__ v

    , kernel_float* __restrict__ parameters
    , const kernel_float* __restrict__ derivatives

    , const kernel_float multiplier

    , const kernel_float pre_calculated_pow1
    , const kernel_float pre_calculated_pow2)
{
    int x = blockIdx.x*TILE + threadIdx.x; // output position
    
    if (x < size) {
        kernel_float mX = m[x];
        kernel_float vX = v[x];

        const kernel_float derivative = derivatives[x];
        
        mX = fmaf((1 - beta1), derivative, beta1 * mX);
        vX = fmaf((1 - beta2), (derivative * derivative), beta2 * vX);
        // mX = beta1 * mX + (1 - beta1) * derivative;
        // vX = beta2 * vX + (1 - beta2) * (derivative * derivative);

        const kernel_float m_hat = mX * pre_calculated_pow1; /// (1 - pow(beta1, t));
        const kernel_float v_hat = vX * pre_calculated_pow2; /// (1 - pow(beta2, t));

        const kernel_float optimizedDerivative = (multiplier * m_hat / (sqrtf(v_hat) + epsilon));

        m[x] = mX;
        v[x] = vX;

        parameters[x] -= optimizedDerivative;
    }
}

extern "C" void ADAM_optimization__apply(const int& size
    , const kernel_float& beta1, const kernel_float& beta2, const kernel_float& epsilon
    , int& t
    , kernel_float* m
    , kernel_float* v

    , kernel_float* parameters
    , const kernel_float* derivatives

    , const kernel_float& multiplier)
{
    t++;

    dim3 threads(TILE);
    dim3 blocks(
        (size + TILE - 1) / TILE
    );
    // launch kernel
    ADAM_optimization__applyKernel<<<blocks, threads>>>(
        size,
        beta1,
        beta2,
        epsilon,
        t,
        m,
        v,
        parameters,
        derivatives,
        multiplier,

        1/(1 - pow(beta1, t)),
        1/(1 - pow(beta2, t))
    );
}