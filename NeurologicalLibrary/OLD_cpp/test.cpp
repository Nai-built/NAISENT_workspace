#include <iostream>
#include <math.h>

void dense_forward_batch_blocked(
    const float* X,
    const float* W,
    const float* bias,
    float* Y,
    int B, int I, int O)
{
    constexpr int BB = 32; // batch block
    constexpr int BO = 64; // output block
    constexpr int BI = 64; // input block

    for (int b0 = 0; b0 < B; b0 += BB)
        for (int o0 = 0; o0 < O; o0 += BO)
            for (int i0 = 0; i0 < I; i0 += BI)
            {
                int bmax = std::min(b0 + BB, B);
                int omax = std::min(o0 + BO, O);
                int imax = std::min(i0 + BI, I);

                for (int b = b0; b < bmax; ++b)
                {
                    float* y = Y + b * O + o0;
                    const float* x = X + b * I;

                    if (i0 == 0)
                        for (int o = 0; o < omax - o0; ++o)
                            y[o] = bias[o0 + o];

                    for (int i = i0; i < imax; ++i)
                    {
                        float xv = x[i];
                        const float* w = W + i * O + o0;

                        for (int o = 0; o < omax - o0; ++o)
                            y[o] += xv * w[o];
                    }
                }
            }
}

