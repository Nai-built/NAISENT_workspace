#pragma once

#include <vector>
#include <span>

typedef float cuda_neurologicalValue;

typedef std::span<cuda_neurologicalValue> neurologicalSpan;
typedef std::span<const cuda_neurologicalValue> neurologicalConstantSpan;

typedef std::span<const int> lengths;

struct cuda_Tensor {
    // cuda_Tensor(const cuda_Tensor&) = delete;
    // cuda_Tensor& operator=(cuda_Tensor&);

    cuda_Tensor(cuda_Tensor&&) noexcept;
    cuda_Tensor& operator=(cuda_Tensor&&) noexcept;

    private:
    cuda_neurologicalValue* ptr;
    public:
    int size;

    cuda_Tensor();
    cuda_Tensor(int _size);
    cuda_Tensor(neurologicalSpan span);
    cuda_Tensor(neurologicalConstantSpan span);
    ~cuda_Tensor();

    cuda_neurologicalValue* data();
    
    void resize(const int& newSize);
    void zeroAll();

    void copyExactData(const cuda_neurologicalValue* data);

    static void toCPU(cuda_Tensor& tensor, neurologicalSpan span);
};
struct cuda_SharedIntTensor {
    cuda_SharedIntTensor(cuda_SharedIntTensor&&) noexcept;
    cuda_SharedIntTensor& operator=(cuda_SharedIntTensor&&) noexcept;

    private:
    int* ptr;
    public:
    int size;

    cuda_SharedIntTensor();
    cuda_SharedIntTensor(int _size);
    ~cuda_SharedIntTensor();

    int* data();
};

struct cuda_TensorSpan {
    private:
    cuda_neurologicalValue* ptr;
    public:
    int size;
    
    cuda_TensorSpan();
    cuda_TensorSpan(cuda_Tensor& tensor);
    
    cuda_neurologicalValue* data();
};
struct cuda_ConstantTensorSpan {
    private:
    const cuda_neurologicalValue* ptr;
    public:
    int size;
    
    cuda_ConstantTensorSpan();
    cuda_ConstantTensorSpan(cuda_Tensor& tensor);
    
    const cuda_neurologicalValue* data();
};