#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuda_Tensor.h"

cuda_Tensor::cuda_Tensor(cuda_Tensor&& other) noexcept {
    ptr = other.ptr;
    size = other.size;

    other.ptr = nullptr;
    other.size = 0;
}
cuda_Tensor& cuda_Tensor::operator=(cuda_Tensor&& other) noexcept {
    if (this != &other) {
        if (this->ptr) cudaFree(this->ptr);

        this->ptr = other.ptr;
        this->size = other.size;

        other.ptr = nullptr;
        other.size = 0;
    }
    return *this;
}
// cuda_Tensor& cuda_Tensor::operator=(cuda_Tensor& other) noexcept {
//     if (this != &other) {
//         if (this->ptr) cudaFree(this->ptr);

//         this->ptr = other.ptr;
//         this->size = other.size;

//         other.ptr = nullptr;
//         other.size = 0;
//     }
//     return *this;
// }

cuda_Tensor::cuda_Tensor() {
    this->ptr = nullptr;
    this->size = 0;
}
cuda_Tensor::cuda_Tensor(int _size) {
    this->size = _size;

    cudaMalloc(&this->ptr, this->size * sizeof(cuda_neurologicalValue));
    cudaMemset(this->ptr, 0, this->size * sizeof(cuda_neurologicalValue));
}
cuda_Tensor::cuda_Tensor(neurologicalSpan span) {
    this->size = span.size();

    cudaMalloc((void**)&this->ptr, this->size * sizeof(cuda_neurologicalValue));
    cudaMemcpy(this->ptr, span.data(), this->size * sizeof(cuda_neurologicalValue), cudaMemcpyHostToDevice);
}
cuda_Tensor::cuda_Tensor(neurologicalConstantSpan span) {
    this->size = span.size();

    cudaMalloc((void**)&this->ptr, this->size * sizeof(cuda_neurologicalValue));
    cudaMemcpy(this->ptr, span.data(), this->size * sizeof(cuda_neurologicalValue), cudaMemcpyHostToDevice);
}
cuda_Tensor::~cuda_Tensor() {
    if (this->ptr) cudaFree(this->ptr);
}

cuda_neurologicalValue* cuda_Tensor::data() {
    return this->ptr;
}

void cuda_Tensor::resize(const int& newSize) {
    // Example: Resizing d_data from oldSize to newSize
    float* new_ptr;

    cudaMalloc(&new_ptr, newSize * sizeof(cuda_neurologicalValue));
    cudaMemset(new_ptr, 0, newSize * sizeof(cuda_neurologicalValue));

    cudaMemcpy(new_ptr, this->ptr, std::min(this->size, newSize) * sizeof(cuda_neurologicalValue), cudaMemcpyDeviceToDevice);
    cudaFree(this->ptr);

    this->ptr = new_ptr;
    this->size = newSize;
}
void cuda_Tensor::zeroAll() {
    cudaMemset(this->ptr, 0, this->size * sizeof(cuda_neurologicalValue));
}

void cuda_Tensor::copyExactData(const cuda_neurologicalValue* data) {
    cudaMemcpy(this->ptr, data, this->size * sizeof(cuda_neurologicalValue), cudaMemcpyDeviceToDevice);
}

void cuda_Tensor::toCPU(cuda_Tensor& tensor, neurologicalSpan span) {
    cudaMemcpy(span.data(), tensor.data(), tensor.size * sizeof(cuda_neurologicalValue), cudaMemcpyDeviceToHost);
}

// SPANS

cuda_TensorSpan::cuda_TensorSpan() {
    this->ptr = nullptr;
    this->size = 0;
}
cuda_TensorSpan::cuda_TensorSpan(cuda_Tensor& tensor) {
    this->ptr = tensor.data();
    this->size = tensor.size;
}
cuda_neurologicalValue* cuda_TensorSpan::data() {
    return this->ptr;
}

cuda_ConstantTensorSpan::cuda_ConstantTensorSpan() {
    this->ptr = nullptr;
    this->size = 0;
}
cuda_ConstantTensorSpan::cuda_ConstantTensorSpan(cuda_Tensor& tensor) {
    this->ptr = tensor.data();
    this->size = tensor.size;
}
const cuda_neurologicalValue* cuda_ConstantTensorSpan::data() {
    return this->ptr;
}



cuda_SharedIntTensor::cuda_SharedIntTensor(cuda_SharedIntTensor&& other) noexcept {
    ptr = other.ptr;
    size = other.size;

    other.ptr = nullptr;
    other.size = 0;
}
cuda_SharedIntTensor& cuda_SharedIntTensor::operator=(cuda_SharedIntTensor&& other) noexcept {
    if (this != &other) {
        if (this->ptr) cudaFree(this->ptr);

        this->ptr = other.ptr;
        this->size = other.size;

        other.ptr = nullptr;
        other.size = 0;
    }
    return *this;
}

cuda_SharedIntTensor::cuda_SharedIntTensor() {
    this->ptr = nullptr;
    this->size = 0;
}
cuda_SharedIntTensor::cuda_SharedIntTensor(int _size) {
    this->size = _size;

    cudaMallocManaged(&this->ptr, this->size * sizeof(int));
    // cudaMemset(this->ptr, 0, this->size * sizeof(int));
}
cuda_SharedIntTensor::~cuda_SharedIntTensor() {
    if (this->ptr) cudaFree(this->ptr);
}

int* cuda_SharedIntTensor::data() {
    return this->ptr;
}