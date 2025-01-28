#include <cuda_runtime.h>
#include <iostream>

// Kernel for adding two arrays
__global__ void addArraysKernel(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the global thread index
    if (idx < n) { // Ensure we don't access out-of-bounds memory
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" __declspec(dllexport) void freeMemory(int* ptr) {
    delete[] ptr;
}

// Host function to add arrays
extern "C" __declspec(dllexport) int* addArrays(int *a, int *b, int n) {
    int *d_a, *d_b, *d_c;  // Device pointers
    int *c = new int[n];   // Allocate memory on the host for the result

    // Allocate memory on the GPU for the arrays
    cudaMalloc((void **)&d_a, n * sizeof(int));
    cudaMalloc((void **)&d_b, n * sizeof(int));
    cudaMalloc((void **)&d_c, n * sizeof(int));

    // Copy the input arrays from host to device
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // Determine grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    addArraysKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy the result array back to the host
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the allocated device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}