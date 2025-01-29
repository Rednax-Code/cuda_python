#include <iostream>
#include <string>

__global__ void addArraysKernel(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the global thread index
    if (idx < n) { // Ensure we don't access out-of-bounds memory
        c[idx] = a[idx];
    }
}

struct cIntArray {
    int* data;
    int length;
};

struct cFloatArray {
    double* data;
    double length;
};

extern "C" __declspec(dllexport) void freeArray(cIntArray* ptr) {
    if (ptr) {
        delete[] ptr->data;  // Free the dynamically allocated array
        delete ptr;          // Free the struct itself
    }
}

extern "C" __declspec(dllexport) const char* addArraysSignature() {
    return "((list[int], list[int]), (list[int]))";
}

// Host function to add arrays
extern "C" __declspec(dllexport) cIntArray* addArrays(cIntArray a, cIntArray b) {
	int *d_a, *d_b, *d_c;
    int n = a.length;

    if (b.length != n) {
        return nullptr;
    }

    // Allocate memory for the result
    cIntArray *result = new cIntArray;
    result->data = new int[n];
    result->length = n;

    // Allocate device memory and perform computations...
    cudaMalloc((void **)&d_a, n * sizeof(int));
    cudaMalloc((void **)&d_b, n * sizeof(int));
    cudaMalloc((void **)&d_c, n * sizeof(int));
    
    cudaMemcpy(d_a, a.data, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	addArraysKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaMemcpy(result->data, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device pointers
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}