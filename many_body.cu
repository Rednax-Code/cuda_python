#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Kernel
__global__ void updateKernel(double *xPos, double *yPos, double *xVelo, double *yVelo, double *m, double *xPos2, double *yPos2, double *xVelo2, double *yVelo2, int n, double dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the global thread index
    if (idx < n) { // Ensure we don't access out-of-bounds memory
        double xAccel = 0.0;
        double yAccel = 0.0;
        for (int i = 0; i < n; i++) {
            double xDist = xPos[i] - xPos[idx];
            double yDist = yPos[i] - yPos[idx];
            double distSqr = pow(xDist, 2.0) + pow(yDist, 2.0);
            xAccel = xAccel + (m[i] * xDist) / (distSqr + 0.5);
            yAccel = yAccel + (m[i] * yDist) / (distSqr + 0.5);
        }
        xVelo2[idx] = xVelo[idx] + xAccel * 6.6743e-11 * dt;
        yVelo2[idx] = yVelo[idx] + yAccel * 6.6743e-11 * dt;
        xPos2[idx] = xPos[idx] + xVelo2[idx] * dt;
        yPos2[idx] = yPos[idx] + yVelo2[idx] * dt;
    }
}

struct cParticleArray {
    double *xPos, *yPos, *xVelo, *yVelo, *mass;
    int n;
};

// Define 
int n;
double *dXPos, *dYPos, *dXVelo, *dYVelo, *dM, *dXPos2, *dYPos2, *dXVelo2, *dYVelo2;

extern "C" __declspec(dllexport) void freeArray(cParticleArray* ptr) {
    if (ptr) {
        // Free the dynamically allocated arrays
        delete[] ptr->xPos;
        delete[] ptr->yPos;
        delete[] ptr->xVelo;
        delete[] ptr->yVelo;
        // Free the struct itself
        delete ptr;
    }
}

extern "C" __declspec(dllexport) void prepare(double* xPos, double* yPos, double* xVelo, double* yVelo, double* mass, int counts) {

    n = counts;

    // Allocate device memory and perform computations...
    cudaMalloc((void **)&dXPos, n * sizeof(double));
    cudaMalloc((void **)&dYPos, n * sizeof(double));
    cudaMalloc((void **)&dXVelo, n * sizeof(double));
    cudaMalloc((void **)&dYVelo, n * sizeof(double));
    cudaMalloc((void **)&dM, n * sizeof(double));
    cudaMalloc((void **)&dXPos2, n * sizeof(double));
    cudaMalloc((void **)&dYPos2, n * sizeof(double));
    cudaMalloc((void **)&dXVelo2, n * sizeof(double));
    cudaMalloc((void **)&dYVelo2, n * sizeof(double));

    cudaMemcpyAsync(dXPos, xPos, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dYPos, yPos, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dXVelo, xVelo, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dYVelo, yVelo, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dM, mass, n * sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    updateKernel<<<blocksPerGrid, threadsPerBlock>>>(dXPos, dYPos, dXVelo, dYVelo, dM, dXPos2, dYPos2, dXVelo2, dYVelo2, n, 0);
    cudaDeviceSynchronize();
}

extern "C" __declspec(dllexport) cParticleArray* update(double dt) {

    cParticleArray *particles = new cParticleArray;
    particles->xPos = new double[n];
    particles->yPos = new double[n];
    particles->xVelo = new double[n];
    particles->yVelo = new double[n];

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    updateKernel<<<blocksPerGrid, threadsPerBlock>>>(dXPos, dYPos, dXVelo, dYVelo, dM, dXPos2, dYPos2, dXVelo2, dYVelo2, n, dt);
    cudaDeviceSynchronize();

    // Copy results internally in GPU
    cudaMemcpyAsync(dXPos, dXPos2, n * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(dYPos, dYPos2, n * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(dXVelo, dXVelo2, n * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(dYVelo, dYVelo2, n * sizeof(double), cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(particles->xPos, dXPos, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles->yPos, dYPos, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles->xVelo, dXVelo, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles->yVelo, dYVelo, n * sizeof(double), cudaMemcpyDeviceToHost);

    return particles;
}

extern "C" __declspec(dllexport) int cleanUp() {
    cudaFree(dXPos);
    cudaFree(dYPos);
    cudaFree(dXVelo);
    cudaFree(dYVelo);
    cudaFree(dM);
    cudaFree(dXPos2);
    cudaFree(dYPos2);
    cudaFree(dXVelo2);
    cudaFree(dYVelo2);
    return 1;
}