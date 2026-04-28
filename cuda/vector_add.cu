#include <iostream>
#include <cuda_runtime.h>

__global__ void vecAdd(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int n = 2000;

    float h_A[n];
    float h_B[n];
    float h_C[n];

    for (int i = 0; i < n; i++) {
        h_A[i] = 2 * i + 1;
        h_B[i] = 2 * i + 2;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, n * sizeof(float));
    cudaMalloc((void**)&d_B, n * sizeof(float));
    cudaMalloc((void**)&d_C, n * sizeof(float));

    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int threadsPerBlock = prop.maxThreadsPerBlock;
    if (threadsPerBlock > n) {
        threadsPerBlock = n;
    }
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Launching " << blocks << " blocks of " << threadsPerBlock << " threads" << std::endl;

    vecAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
