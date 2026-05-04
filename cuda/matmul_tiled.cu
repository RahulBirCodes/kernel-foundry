#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

// A(m, n) x B(n, k)
#define TILE_WIDTH 16
__global__ void tiledMatMul(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output elem we're working on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // loop over required tiles
    float Cvalue = 0.0f;
    for (int ph = 0; ph < ceil(Width / (float)TILE_WIDTH); ++ph) {
        // collaboartive effort to load values into shared memory from each thread
        if ((Row < m) && (ph * TILE_WIDTH + tx < n)) {
            Ads[ty][tx] = A[Row * n + ph * TILE_WIDTH + tx];
        } else {
            Ads[ty][tx] = 0.0f;
        }

        if ((ph * TILE_WIDTH + ty < n) && (Col < k)) {
            Bds[ty][tx] = B[(ph * TILE_WIDTH + ty) * k + Col];
        } else {
            Bds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += Ads[ty][i] * Bds[i][tx];
        }

        __syncthreads();
    }

    if ((Row < m) && (Col < k)) {
        C[Row * k + Col] = Cvalue;
    }
}

int main() {
    const int Width = 18;
    const int size = Width * Width;

    float* hA = new float[size];
    float* hB = new float[size];
    float* hC = new float[size];

    std::srand(1234);
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            hA[row * Width + col] = static_cast<float>(std::rand() % 10);
            hB[row * Width + col] = (row == col) ? 1.0f : 0.0f;
        }
    }

    float* dA;
    float* dB;
    float* dC;

    cudaMalloc((void**)&dA, size * sizeof(float));
    cudaMalloc((void**)&dB, size * sizeof(float));
    cudaMalloc((void**)&dC, size * sizeof(float));

    cudaMemcpy(dA, hA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Width + TILE_WIDTH - 1) / TILE_WIDTH);

    tiledMatMul<<<dimGrid, dimBlock>>>(dA, dB, dC, Width);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Random matrix A:" << std::endl;
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            std::cout << hA[row * Width + col] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Identity matrix B:" << std::endl;
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            std::cout << hB[row * Width + col] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Matrix C = A x B:" << std::endl;
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            std::cout << hC[row * Width + col] << "\t";
        }
        std::cout << std::endl;
    }

    bool matches = true;
    for (int i = 0; i < size; ++i) {
        if (static_cast<int>(hA[i]) != static_cast<int>(hC[i])) {
            matches = false;
            break;
        }
    }

    std::cout << std::endl;
    if (matches) {
        std::cout << "Check passed: hA and hC match." << std::endl;
    } else {
        std::cout << "Check failed: hA and hC do not match." << std::endl;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}
