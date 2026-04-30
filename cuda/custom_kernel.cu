#include <iostream>
#include <cuda_runtime.h>

__global__ void matMul(float* a, float* b, float* c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float t = 0.0f;

        for (int i = 0; i < k; ++i) {
            t += a[row * k + i] * b[i * n + col];
        }

        c[row * n + col] = t;
    }
}

int main() {
    const int m = 2;
    const int k = 3;
    const int n = 4;

    const int aSize = m * k;
    const int bSize = k * n;
    const int cSize = m * n;

    float hA[aSize] = {
        1, 2, 3,
        4, 5, 6
    };

    float hB[bSize] = {
        7, 8, 9, 10,
        11, 12, 13, 14,
        15, 16, 17, 18
    };

    float hC[cSize];

    float *dA, *dB, *dC;

    cudaMalloc((void**)&dA, aSize * sizeof(float));
    cudaMalloc((void**)&dB, bSize * sizeof(float));
    cudaMalloc((void**)&dC, cSize * sizeof(float));

    cudaMemcpy(dA, hA, aSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(2, 2);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (m + blockSize.y - 1) / blockSize.y);

    matMul<<<gridSize, blockSize>>>(dA, dB, dC, m, k, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, cSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Matrix A (" << m << "x" << k << "):" << std::endl;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            std::cout << hA[row * k + col] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Matrix B (" << k << "x" << n << "):" << std::endl;
    for (int row = 0; row < k; row++) {
        for (int col = 0; col < n; col++) {
            std::cout << hB[row * n + col] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Matrix C = A x B (" << m << "x" << n << "):" << std::endl;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            std::cout << hC[row * n + col] << "\t";
        }
        std::cout << std::endl;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
