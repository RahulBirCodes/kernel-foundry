#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <assert.h>

void writeBinaryFile(const char* path, const float* data, int count) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), count * sizeof(float));
}

std::string shellQuote(const std::string& text) {
    std::string quoted = "'";
    for (char c : text) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

void verifyAgainstPyTorch(const float* input, const float* filter, const float* output,
                          int inChannels, int outChannels,
                          int height, int width, int kernelSize) {
    const char* inputPath = "conv_input.bin";
    const char* filterPath = "conv_filter.bin";
    const char* outputPath = "conv_output.bin";

    writeBinaryFile(inputPath, input, inChannels * height * width);
    writeBinaryFile(filterPath, filter, outChannels * inChannels * kernelSize * kernelSize);
    writeBinaryFile(outputPath, output, outChannels * height * width);

    std::string pythonCode = R"PY(
import sys
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch is not installed.")
    sys.exit(1)

if len(sys.argv) != 9:
    print(
        "Usage: python3 -c <code> "
        "<input.bin> <filter.bin> <output.bin> "
        "<in_channels> <out_channels> <height> <width> <kernel_size>"
    )
    sys.exit(1)

input_path = sys.argv[1]
filter_path = sys.argv[2]
output_path = sys.argv[3]
in_channels = int(sys.argv[4])
out_channels = int(sys.argv[5])
height = int(sys.argv[6])
width = int(sys.argv[7])
kernel_size = int(sys.argv[8])

input_data = np.fromfile(input_path, dtype=np.float32).reshape(1, in_channels, height, width)
filter_data = np.fromfile(filter_path, dtype=np.float32).reshape(
    out_channels, in_channels, kernel_size, kernel_size
)
cuda_output = np.fromfile(output_path, dtype=np.float32).reshape(out_channels, height, width)

input_tensor = torch.from_numpy(input_data)
filter_tensor = torch.from_numpy(filter_data)
pytorch_output = F.conv2d(input_tensor, filter_tensor, padding=kernel_size // 2)
pytorch_output = pytorch_output.squeeze(0).numpy()

if np.allclose(cuda_output, pytorch_output, atol=1e-5):
    print("Outputs match PyTorch.")
    sys.exit(0)

max_diff = np.max(np.abs(cuda_output - pytorch_output))
print("Outputs do not match PyTorch.")
print("Max absolute difference:", max_diff)
sys.exit(1)
)PY";

    std::string command = "python3 -c " +
        shellQuote(pythonCode) +
        " " + shellQuote(inputPath) +
        " " + shellQuote(filterPath) +
        " " + shellQuote(outputPath) +
        " " + shellQuote(std::to_string(inChannels)) +
        " " + shellQuote(std::to_string(outChannels)) +
        " " + shellQuote(std::to_string(height)) +
        " " + shellQuote(std::to_string(width)) +
        " " + shellQuote(std::to_string(kernelSize));

    int status = std::system(command.c_str());
    if (status == 0) {
        std::cout << "PyTorch verification passed." << std::endl;
    } else {
        std::cout << "PyTorch verification failed." << std::endl;
    }
}

__global__ void zeroPaddedConv2D(float* input, float* filter, float* output,
                                 int inChannels, int outChannels,
                                 int height, int width, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int co = blockIdx.z * blockDim.z + threadIdx.z;

    if (co < outChannels && row < height && col < width) {
        int radius = kernelSize / 2;
        float sum = 0.0f;

        for (int ci = 0; ci < inChannels; ci++) {
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    int inRow = row + i;
                    int inCol = col + j;

                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        int inputIndex = (ci * height + inRow) * width + inCol;
                        int filterRow = i + radius;
                        int filterCol = j + radius;
                        int filterIndex = ((co * inChannels + ci) * kernelSize + filterRow) * kernelSize + filterCol;

                        sum += input[inputIndex] * filter[filterIndex];
                    }
                }
            }
        }

        int outputIndex = (co * height + row) * width + col;
        output[outputIndex] = sum;
    }
}

int main() {
    const int inChannels = 64;
    const int outChannels = 128;
    const int height = 1024;
    const int width = 1024;
    const int kernelSize = 3;

    assert(kernelSize % 2 != 0 && "kernelSize must be odd for zero padding conv");

    const int inputSize = inChannels * height * width;
    const int filterSize = outChannels * inChannels * kernelSize * kernelSize;
    const int outputSize = outChannels * height * width;

    float *h_input  = new float[inputSize];
    float *h_filter = new float[filterSize];
    float *h_output = new float[outputSize];

    for (int ci = 0; ci < inChannels; ci++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int inputIndex = (ci * height + row) * width + col;
                h_input[inputIndex] = (ci + 1) * (row * width + col + 1);
            }
        }
    }

    float averageWeight = 1.0f / (kernelSize * kernelSize);
    for (int i = 0; i < filterSize; i++) {
        h_filter[i] = averageWeight;
    }

    float *d_input, *d_filter, *d_output;

    cudaMalloc((void**)&d_input, inputSize * sizeof(float));
    cudaMalloc((void**)&d_filter, filterSize * sizeof(float));
    cudaMalloc((void**)&d_output, outputSize * sizeof(float));

    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y,
                  (outChannels + blockSize.z - 1) / blockSize.z);

    zeroPaddedConv2D<<<gridSize, blockSize>>>(d_input, d_filter, d_output,
                                              inChannels, outChannels,
                                              height, width, kernelSize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Input tensor [channel][row][col]:" << std::endl;
    for (int ci = 0; ci < inChannels; ci++) {
        std::cout << "Input channel " << ci << ":" << std::endl;
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int inputIndex = (ci * height + row) * width + col;
                std::cout << h_input[inputIndex] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Output tensor [out_channel][row][col]:" << std::endl;
    for (int co = 0; co < outChannels; co++) {
        std::cout << "Output channel " << co << ":" << std::endl;
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int outputIndex = (co * height + row) * width + col;
                std::cout << h_output[outputIndex] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    verifyAgainstPyTorch(h_input, h_filter, h_output,
                         inChannels, outChannels,
                         height, width, kernelSize);

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_filter;
    delete[] h_output;

    return 0;
}
