#include <stdio.h>
#include <cuda.h>
#include "helper_function.h"

const size_t COLS = 32;
const size_t ROWS = 2048;

void softmax_gpu_warp_reduction_text(const float *cpu_input, float *cpu_output, const int rows, const int cols, const int times)
{
    const size_t rows_per_block = 8;
    const size_t cols_per_block = cols;
    
    dim3 grid_dim(rows/rows_per_block, 1, 1);
    dim3 block_dim(cols_per_block, rows_per_block, 1);

    float *output;
    cudaMalloc((void **)&output, rows * cols * sizeof(float));

    cudaArray *cu_array;
    cudaChannelFormatKind kind = cudaChannelFormatKindFloat;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, kind);

    cudaError_t returnValue;
    returnValue = cudaMallocArray(&cu_array, &channelDesc, cols, rows);
    returnValue = (cudaError_t)(returnValue | cudaMemcpy(cu_array, cpu_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    if(returnValue != cudaSuccess) {
        printf("\n Got error while running CUDA API Array Copy %d\n", returnValue);
    }

    cudaFreeArray(cu_array);
    cudaFree(output);
}

int main()
{
    float *cpu_input, *gpu_output, *cpu_output;
    cpu_input = (float *)malloc(ROWS * COLS * sizeof(float));
    cpu_output = (float *)malloc(ROWS * COLS * sizeof(float));
    gpu_output = (float *)malloc(ROWS * COLS * sizeof(float));
    for (int i = 0; i < COLS * ROWS; i++)
    {
        cpu_input[i] = i % 10;
    }
    softmax_gpu_warp_reduction_text(cpu_input, cpu_output, ROWS, COLS, 1);
    for (int i = 0; i < 10; i++)
    {
        printf("%.4e ", cpu_output[i]);
    }
    printf("\n");

    softmax_cpu(cpu_input, cpu_output, ROWS, COLS);
    for (int i = 0; i < 10; i++)
    {
        printf("%.4e ", cpu_output[i]);
    }
    printf("\n");

    float max_error = -__FLT_MAX__;
    for (int i = 0; i < COLS * ROWS; i++)
    {
        max_error = max(max_error, abs(gpu_output[i] - cpu_output[i]));
    }
    printf("max_error: %f\n", max_error);
    free(cpu_input);
    free(cpu_output);
    free(gpu_output);
    return 0;
}