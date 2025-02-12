#include <stdio.h>
#include <cuda.h>
#include "helper_function.h"

const size_t COLS = 32;
const size_t ROWS = 2048;

__global__ void softmax_warp_reduction(const float *cpu_input, float *output, const int rows, const int cols)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;

    // 避免bank conflict
    __shared__ float input[33*8];
    __shared__ float globalMax[8];
    // __shared__ float globalSum[8];
    __shared__ float globalSum[8];

    // copy to shared memory
    input[threadIdx.x + threadIdx.y * blockDim.x] = cpu_input[row * cols + threadIdx.x];

    float max_val = max(-__FLT_MAX__, input[threadIdx.x + threadIdx.y * blockDim.x]);
    max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 16));
    max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 8));
    max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 4));
    max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 2));
    max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 1));
    if (threadIdx.x == 0)
    {
        globalMax[threadIdx.y] = max_val;
    }
    __syncthreads();
    // globalMax[threadIdx.y] = max(max_val, __shfl_xor_sync(0x00000001, max_val, 1));
    
    float sum = expf(input[threadIdx.x + threadIdx.y * blockDim.x] - globalMax[threadIdx.y]);
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    if (threadIdx.x == 0)
    {
        globalSum[threadIdx.y] = sum;
    }
    __syncthreads();
    // globalSum[threadIdx.y] = sum + __shfl_xor_sync(0x1, sum, 1);
    
    output[row * cols + threadIdx.x] = expf(input[threadIdx.x + threadIdx.y * blockDim.x] - globalMax[threadIdx.y]) * __fdividef(1.0F, globalSum[threadIdx.y]);
}

void softmax_gpu_warp_reduction(const float *cpu_input, float *cpu_output, const int rows, const int cols, const int times)
{
    const size_t rows_per_block = 8;
    const size_t cols_per_block = cols;
    
    dim3 grid_dim(rows/rows_per_block, 1, 1);
    dim3 block_dim(cols_per_block, rows_per_block, 1);

    float *input, *output;
    cudaMalloc((void **)&input, rows * cols * sizeof(float));
    cudaMalloc((void **)&output, rows * cols * sizeof(float));
    cudaMemcpy(input, cpu_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float ker_time = 0;
    int repeat = times;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        softmax_warp_reduction<<<grid_dim, block_dim>>>(input, output, rows, cols);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);

    cudaMemcpy(cpu_output, output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(input);
    cudaFree(output);

    printf("kernel time:%.4fms\n", ker_time/repeat);
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
    softmax_gpu_warp_reduction(cpu_input, gpu_output, ROWS, COLS, 1);
    for (int i = 0; i < 10; i++)
    {
        printf("%.4e ", gpu_output[i]);
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


