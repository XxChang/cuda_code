#include <stdio.h>
#include <cuda.h>
#include "helper_function.h"

const size_t COLS = 32;
const size_t ROWS = 2048;

__global__ void softmax_warp_reduction_float4(const float *cpu_input, float *output, const int rows, const int cols)
{
    int row0 = blockIdx.x * 4 * blockDim.y + 0 + threadIdx.y;
    int row1 = blockIdx.x * 4 * blockDim.y + 1 + threadIdx.y;
    int row2 = blockIdx.x * 4 * blockDim.y + 2 + threadIdx.y;
    int row3 = blockIdx.x * 4 * blockDim.y + 3 + threadIdx.y;

    __shared__ float4 input[32 * 32];
    __shared__ float4 globalMax[32];
    __shared__ float4 globalSum[32];

    // copy to shared memory
    input[threadIdx.x + threadIdx.y * blockDim.x].x = cpu_input[row0 * cols + threadIdx.x];
    input[threadIdx.x + threadIdx.y * blockDim.x].y = cpu_input[row1 * cols + threadIdx.x];
    input[threadIdx.x + threadIdx.y * blockDim.x].z = cpu_input[row2 * cols + threadIdx.x];
    input[threadIdx.x + threadIdx.y * blockDim.x].w = cpu_input[row3 * cols + threadIdx.x];


    float max_val_x = max(-__FLT_MAX__, input[threadIdx.x + threadIdx.y * blockDim.x].x);
    max_val_x = max(max_val_x, __shfl_xor_sync(0xffffffff, max_val_x, 16));
    max_val_x = max(max_val_x, __shfl_xor_sync(0xffffffff, max_val_x, 8));
    max_val_x = max(max_val_x, __shfl_xor_sync(0xffffffff, max_val_x, 4));
    max_val_x = max(max_val_x, __shfl_xor_sync(0xffffffff, max_val_x, 2));
    max_val_x = max(max_val_x, __shfl_xor_sync(0xffffffff, max_val_x, 1));

    float max_val_y = max(-__FLT_MAX__, input[threadIdx.x + threadIdx.y * blockDim.x].y);
    max_val_y = max(max_val_y, __shfl_xor_sync(0xffffffff, max_val_y, 16));
    max_val_y = max(max_val_y, __shfl_xor_sync(0xffffffff, max_val_y, 8));
    max_val_y = max(max_val_y, __shfl_xor_sync(0xffffffff, max_val_y, 4));
    max_val_y = max(max_val_y, __shfl_xor_sync(0xffffffff, max_val_y, 2));
    max_val_y = max(max_val_y, __shfl_xor_sync(0xffffffff, max_val_y, 1));

    float max_val_z = max(-__FLT_MAX__, input[threadIdx.x + threadIdx.y * blockDim.x].z);
    max_val_z = max(max_val_z, __shfl_xor_sync(0xffffffff, max_val_z, 16));
    max_val_z = max(max_val_z, __shfl_xor_sync(0xffffffff, max_val_z, 8));
    max_val_z = max(max_val_z, __shfl_xor_sync(0xffffffff, max_val_z, 4));
    max_val_z = max(max_val_z, __shfl_xor_sync(0xffffffff, max_val_z, 2));
    max_val_z = max(max_val_z, __shfl_xor_sync(0xffffffff, max_val_z, 1));

    float max_val_w = max(-__FLT_MAX__, input[threadIdx.x + threadIdx.y * blockDim.x].w);
    max_val_w = max(max_val_w, __shfl_xor_sync(0xffffffff, max_val_w, 16));
    max_val_w = max(max_val_w, __shfl_xor_sync(0xffffffff, max_val_w, 8));
    max_val_w = max(max_val_w, __shfl_xor_sync(0xffffffff, max_val_w, 4));
    max_val_w = max(max_val_w, __shfl_xor_sync(0xffffffff, max_val_w, 2));
    max_val_w = max(max_val_w, __shfl_xor_sync(0xffffffff, max_val_w, 1));
    if (threadIdx.x == 0)
    {
        globalMax[threadIdx.y].x = max_val_x;
        globalMax[threadIdx.y].y = max_val_y;
        globalMax[threadIdx.y].z = max_val_z;
        globalMax[threadIdx.y].w = max_val_w;
    }
    __syncthreads();
    // globalMax[threadIdx.y] = max(max_val, __shfl_xor_sync(0x00000001, max_val, 1));
    
    float sum_x = expf(input[threadIdx.x + threadIdx.y * blockDim.x].x - globalMax[threadIdx.y].x);
    sum_x += __shfl_xor_sync(0xffffffff, sum_x, 16);
    sum_x += __shfl_xor_sync(0xffffffff, sum_x, 8);
    sum_x += __shfl_xor_sync(0xffffffff, sum_x, 4);
    sum_x += __shfl_xor_sync(0xffffffff, sum_x, 2);
    sum_x += __shfl_xor_sync(0xffffffff, sum_x, 1);

    float sum_y = expf(input[threadIdx.x + threadIdx.y * blockDim.x].y - globalMax[threadIdx.y].y);
    sum_y += __shfl_xor_sync(0xffffffff, sum_y, 16);
    sum_y += __shfl_xor_sync(0xffffffff, sum_y, 8);
    sum_y += __shfl_xor_sync(0xffffffff, sum_y, 4);
    sum_y += __shfl_xor_sync(0xffffffff, sum_y, 2);
    sum_y += __shfl_xor_sync(0xffffffff, sum_y, 1);

    float sum_z = expf(input[threadIdx.x + threadIdx.y * blockDim.x].z - globalMax[threadIdx.y].z);
    sum_z += __shfl_xor_sync(0xffffffff, sum_z, 16);
    sum_z += __shfl_xor_sync(0xffffffff, sum_z, 8);
    sum_z += __shfl_xor_sync(0xffffffff, sum_z, 4);
    sum_z += __shfl_xor_sync(0xffffffff, sum_z, 2);
    sum_z += __shfl_xor_sync(0xffffffff, sum_z, 1);

    float sum_w = expf(input[threadIdx.x + threadIdx.y * blockDim.x].w - globalMax[threadIdx.y].w);
    sum_w += __shfl_xor_sync(0xffffffff, sum_w, 16);
    sum_w += __shfl_xor_sync(0xffffffff, sum_w, 8);
    sum_w += __shfl_xor_sync(0xffffffff, sum_w, 4);
    sum_w += __shfl_xor_sync(0xffffffff, sum_w, 2);
    sum_w += __shfl_xor_sync(0xffffffff, sum_w, 1);

    if (threadIdx.x == 0)
    {
        globalSum[threadIdx.y].x = sum_x;
        globalSum[threadIdx.y].y = sum_y;
        globalSum[threadIdx.y].z = sum_z;
        globalSum[threadIdx.y].w = sum_w;
    }
    __syncthreads();
    // globalSum[threadIdx.y] = sum + __shfl_xor_sync(0x1, sum, 1);
    
    output[row0 * cols + threadIdx.x] = expf(input[threadIdx.x + threadIdx.y * blockDim.x].x - globalMax[threadIdx.y].x) * __fdividef(1.0F, globalSum[threadIdx.y].x);
    output[row1 * cols + threadIdx.x] = expf(input[threadIdx.x + threadIdx.y * blockDim.x].y - globalMax[threadIdx.y].y) * __fdividef(1.0F, globalSum[threadIdx.y].y);
    output[row2 * cols + threadIdx.x] = expf(input[threadIdx.x + threadIdx.y * blockDim.x].z - globalMax[threadIdx.y].z) * __fdividef(1.0F, globalSum[threadIdx.y].z);
    output[row3 * cols + threadIdx.x] = expf(input[threadIdx.x + threadIdx.y * blockDim.x].w - globalMax[threadIdx.y].w) * __fdividef(1.0F, globalSum[threadIdx.y].w);
}

void softmax_gpu_warp_reduction(const float *cpu_input, float *cpu_output, const int rows, const int cols, const int times)
{
    const size_t rows_per_block = 32;
    const size_t cols_per_block = cols;
    
    dim3 grid_dim(rows/(rows_per_block * 4), 1, 1);
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
        softmax_warp_reduction_float4<<<grid_dim, block_dim>>>(input, output, rows, cols);
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


