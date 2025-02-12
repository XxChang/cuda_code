#include "helper_function.h"
#include <stdio.h>
#include <math.h>
#include <stdint.h>

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

void softmax_cpu(const float *input, float *output, int rows, int cols)
{
    float m[rows];
    for (int row = 0; row < rows; row++) {
        float max_val = -__FLT_MAX__;
        for (int col = 0; col < cols; col++) {
            max_val = (input[row * cols + col] > max_val) ? input[row * cols + col] : max_val;
        }
        m[row] = max_val;
    }

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            output[row * cols + col] = expf(input[row * cols + col] - m[row]);
        }
    }

    for (int row = 0; row < rows; row++) {
        float sum = 0;
        for (int col = 0; col < cols; col++) {
            sum += output[row * cols + col];
        }
        for (int col = 0; col < cols; col++) {
            output[row * cols + col] /= sum;
        }
    }
}

__global__ void softmax(const float *input, float *output, const int rows, const int cols)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;

    __shared__ float tmp[32];
    __shared__ float globalMax;
    __shared__ float globalSum;

    float val = -__FLT_MAX__;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        val = max(val, input[row * cols + i]);
    }
    tmp[threadIdx.x + blockDim.x * threadIdx.y] = val;
    for (int step = blockDim.x / 2; step > 0; step /= 2)
    {
        if (threadIdx.x < step)
        {
            tmp[threadIdx.x + blockDim.x * threadIdx.y] = max(tmp[threadIdx.x + blockDim.x * threadIdx.y], tmp[threadIdx.x + blockDim.x * threadIdx.y + step]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        globalMax = tmp[blockDim.x * threadIdx.y];
    }
    __syncthreads();

    val = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        val += expf(input[row * cols + i] - globalMax);
    }
    tmp[threadIdx.x + blockDim.x * threadIdx.y] = val;
    for (int step = blockDim.x / 2; step > 0; step /= 2)
    {
        if (threadIdx.x < step)
        {
            tmp[threadIdx.x + blockDim.x * threadIdx.y] += tmp[threadIdx.x + blockDim.x * threadIdx.y + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        globalSum = tmp[blockDim.x * threadIdx.y];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        output[row * cols + i] = expf(input[row * cols + i] - globalMax) * __fdividef(1.0F, globalSum);
    }
}

void softmax_gpu_baseline(const float *cpu_input, float *cpu_output, int rows, int cols, int times)
{
    const size_t rows_per_block = 1;
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
        softmax<<<grid_dim, block_dim>>>(input, output, rows, cols);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);

    cudaMemcpy(cpu_output, output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(input);
    cudaFree(output);

    printf("kernel time:%.4fms\n", ker_time/repeat);
}

bool is_aligned128(void *ptr) {
    return ((uintptr_t)ptr & 0x7F) == 0;
}
