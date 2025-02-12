#include <stdio.h>
#include <cuda.h>
#include "helper_function.h"

const size_t COLS = 1024;
const size_t ROWS = 1024;

__global__ void softmax(const float *input, float *output, int M, int N)
{
    int row = blockIdx.x;
    __shared__ float tmp[COLS];
    __shared__ float globalMax;
    __shared__ float globalSum;

    float val = -__FLT_MAX__;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        val = max(val, input[row * N + i]);
    }
    tmp[threadIdx.x] = val;
    for (int step = COLS / 2; step > 0; step /= 2)
    {
        if (threadIdx.x < step)
        {
            tmp[threadIdx.x] = max(tmp[threadIdx.x], tmp[threadIdx.x + step]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        globalMax = tmp[0];
    }
    __syncthreads();

    val = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        val += __expf(input[row * N + i] - globalMax);
    }
    tmp[threadIdx.x] = val;
    __syncthreads();
    for (int step = COLS / 2; step > 0; step /= 2)
    {
        if (threadIdx.x < step)
        {
            tmp[threadIdx.x] += tmp[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        globalSum = tmp[0];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += COLS)
    {
        output[row * N + i] = __expf(input[row * N + i] - globalMax) * __fdividef(1.0F, globalSum);
    }
}

void cpu_softmax(const float *cpu_input, float *cpu_output, int M, int N)
{
    double st, ela;
    st = get_walltime();

    int num_block = M;
    dim3 block_dim(N, 1, 1);
    dim3 grid_dim(num_block, 1, 1);

    float *input, *output;
    cudaMalloc((void **)&input, M * N * sizeof(float));
    cudaMalloc((void **)&output, M * N * sizeof(float));
    cudaMemcpy(input, cpu_input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float ker_time = 0;
    // int repeat = 20;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // for (int i = 0; i < repeat; i++)
    // {
        softmax<<<grid_dim, block_dim>>>(input, output, M, N);
    // }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(cpu_output, output, M * N, cudaMemcpyDeviceToHost);

    cudaFree(input);
    cudaFree(output);

    ela = get_walltime() - st;

    printf("kernel time:%.4fms, use time:%.4f\n", ker_time, ela);
}

int main()
{
    float *cpu_input, *cpu_output;
    cpu_input = (float *)malloc(ROWS * COLS * sizeof(float));
    cpu_output = (float *)malloc(ROWS * COLS * sizeof(float));
    for (int i = 0; i < COLS * ROWS; i++)
    {
        cpu_input[i] = i % 10;
    }
    cpu_softmax(cpu_input, cpu_output, ROWS, COLS);
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
        max_error = max(max_error, abs(cpu_output[i] - cpu_output[i]));
    }
    printf("max_error: %f\n", max_error);
    free(cpu_input);
    free(cpu_output);
    return 0;
}
