#include "helper_function.h"
#include <stdio.h>
#include <math.h>

const size_t COLS = 1024;
const size_t ROWS = 1024;

__global__ void softmax(cudaTextureObject_t texObj, float *output, int M, int N)
{
    int row = blockIdx.x;
    __shared__ float tmp[COLS];
    __shared__ float globalMax;
    __shared__ float globalSum;
    
    tmp[threadIdx.x] = -__FLT_MAX__;
    for (int step = COLS / 2; step > 0; step /= 2)
    {
        if (threadIdx.x < step)
        {
            // tmp[threadIdx.x] = max(tmp[threadIdx.x], tmp[threadIdx.x + step]);
            tmp[threadIdx.x] = max(tmp[threadIdx.x], tex2D<float>(texObj, threadIdx.x, step));
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        globalMax = tmp[0];
    }
    __syncthreads();

    float val = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        val += __expf(tex2D<float>(texObj, row, i) - globalMax);
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
        output[row * N + i] = __expf(tex2D<float>(texObj, row, i) - globalMax) * __fdividef(1.0F, globalSum);
    }
}

void gpu_softmax(const float *cpu_input, float *cpu_output, int M, int N)
{
    double st, ela;
    st = get_walltime();
    cudaArray *cu_array;
    float *output;
    cudaChannelFormatKind kind = cudaChannelFormatKindFloat;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, kind);

    cudaError_t returnValue;
    returnValue = cudaMallocArray(&cu_array, &channelDesc, M, N);
    returnValue = (cudaError_t)(returnValue | cudaMemcpy(cu_array, cpu_input, M * N * sizeof(float), cudaMemcpyHostToDevice));

    if(returnValue != cudaSuccess)
        printf("\n Got error while running CUDA API Array Copy\n");

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cu_array;
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    returnValue = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    cudaMalloc((void **)&output, M * N * sizeof(float));

    dim3 block_dim(N, 1, 1);
    dim3 grid_dim(M, 1, 1);
    cudaEvent_t start, stop;
    float ker_time = 0;
    int repeat = 20;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < repeat; i++)
    {
        softmax<<<grid_dim, block_dim>>>(texObj, output, M, N);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(cpu_output, output, M * N, cudaMemcpyDeviceToHost);

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cu_array);
    cudaFree(output);

    ela = get_walltime() - st;

    printf("kernel time:%.4fms, use time:%.4f\n", ker_time/repeat, ela);
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
    gpu_softmax(cpu_input, cpu_output, ROWS, COLS);
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
}