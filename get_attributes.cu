#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("Number of Streaming Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("Total amount of global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Clock rate: %.2f GHz\n", deviceProp.clockRate / 1e6);
        printf("Total amount of constant memory: %.2f KB\n", deviceProp.totalConstMem / 1024.0);
        printf("Total amount of shared memory per block: %.2f KB\n", deviceProp.sharedMemPerBlock / 1024.0);
        printf("Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("Warp size: %d\n", deviceProp.warpSize);
        printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Maximum sizes of each dimension of a block: %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("Maximum sizes of each dimension of a grid: %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("\n");
    }

    return 0;
}
