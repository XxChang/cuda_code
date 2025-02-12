#include <stdio.h>
#include <cuda.h>
#include "helper_function.h"

const size_t COLS = 32;
const size_t ROWS = 2048;

int main()
{
    float *cpu_input, *gpu_output, *cpu_output;
    cpu_input = (float *)malloc(ROWS * COLS * sizeof(float));
    gpu_output = (float *)malloc(ROWS * COLS * sizeof(float));
    cpu_output = (float *)malloc(ROWS * COLS * sizeof(float));
    for (int i = 0; i < COLS * ROWS; i++)
    {
        cpu_input[i] = i % 10;
    }
    softmax_gpu_baseline(cpu_input, gpu_output, ROWS, COLS, 10);
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