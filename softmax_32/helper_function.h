#ifndef __HELPER_FUNCTION_H__
#define __HELPER_FUNCTION_H__

#include <sys/time.h>


double get_walltime();

void softmax_cpu(const float *input, float *output, int rows, int cols);
void softmax_gpu_baseline(const float *input, float *output, int rows, int cols, int times = 1);
bool is_aligned128(void *ptr);

#endif
