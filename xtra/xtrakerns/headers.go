package xtrakerns

//Headers are used for all kernels
const Headers = `
#include <cuda.h>
#include <stdbool.h>
#include <cuda_fp16.h>
`

//Defines are used for all kernels
const Defines = `
#define StartAxis(i,axis) int i = blockIdx.axis * blockDim.axis + threadIdx.axis;
#define CUDA_GRID_LOOP_X(i, n)                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

#define CUDA_GRID_AXIS_LOOP(i, n, axis)                                 \
    for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n; \
         i += blockDim.axis * gridDim.axis)
`
