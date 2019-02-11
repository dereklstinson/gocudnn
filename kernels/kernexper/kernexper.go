package kernexper

const kerneldefines = `
#define CUDA_GRID_LOOP_X(i, n)                                 \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
	 i += blockDim.x * gridDim.x)

#define CUDA_GRID_AXIS_LOOP(i, n, axis)                                 \
for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n; \
	 i += blockDim.axis * gridDim.axis)
`

type Kernel struct {
}

var functionbegning = `extern "C" __global__ void`
