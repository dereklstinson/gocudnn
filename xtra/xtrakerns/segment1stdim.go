package xtrakerns

//Segment1stDim -- is paired with the host -- it segments the first dim of a tensor
func Segment1stDim() Kernel {
	return Kernel{
		Name: `Segment1stDim`,
		Code: `extern "C" __global__ void Segment1stDim(const int start_index, const float *src, float *dst, const int size)
		{
			int i = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x) + threadIdx.x;
			int start_location = start_index * size;
			if (i < size)
			{
				dst[i] = src[start_location + i];
			}
		}`,
	}
}

//Segment1stDimFP16 -- is paired with the host -- it segments the first dim of a tensor
func Segment1stDimFP16() Kernel {
	return Kernel{
		Name: `Segment1stDimFP16`,
		Code: `extern "C" __global__ void Segment1stDimFP16(const int start_index, const __half *src, __half *dst, const int size)
		{
			int i = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x) + threadIdx.x;
			int start_location = start_index * size;
			if (i < size)
			{
				dst[i] = src[start_location + i];
			}
		}`,
	}
}
