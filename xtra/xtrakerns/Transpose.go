package xtrakerns

//Transpose is the kernel for transpose
func Transpose() Kernel {
	return Kernel{
		Name: `Transpose`,
		Code: `
		extern "C" __global__ void Transpose(int numthreads,
			const float *src,
			const int *buf,
			const int ndims,
			float *dest)
		{
		const int *src_strides = buf; 
		const int *dest_strides = &buf[ndims];
		const int *perm = &buf[ndims * 2];
		
		CUDA_GRID_LOOP_X(destIdx, numthreads)
		{
		int srcIdx = 0;
		int t = destIdx;
		for (int i = 0; i < ndims; ++i)
		{
		 const int ratio = t / dest_strides[i];
		 t -= ratio * dest_strides[i];
		 srcIdx += (ratio * src_strides[perm[i]]);
		}
		dest[destIdx] = src[srcIdx];
		}  
		}
		`,
	}
}

//TransposeFP16 is the kernel for transpose
func TransposeFP16() Kernel {
	return Kernel{
		Name: `TransposeFP16`,
		Code: `
		extern "C" __global__ void TransposeFP16(int numthreads,
			const __half *src,
			const int *buf,
			const int ndims,
			__half *dest)
		{
		const int *src_strides = buf; 
		const int *dest_strides = &buf[ndims];
		const int *perm = &buf[ndims * 2];
		
		CUDA_GRID_LOOP_X(destIdx, numthreads)
		{
		int srcIdx = 0;
		int t = destIdx;
		for (int i = 0; i < ndims; ++i)
		{
		 const int ratio = t / dest_strides[i];
		 t -= ratio * dest_strides[i];
		 srcIdx += (ratio * src_strides[perm[i]]);
		}
		dest[destIdx] = src[srcIdx];
		}  
		}`,
	}
}
