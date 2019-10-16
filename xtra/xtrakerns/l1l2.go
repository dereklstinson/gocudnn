package xtrakerns

//L1L2 are the l1l2 functions for weight normalization
func L1L2() Kernel {
	return Kernel{
		Name: `L1L2`,
		Code: `extern "C" __global__ void L1L2(
			const int length,
			float *dw,          //input and output
			const float *w,     //input needs to ba an array
			float *l1,          //output set to zero
			float *l2,          //output set to zero
			const float batch,  // should be an int but just send it as a float
			const float decay1, //input
			const float decay2)
		{ //input
		
			CUDA_GRID_LOOP_X(i, length)
			{
		
				atomicAdd(l1, abs(w[i]) * decay1);
				atomicAdd(l2, (w[i] * w[i] * decay2) / 2.0);
				const float gradl1 = decay1 * (w[i] > 0 ? 1 : -1);
				const float gradl2 = w[i] * decay2;
				dw[i] = (dw[i] + gradl2 + gradl1) / batch;
			}
		}`,
	}
}

//L1L2FP16 is the L1L2 normalization functions
func L1L2FP16() Kernel {
	return Kernel{
		Name: `L1L2FP16`,
		Code: `extern "C" __global__ void L1L2FP16(
			const int length,
			__half *dw,          //input and output
			const __half *w,     //input needs to ba an array
			__half *l1,          //output set to zero
			__half *l2,          //output set to zero
			const __half batch,  // should be an int but just send it as a float
			const __half decay1, //input
			const __half decay2)
		{ //input
		  const __half one1 = __float2half(1.0);
			const __half zero0 = __float2half(0);
			CUDA_GRID_LOOP_X(i, length)
			{
				__half abs = w[i];
				if (__hlt(abs,zero0)){
					abs=-abs;
				}
				//atomicAdd(l1, abs(w[i]) * decay1);
				atomicAdd(l1,__hmul(abs,decay1));
				//atomicAdd(l2, (w[i] * w[i] * decay2) / 2.0);
				atomicAdd(l2, __hdiv(__hmul(__hmul(w[i] , w[i]) , decay2) , 2.0));
				//const float gradl1 = decay1 * (w[i] > 0 ? 1 : -1);
				const __half gradl1 = __hmul(decay1, (__hgt(w[i],zero0) ? one1 : -one1));
				//const float gradl2 = w[i] * decay2;
				const __half gradl2 = __hmul(w[i] ,decay2);
				//dw[i] = (dw[i] + gradl2 + gradl1) / batch;     
				dw[i] = __hdiv(__hadd(__hadd(dw[i], gradl2) , gradl1) , batch);
			}
		}`,
	}
}
