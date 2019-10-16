package xtrakerns

//MSELossFP16 performs the mean squared error loss function
func MSELossFP16() Kernel {
	return Kernel{
		Name: `MSELossFP16`,
		Code: `extern "C" __global__ void MSELossFP16(const int n, 
			__half *errors, 
			const __half *target,
			const __half *networkout, 
			__half *loss,
			const __half alpha,
			const __half beta)
{
StartAxis(stx,x)
int n2=n/2;
__half2 *errors2=(__half2*)errors, *target2=(__half2*)target, *networkout2=(__half2*)networkout, *loss2=(__half2*)loss;
//  const __half2 alpha2=__halves2half2(alpha), beta2=__halves2half2(beta);
const __half2 htwo2=__halves2half2(__float2half(2.0),__float2half(2.0));
const __half htwo= __float2half(2.0);
loss[0]=0;
CUDA_GRID_LOOP_X(i, n2)
{
const __half2 y = __hsub2(networkout2[i] , target2[i]);
errors2[i] = y;
atomicAdd(loss2, __h2div(__hmul2(y , y) ,htwo2));
}
if (stx==0 && (n%2)){
const int i=n-1;
const __half y = __hsub(networkout[i] , target[i]);
errors[i] = y;
atomicAdd(loss, __hdiv(__hmul(y , y) , htwo));
}



}`,
	}
}

//MSELoss performs the mean squared error loss function
func MSELoss() Kernel {
	return Kernel{
		Name: `MSELoss`,
		Code: `extern "C" __global__ void MSELoss(const int length, 
			float *errors, 
			const float *target,
			const float *networkout, 
			float *loss,
			const float alpha,
			const float beta)
{

loss[0]=0;
CUDA_GRID_LOOP_X(i, length)
{
const float y = networkout[i] - target[i];
errors[i] = y;
atomicAdd(loss, (y * y) / 2);
}


}`,
	}
}

//MSELossbyBatches performs the mean squared error loss function by batches
//Good for gans
func MSELossbyBatches() Kernel {
	return Kernel{
		Name: `MSELossbyBatches`,
		Code: `extern "C" __global__ void MSELossbyBatches(const int xthreads,const int ythreads, float *errors, const float *target, const float *networkout, float *loss)
		{
		
			CUDA_GRID_AXIS_LOOP(xIdx,xthreads,x)
			{
				const int offset=ythreads*xIdx;
					CUDA_GRID_AXIS_LOOP(yIdx, ythreads,y)
					{  
					 const float y = networkout[offset+yIdx] - target[offset+yIdx];
					 errors[offset+yIdx] = y;
					 atomicAdd(&loss[xIdx], (y * y) / 2);
					}
			}
		}`,
	}
}

//MSELossbyBatchesFP16 performs the mean squared error loss function by batches
//Good for gans
func MSELossbyBatchesFP16() Kernel {
	return Kernel{
		Name: `MSELossbyBatchesFP16`,
		Code: `extern "C" __global__ void MSELossbyBatchesFP16(const int xthreads,
			const int ythreads,
			 __half *errors, 
			 const __half *target, 
			 const __half *networkout, 
			 __half *loss)
			{
			  const __half htwo= __float2half(2.0);
				CUDA_GRID_AXIS_LOOP(xIdx,xthreads,x)
				{
					const int i=ythreads*xIdx;
						CUDA_GRID_AXIS_LOOP(yIdx, ythreads,y)
						{  
							const __half y = __hsub(networkout[i] , target[i]);
					errors[i] = y;
						 atomicAdd(&loss[xIdx], __hdiv(__hmul(y , y) , htwo));
						}
				}
			}`,
	}
}
