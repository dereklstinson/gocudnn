package xtrakerns

//Kernel is used to build kernels
type Kernel struct {
	Name string
	Code string
}

//SwapEveryOther will swap the batches between 2 tensors.
//It will be either the even or the odd.
//Both tensors have to be equal in size and dims.
//if even is >0 then it will do the even batches.
//Make sure labels are swapped on host end.
func SwapEveryOther() Kernel {
	return Kernel{
		Name: `SwapEveryOther`,
		Code: `
		extern "C" __global__ void SwapEveryOther(
			const int xThreads, //total batches
			const int totalbatches,
			float *t1,
			float *t2,
		   const int start,
		const int stride)
		{
		const int BVol = xThreads;
		
					for (int i =start;i<totalbatches;i+=stride)
				{   
						CUDA_GRID_LOOP_X(xIdx, xThreads)
						{ 
							const float swapper =  t1[(i*BVol)+(xIdx)];
							t1[(i*BVol) +xIdx]=t2[(i*BVol)+xIdx];
							t2[(i*BVol)+xIdx]=swapper;
						}
		
					__syncthreads();
				}    
		}`,
	}
}

//SwapEveryOtherFP16 will swap the batches between 2 tensors.
//It will be either the even or the odd.
//Both tensors have to be equal in size and dims.
//if even is >0 then it will do the even batches.
//Make sure labels are swapped on host end.
func SwapEveryOtherFP16() Kernel {
	return Kernel{
		Name: `SwapEveryOtherFP16`,
		Code: `
extern "C" __global__ void SwapEveryOtherFP16(
    const int n, //total batches
    const int totalbatches,
    __half *t1,
    __half *t2,
   const int start,
const int stride)
{
StartAxis(stx,x)
const int BVol = n/2;
__half2 *t1h=(half2 *)t1;
__half2 *t2h=(half2 *)t2;

            for (int i =start;i<totalbatches;i+=stride)
        {
     
            
                CUDA_GRID_LOOP_X(xIdx, BVol)
                { 
                    const __half2 swapper =  t1h[(i*BVol)+(xIdx)];
                    t1h[(i*BVol) +xIdx]=t2h[(i*BVol)+xIdx];
                    t2h[(i*BVol)+xIdx]=swapper;
                }
                if (stx==0 && (n%2)){
                    const int xIdx=n-1;
                    const __half swapper =  t1[(i*n)+(xIdx)];
                    t1[(i*n) +(xIdx)]=t1[(i*n)+(xIdx)];
                    t2[(i*n)+(xIdx)]=swapper;
                }

            __syncthreads();
        }      
}`,
	}
}

//SwapUpperLower will swap either the upper or lower batches
//Right Now inverse doesn't do anything
func SwapUpperLower() Kernel {
	return Kernel{
		Name: `SwapUpperLower`,
		Code: `extern "C" __global__ void SwapUpperLower(
			const int xThreads, //batchsize
			const int yThreads, //batchvol
			float *t1,
			float *t2,
			const int t1upper,
			const int t2upper,
			const int inverse)
		{
		const int BVol = yThreads;
		  
			if (t1upper>0)
			{
				CUDA_GRID_AXIS_LOOP(xIdx, xThreads/2,x)
				{ 
					int t2Idx;
					if (t2upper>0){
						t2Idx=xIdx;
					}else{
						t2Idx=xThreads/2 +xIdx;
					}
				   
					if (xIdx < xThreads && t2Idx<xThreads)
					{
						CUDA_GRID_AXIS_LOOP(yIdx, yThreads,y)
						{
							
							const float swapper =  t1[(xIdx*BVol)+(yIdx)];
							t1[(xIdx*BVol) +yIdx]=t2[(t2Idx*BVol)+yIdx];
							t2[(xIdx*BVol)+yIdx]=swapper;
						} 
					}
				}   
			}
			else  
			{
				CUDA_GRID_AXIS_LOOP(xIdx, xThreads/2,x)
				{
					const int halfIdx=(xThreads/2)+xIdx;
					int t2Idx;
					if (t2upper>0){
						t2Idx=xIdx;
					}else{
						t2Idx=halfIdx;
					}
				 
					if (halfIdx < xThreads)
					{
						CUDA_GRID_AXIS_LOOP(yIdx, yThreads,y)
						{
							const float swapper =  t1[(halfIdx*BVol)+(yIdx)];
							t1[(halfIdx*BVol) +yIdx]=t2[(t2Idx*BVol)+yIdx];
							t2[(halfIdx*BVol)+yIdx]=swapper;
						}
					}
				}   
			}
		}`,
	}
}

//SwapUpperLowerFP16 is like the FP32 version
func SwapUpperLowerFP16() Kernel {
	return Kernel{
		Name: `SwapUpperLowerFP16`,
		Code: `extern "C" __global__ void SwapUpperLowerFP16(
			const int xThreads, //batchsize
			const int yThreads, //batchvol
			__half *t1,
			__half *t2,
			const int t1upper,
			const int t2upper,
			const int inverse)
		{
		const int BVol = yThreads;
			if (t1upper>0)
			{
				CUDA_GRID_AXIS_LOOP(xIdx,xThreads/2,x)
				{ 
					int t2Idx;
					if (t2upper>0){
						t2Idx=xIdx;
					}else{
						t2Idx=xThreads/2 +xIdx;
					}
				   
					if (xIdx < xThreads && t2Idx<xThreads)
					{
						CUDA_GRID_AXIS_LOOP(yIdx, BVol,y)
						{
							
							const __half swapper =  t1[(xIdx*BVol)+(yIdx)];
							t1[(xIdx*BVol) +yIdx]=t2[(t2Idx*BVol)+yIdx];
							t2[(xIdx*BVol)+yIdx]=swapper;
						} 
					}
				}
			   
			}
			else  
			{
				CUDA_GRID_AXIS_LOOP(xIdx, xThreads/2,x)
				{
					const int halfIdx=(xThreads/2)+xIdx;
					int t2Idx;
					if (t2upper>0){
						t2Idx=xIdx;
					}else{
						t2Idx=halfIdx;
					}
				 
					if (halfIdx < xThreads)
					{
						CUDA_GRID_AXIS_LOOP(yIdx, yThreads,y)
						{
							const __half swapper =  t1[(halfIdx*BVol)+(yIdx)];
							t1[(halfIdx*BVol) +yIdx]=t2[(t2Idx*BVol)+yIdx];
							t2[(halfIdx*BVol)+yIdx]=swapper;
						}
					}
				}   
			}
		}`,
	}
}
