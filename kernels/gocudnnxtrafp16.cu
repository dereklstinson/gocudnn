#include <cuda.h>
#include <stdbool.h>
#include <cuda_fp16.h>

#define CUDA_GRID_LOOP_X(i, n)                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

#define CUDA_GRID_AXIS_LOOP(i, n, axis)                                 \
    for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n; \
         i += blockDim.axis * gridDim.axis)




//forwardPrelu does the forward Prelu

    
//backwardPrelu does the backprop of the parametric float


/* 
Leaky functions
*/














//MakePlanarImageBatchesUint8 - for this to work all the each batch should have the same amount of channels and all the channels
//need to be the same size 
extern "C" __global__ void MakePlanarImageBatchesUint8(const int XThreads, //Should be channel size
                                                 const int Batches,
                                                 const int channelsperbatch,
                                                 const float *Srcs, //all the channels for everything.
                                                 float *dest)
{
    const int batchsize = XThreads*channelsperbatch;
    for (int i = 0;i<Batches;i++)
    {
        for (int j = 0;j<channelsperbatch;j++)
        {
            CUDA_GRID_LOOP_X(xIdx, XThreads)
            {
               dest[(i*batchsize)+(j*XThreads)+xIdx]=Srcs[(j*XThreads)+xIdx];
            }
        }
    
    }
}