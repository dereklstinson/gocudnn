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





extern "C" __global__ void backwardleakyfloatalpha(const int length,
                                              const float *x,
                                              float *dx,
                                              const float *dy,
                                              const float coef,
                                              const float alpha)
{
 
    CUDA_GRID_LOOP_X(i, length)
    {

        if (x[i] > 0.0)
        {
            dx[i] = dy[i]*alpha;
        }
        else
        {
            const float current=dy[i]*coef;
            dx[i] = current *alpha;
        }
         __syncthreads();
    }
}


extern "C" __global__ void forwardleakyfloat(const int length,
                                             const float *x,
                                             float *y,
                                             const float coef)
{
    CUDA_GRID_LOOP_X(i, length)
    {
        if (x[i] > 0.0)
        {
            y[i] = x[i];
        }
        else
        {
            y[i] = x[i] * coef;
        }
    }
}
extern "C" __global__ void backwardleakyfloat(const int length,
                                              const float *x,
                                              float *dx,
                                              const float *dy,
                                              const float coef)
{

    CUDA_GRID_LOOP_X(i, length)
    {

        if (x[i] > 0.0)
        {

            dx[i] = dy[i];
        }
        else
        {

            dx[i] = dy[i] * coef;
        }
    }
}

extern "C" __global__ void MSELoss(const int length, 
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

   
}
extern "C" __global__ void MSELossbyBatches(const int xthreads,const int ythreads, float *errors, const float *target, const float *networkout, float *loss)
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
}
extern "C" __global__ void ConcatForwardNCHW( const int XThreads,
                                              const int Batches,
                                              const int Channels1,
                                              const int src1vol,
                                              const float *Src1,
                                              const int Channels2,
                                              const int src2vol,
                                              const float *Src2,
                                              float *dest)
{
    for (int i = 0;i<Batches;i++)
    {
        const int Stride= Batches*(src1vol+src2vol);
        const int src1batchstride=src1vol*i;
        const int src2batchstride=src2vol*i;
        for (int j=0;j<Channels1;j++)
        {
            CUDA_GRID_LOOP_X(xIdx, XThreads)
            {
           dest[Stride+(j*XThreads)+xIdx]  = Src1[src1batchstride+(j*XThreads)+xIdx];
            }
        }
        for (int j=0;j<Channels2;j++){
            CUDA_GRID_LOOP_X(xIdx, XThreads)
            {
           dest[Stride+(j*XThreads)+src1vol+xIdx]  = Src2[src2batchstride+(j*XThreads)+xIdx];
            }
        }
    }
}
extern "C" __global__ void ConcatBackwardNCHW( const int XThreads,
                                              const int Batches,
                                              const int Channels1,
                                              const int src1vol,
                                               float *Src1,
                                              const int Channels2,
                                              const int src2vol,
                                               float *Src2,
                                              const float *dest)
{
    for (int i = 0;i<Batches;i++)
    {
        const int Stride= Batches*(src1vol+src2vol);
        const int src1batchstride=src1vol*i;
        const int src2batchstride=src2vol*i;
        for (int j=0;j<Channels1;j++)
        {
            CUDA_GRID_LOOP_X(xIdx, XThreads)
            {
                 Src1[src1batchstride+(j*XThreads)+xIdx]=  dest[Stride+(j*XThreads)+xIdx];  
            }
        }
        for (int j=0;j<Channels2;j++){
            CUDA_GRID_LOOP_X(xIdx, XThreads)
            {
                Src2[src2batchstride+(j*XThreads)+xIdx]  = dest[Stride+(j*XThreads)+src1vol+xIdx];  
            }
        }
    }
}
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