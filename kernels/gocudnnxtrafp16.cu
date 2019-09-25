#include <cuda.h>
#include <stdbool.h>
#include <cuda_fp16.h>

#define CUDA_GRID_LOOP_X(i, n)                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

#define CUDA_GRID_AXIS_LOOP(i, n, axis)                                 \
    for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n; \
         i += blockDim.axis * gridDim.axis)



//ThreshForward is kind of memory expensive, mostly because it is experimental.
//To test start the positive at random uniform numbers between .9 and 1.1
//and do the negcoefs between .01 and .2 or something along those lines.
//maybe the threshold should be between -.3 and .3 uniform number


extern "C" __global__ void ThreshBackwardhalf(const int XThreads,
                                          const int batchsize,
                                          const __half *x,
                                          __half *dx,
                                          const __half *dy,
                                          const __half *negcoefs,
                                          __half *dnegcoefs,
                                          const __half *threshhold,
                                          const __half *poscoefs,
                                          __half *dposcoefs)
{

    for (int i=0;i<batchsize;i++)
    {
        int stride=XThreads*i;
            CUDA_GRID_LOOP_X(xIdx,XThreads)
            {
                if (__hgt(x[stride+xIdx],threshhold[xIdx]))
                {
                  //  dx[stride+xIdx]=  poscoefs[xIdx]*dy[stride+xIdx];
                    dx[stride+xIdx]=__hmul(dy[stride+xIdx],poscoefs[xIdx]);
                   // dposcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];
dposcoefs[xIdx]=__hfma(dy[xIdx],x[stride+xIdx],dposcoefs[xIdx])
                }
                else
                {
                    dx[stride+xIdx]=  negcoefs[xIdx]*dy[stride+xIdx];
                    dnegcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];
                }
            }
    }
}

//forwardPrelu does the forward Prelu
extern "C" __global__ void PreluForward(const int XThreads,
                                        const int batchsize,
                                        const float *x,
                                        float *y,
                                        const float *coefs)
{
  
    for (int i=0;i<batchsize;i++)
    {
        int stride=XThreads*i;
            CUDA_GRID_LOOP_X(xIdx,XThreads)
            {
                if (x[stride+xIdx]>0)
                {
                    y[stride+xIdx]=  x[stride+xIdx];
                }
                else
                {
                    y[stride+xIdx]=  coefs[xIdx]*x[stride+xIdx];
                }
            }
    }
   
}
    
//backwardPrelu does the backprop of the parametric float

extern "C" __global__ void PreluBackward(const int XThreads,
                                                          const int batchsize,
                                                          float *dx,
                                                          const float *x,
                                                          const float *dy,
                                                          const float *coefs,
                                                          float *dcoefs)
{
    for (int i=0;i<batchsize;i++)
    {
        int stride=XThreads*i;
            CUDA_GRID_LOOP_X(xIdx,XThreads)
            {
                if (x[stride+xIdx]>0)
                {
                    dx[stride+xIdx]=  dy[stride+xIdx];
                }
                else
                {
                    dx[stride+xIdx]=  coefs[xIdx]*dy[stride+xIdx];
                    dcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];
                }
            }
    }
}

/*
Leaky functions
*/

extern "C" __global__ void forwardleakyfloatalphabeta(const int length,
                                             const float *x,
                                             float *y,
                                             const float coef,
                                             const float alpha,
                                              const float beta)
{

    CUDA_GRID_LOOP_X(i, length)
    {
        const float previous = y[i];
        if (x[i] > 0.0)
        {
            const float current = x[i];
            y[i] = (beta*previous) + (alpha *current) ;
        }
        else
        {
              const float current = x[i]*coef;
           y[i] = (beta*previous) + (alpha *current) ;
        }
          __syncthreads();
    }
}
extern "C" __global__ void backwardleakyfloatalphabeta(const int length,
                                              const float *x,
                                              float *dx,
                                              const float *dy,
                                              const float coef,
                                              const float alpha,
                                              const float beta)
{

    CUDA_GRID_LOOP_X(i, length)
    {
        const float previous = dx[i];
        if (x[i] > 0.0)
        {
            const float current= dy[i];
            dx[i] =(beta *previous) + (current * alpha);
        }
        else
        {
            const float current= dy[i]*coef;
            dx[i] = (beta *previous) + (current * alpha);
        }
        __syncthreads();
    }
}

extern "C" __global__ void forwardleakyfloatalpha(const int length,
                                             const float *x,
                                             float *y,
                                             const float coef,
                                             const float alpha)
{

    CUDA_GRID_LOOP_X(i, length)
    {
        
        if (x[i] > 0.0)
        {
            y[i] = alpha *x[i];
        }
        else
        {
            const float current=x[i]*coef;
            y[i] =current * alpha;
        }
         __syncthreads();
    }
}
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