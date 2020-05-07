#include <cuda.h>
#include <stdbool.h>
#include <cuda_fp16.h>
#define StartAxis(i,axis) int i = blockIdx.axis * blockDim.axis + threadIdx.axis;
#define GRID_LOOP_X(i, n)                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

#define GRID_AXIS_LOOP(i, n, axis)                                 \
    for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n; \
         i += blockDim.axis * gridDim.axis)


extern "C" __global__ void Transpose(int numthreads,
               const float *src,
               const int *buf,
               const int ndims,
               float *dest)
{
    const int *src_strides = buf; 
    const int *dest_strides = &buf[ndims];
    const int *perm = &buf[ndims * 2];

    GRID_LOOP_X(destIdx, numthreads)
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



/*SwapEveryOther will swap the batches between 2 tensors. 
 It will be either the even or the odd.
   Both tensors have to be equal in size and dims.
   if even is >0 then it will do the even batches.
   Make sure labels are swapped on host end.
   */
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
                GRID_LOOP_X(xIdx, xThreads)
                { 
                    const float swapper =  t1[(i*BVol)+(xIdx)];
                    t1[(i*BVol) +xIdx]=t2[(i*BVol)+xIdx];
                    t2[(i*BVol)+xIdx]=swapper;
                }

            __syncthreads();
        }    
}



//SwapUpperLower will swap either the upper or lower batches
//Right Now inverse doesn't do anything
extern "C" __global__ void SwapUpperLower(
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
        GRID_AXIS_LOOP(xIdx, xThreads/2,x)
        { 
            int t2Idx;
            if (t2upper>0){
                t2Idx=xIdx;
            }else{
                t2Idx=xThreads/2 +xIdx;
            }
           
            if (xIdx < xThreads && t2Idx<xThreads)
            {
                GRID_AXIS_LOOP(yIdx, yThreads,y)
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
        GRID_AXIS_LOOP(xIdx, xThreads/2,x)
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
                GRID_AXIS_LOOP(yIdx, yThreads,y)
                {
                    const float swapper =  t1[(halfIdx*BVol)+(yIdx)];
                    t1[(halfIdx*BVol) +yIdx]=t2[(t2Idx*BVol)+yIdx];
                    t2[(halfIdx*BVol)+yIdx]=swapper;
                }
            }
        }   
    }
}

   
//ShapetoBatch4DNHWC Does a stride shape to batch. Make sure values on receiving end are set to zero when s2b is 0
extern "C" __global__ void ShapetoBatch4DNHWC(
    const int xThreads,
    const int yThreads,
    const int zThreads,
    const int hSize,
    const int wSize,
    const int num_original_batches,
    const int BatchVolume,
    const int OriginalVol,
    const int N1,
    const int N2,
    const int hstride,
    const int wstride,
    float *shape,
    float *batch,
    const int h_over_scan,
    const int w_over_scan,
    const bool S2B)
{
    int batch0 = N2 * xThreads * yThreads * zThreads;
    int batch1 = xThreads * yThreads * zThreads;
    int batch2 = yThreads * zThreads;
    int batch3 = zThreads;
    for (int b = 0;b<num_original_batches;b++)
    {
        const int ShapeOffset = OriginalVol*b;
        const int BatchOffset=BatchVolume*b;
    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N2; j++)
        {
            GRID_AXIS_LOOP(xIdx, xThreads, x)
            {
                GRID_AXIS_LOOP(yIdx, yThreads, y)
                {
                    GRID_AXIS_LOOP(zIdx, zThreads, z)
                    {

                        int oh = (hstride * i) + xIdx;
                        int ow = (wstride * j) + yIdx;

                        if (S2B)
                        {
                            if (oh < hSize && ow < wSize)
                            {
                                batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] =
                                    shape[ShapeOffset + (oh * hSize * zThreads) + (ow * zThreads) + zIdx];
                            }
                            else
                            {
                                if (h_over_scan>0 && ow<wSize){
                                batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0;
                                }
                                if (w_over_scan>0 && oh<hSize){
                                    batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0;
                                }
                            }
                        }
                        else
                        {
                            shape[ShapeOffset + (oh * hSize * zThreads) + (ow * zThreads) + zIdx] +=
                                batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx];
                        }
                    }
                }
            }
        }
    }
}
}



//ShapetoBatch4DNCHW Does a stride shape to batch. Make sure values on receiving end are set to zero when s2b is 0


extern "C" __global__ void ShapetoBatch4DNCHW(
    const int xThreads,
    const int yThreads,
    const int zThreads,
    const int hSize,
    const int wSize,
    const int num_original_batches,
    const int BatchVolume,
    const int OriginalVol,
    const int N1,
    const int N2,
    const int hstride,
    const int wstride,
    float *shape,
    float *batch,
    const int h_over_scan,
    const int w_over_scan,
    const bool S2B)
{
    int batch0 = N2 * xThreads * yThreads * zThreads;
    int batch1 = xThreads * yThreads * zThreads;
    int batch2 = xThreads * yThreads;
    int batch3 = yThreads;
    for (int b = 0;b<num_original_batches;b++)
    {
        const int ShapeOffset = OriginalVol*b;
        const int BatchOffset=BatchVolume*b;
    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N2; j++)
        {
            GRID_AXIS_LOOP(xIdx, xThreads, x)
            {
                GRID_AXIS_LOOP(yIdx, yThreads, y)
                {
                    GRID_AXIS_LOOP(zIdx, zThreads, z)
                    {

                        int oh = (hstride * i) + yIdx;
                        int ow = (wstride * j) + zIdx;

                        if (S2B )
                        {
                            if (oh < hSize && ow < wSize)
                            {
                                batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] =
                                    shape[ShapeOffset + (xIdx * wSize * hSize) + (oh * wSize) + ow];
                            }
                            else
                            {
                                if (h_over_scan>0 && ow<wSize){
                                    batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0;
                                }
                                if (w_over_scan>0 && oh<hSize){
                                    batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0; 
                                }
                               
                            }
                        }
                        else
                        {
                            shape[ShapeOffset + (xIdx * wSize * hSize) + (oh * wSize) + ow] +=
                                batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx];
                        }
                    }
                }
            }
        }
    }
}
}


extern "C" __global__ void NearestNeighborNHWC(
    const int aligncorners,
    const int threads,
    const float *src,
    const int src_height,
    const int src_width,
    const int channels,
    const int dest_height,
    const int dest_width,
    const float height_scale,
    const float width_scale,
    float *dest)
{
    GRID_LOOP_X(i, threads)
    {
        int n = i;
        int c = n % channels;
        n /= channels;
        int dest_x = n % dest_width;
        n /= dest_width;
        int dest_y = n % dest_height;
        n /= dest_height;
        const float *src_data_n = &src[n * channels * src_height * src_width];
        const int src_y = fminf((aligncorners) ? (roundf(dest_y * height_scale))
                                               : (floorf(dest_y * height_scale)),
                                src_height - 1);

        const int src_x = fminf((aligncorners) ? (roundf(dest_x * width_scale))
                                               : (floorf(dest_x * width_scale)),
                                src_width - 1);
        const int idx = (src_y * src_width + src_x) * channels + c;
        dest[i] = src_data_n[idx];
    }
}
extern "C" __global__ void NearestNeighborNCHW(
    const int aligncorners,
    const int threads,
    const float *src,
    const int src_height,
    const int src_width,
    const int channels,
    const int dest_height,
    const int dest_width,
    const float height_scale,
    const float width_scale,
    float *dest)
{
    GRID_LOOP_X(i, threads)
    {
        int n = i;
        int dest_x = n % dest_width;
        n /= dest_width;
        int dest_y = n % dest_height;
        n /= dest_height;
        int c = n % channels;
        n /= channels;
        const float *src_data_n = &src[n * channels * src_height * src_width];
        const int src_y = fminf((aligncorners) ? (roundf(dest_y * height_scale))
                                               : (floorf(dest_y * height_scale)),
                                src_height - 1);

        const int src_x = fminf((aligncorners) ? (roundf(dest_x * width_scale))
                                               : (floorf(dest_x * width_scale)),
                                src_width - 1);
        const int idx = (c * src_height * src_width) + (src_y * src_width) + src_x;
        dest[i] = src_data_n[idx];
    }
}
extern "C" __global__ void NearestNeighborNCHWBack(
    const int aligncorners,
    const int threads,
    float *src,
    const int src_height,
    const int src_width,
    const int channels,
    const int dest_height,
    const int dest_width,
    const float height_scale,
    const float width_scale,
    float *dest)
{
    GRID_LOOP_X(i, threads)
    {
        int n = i;
        int src_x = n % src_width;
        n /= src_width;
        int src_y = n % src_height;
        n /= src_height;
        int c = n % channels;
        n /= channels;
        float *src_data_n = &src[n * channels * src_height * src_width];
        const int dest_y = fminf((aligncorners) ? (roundf(src_y * height_scale))
                                                : (floorf(src_y * height_scale)),
                                 dest_height - 1);

        const int dest_x = fminf((aligncorners) ? (roundf(src_x * width_scale))
                                                : (floorf(src_x * width_scale)),
                                 dest_width - 1);
        const int idx = (c * dest_width * dest_height) + (dest_y * dest_width) + dest_x;
        atomicAdd(&src_data_n[idx], dest[i]);
    }
}
extern "C" __global__ void NearestNeighborNHWCBack(
    const int aligncorners,
    const int threads,
    float *src,
    const int src_height,
    const int src_width,
    const int channels,
    const int dest_height,
    const int dest_width,
    const float height_scale,
    const float width_scale,
    float *dest)
{
    GRID_LOOP_X(i, threads)
    {
        int n = i;
        int c = n % channels;
        n /= channels;
        int src_x = n % src_width;
        n /= src_width;
        int src_y = n % src_height;
        n /= src_height;
        float *src_data_n = &src[n * channels * src_height * src_width];
        const int dest_y = fminf((aligncorners) ? (roundf(src_y * height_scale))
                                                : (floorf(src_y * height_scale)),
                                 dest_height - 1);

        const int dest_x = fminf((aligncorners) ? (roundf(src_x * width_scale))
                                                : (floorf(src_x * width_scale)),
                                 dest_width - 1);
        const int idx = (dest_y * dest_width + dest_x) * channels + c;
        atomicAdd(&src_data_n[idx], dest[i]);
    }
}
extern "C" __global__ void AdaGrad(const int length,
                                        float *weights,   //weights input and output
                                        float *dw,        //input and will have to set to zero
                                        float *gsum,      //storage
                                        const float rate, //input
                                        const float eps,
                                        const float dwalpha)
{ //input
    GRID_LOOP_X(cell, length)
    {
        gsum[cell] =  gsum[cell] + (dw[cell] * dw[cell]);
        weights[cell] += -(rate * dw[cell]) / (sqrtf(gsum[cell]) + eps);
        dw[cell] = dw[cell]*dwalpha; //smoothing factor.
    }
}


extern "C" __global__ void Adam(const int n,
                                     float *w,
                                     float *gsum,
                                     float *xsum,
                                     float *dw,
                                     const float rate,
                                     const float beta1,
                                     const float beta2,
                                     const float eps,
                                     const float denombeta1,
                                     const float denombeta2,
                                     const float dwalpha)
{

    GRID_LOOP_X(i, n)
    {
      
        gsum[i] = (beta1 * gsum[i]) + ((1.0 - beta1) * dw[i]);
        float gsumt = gsum[i] /denombeta1;
        xsum[i] = (beta2 * xsum[i]) + ((1.0 - beta2) * (dw[i] * dw[i]));
        float xsumt = xsum[i] / denombeta2;
        w[i] += -(rate * gsumt) / (sqrtf(xsumt) + eps);
        dw[i]=  dwalpha*dw[i]; //smoothing factor
    }
  
}

extern "C" __global__ void AdaDelta(const int length,
                                         float *weights,   //weights input and output
                                         float *gsum,      //storage
                                         float *xsum,      //storage
                                         float *dw,        //input and will have to set to zero
                                         const float rate, //input
                                         const float eps,
                                         const float ro,
                                         const float dwalpha)
{

    GRID_LOOP_X(i, length)
    {

        gsum[i] = (ro * gsum[i]) + ((1.0-ro)*dw[i] * dw[i]);
        const float dx = sqrtf((xsum[i]+eps)/(gsum[i]+eps))*dw[i];
        xsum[i]=(ro*xsum[i])+((1-ro)*dx*dx);
        weights[i] -= dx;
        dw[i] = dw[i]*dwalpha;
    }
}
/*
//This is paired with the host
extern "C" __global__ void Segment1stDim(const int start_index, const float *src, float *dst, const int size)
{
    int i = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x) + threadIdx.x;
    int start_location = start_index * size;
    if (i < size)
    {
        dst[i] = src[start_location + i];
    }
}
//This is paired with the host
extern "C" __global__ void Segment1stDimhalf(const int start_index, const __half *src, __half *dst, const int size)
{
    int i = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x) + threadIdx.x;
    int start_location = start_index * size;
    if (i < size)
    {
        dst[i] = src[start_location + i];
    }
}
*/
extern "C" __global__ void L1L2(
    const int length,
    float *dw,          //input and output
    const float *w,     //input needs to ba an array
    float *l1,          //output set to zero
    float *l2,          //output set to zero
    const float batch,  // should be an int but just send it as a float
    const float decay1, //input
    const float decay2)
{ //input

    GRID_LOOP_X(i, length)
    {

        atomicAdd(l1, abs(w[i]) * decay1);
        atomicAdd(l2, (w[i] * w[i] * decay2) / 2.0);
        const float gradl1 = decay1 * (w[i] > 0 ? 1 : -1);
        const float gradl2 = w[i] * decay2;
        dw[i] = (dw[i] + gradl2 + gradl1) / batch;
    }
}
//ThreshForward is kind of memory expensive, mostly because it is experimental.
//To test start the positive at random uniform numbers between .9 and 1.1
//and do the negcoefs between .01 and .2 or something along those lines.
//maybe the threshold should be between -.3 and .3 uniform number
extern "C" __global__ void ThreshForward(const int XThreads,
                                         const int batchsize,
                                         const float *x,
                                         float *y,
                                         const float *negcoefs,
                                         const float *threshhold,
                                         const float *poscoefs)
{
    for (int i=0;i<batchsize;i++)
    {
        int stride=XThreads*i;
            GRID_LOOP_X(xIdx,XThreads)
            {
                if (x[stride+xIdx]>threshhold[xIdx])
                {
                    y[stride+xIdx]=  x[stride+xIdx]*poscoefs[xIdx];
                }
                else
                {
                    y[stride+xIdx]=  negcoefs[xIdx]*x[stride+xIdx];
                }
            }
    }
}

//Backward 
// Max(x,thresh)
extern "C" __global__ void ThreshBackward(const int XThreads,
                                          const int batchsize,
                                          const float *x,
                                          float *dx,
                                          const float *dy,
                                          const float *negcoefs,
                                          float *dnegcoefs,
                                          const float *threshhold,
                                          float *dthreshhold,
                                          const float *poscoefs,
                                          float *dposcoefs)
{

    for (int i=0;i<batchsize;i++)
    {
        int stride=XThreads*i;
            GRID_LOOP_X(xIdx,XThreads)
            {
                if (x[stride+xIdx]>threshhold[xIdx])  
                {
                    dx[stride+xIdx]=  poscoefs[xIdx]*dy[stride+xIdx];
                    dposcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];

                }
                else
                {
                    dx[stride+xIdx]=  negcoefs[xIdx]*dy[stride+xIdx];
                    dnegcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];
                }
                dthreshhold[xIdx]+=dy[xIdx];
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
            GRID_LOOP_X(xIdx,XThreads)
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
            GRID_LOOP_X(xIdx,XThreads)
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

extern "C" __global__ void LeakyForwardAlphaBeta(const int length,
                                             const float *x,
                                             float *y,
                                             const float coef,
                                             const float alpha,
                                              const float beta)
{

    GRID_LOOP_X(i, length)
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



extern "C" __global__ void LeakyBackwardAlphaBeta(const int length,
                                              const float *x,
                                              float *dx,
                                              const float *dy,
                                              const float coef,
                                              const float alpha,
                                              const float beta)
{

    GRID_LOOP_X(i, length)
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
extern "C" __global__ void LeakyForwardAlpha(const int length,
                                             const float *x,
                                             float *y,
                                             const float coef,
                                             const float alpha)
{

    GRID_LOOP_X(i, length)
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

extern "C" __global__ void LeakyBackwardAlpha(const int length,
                                              const float *x,
                                              float *dx,
                                              const float *dy,
                                              const float coef,
                                              const float alpha)
{
 
    GRID_LOOP_X(i, length)
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


extern "C" __global__ void LeakyForward(const int length,
                                             const float *x,
                                             float *y,
                                             const float coef)
{
    GRID_LOOP_X(i, length)
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

extern "C" __global__ void LeakyBackward(const int length,
                                              const float *x,
                                              float *dx,
                                              const float *dy,
                                              const float coef)
{

    GRID_LOOP_X(i, length)
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
    GRID_LOOP_X(i, length)
    {
        const float y = networkout[i] - target[i];
        errors[i] = y;
        atomicAdd(loss, (y * y) / 2);
    }

   
}

extern "C" __global__ void MSELossbyBatches(const int xthreads,const int ythreads, float *errors, const float *target, const float *networkout, float *loss)
{

    GRID_AXIS_LOOP(xIdx,xthreads,x)
    {
        const int offset=ythreads*xIdx;
            GRID_AXIS_LOOP(yIdx, ythreads,y)
            {  
             const float y = networkout[offset+yIdx] - target[offset+yIdx];
             errors[offset+yIdx] = y;
             atomicAdd(&loss[xIdx], (y * y) / 2);
            }
    }
}

extern "C" __global__ void ConcatForwardNHWCEX( const int XThreads,
                                                const int YThreads,
                                                const int Batches,
                                                const int ConcatBatchVolume,
                                                const int *Channels,
                                                const float *srcs, 
                                                const int *vols, 
                                                const int length,
                                                float *dest)
{
    for (int i = 0;i<Batches;i++)
    {
        const int deststride= Batches*(ConcatBatchVolume);
        int destchanneloffset = 0;
        for (int j=0;j<length;j++)
        {
             const int srcstride=vols[j]*i;
        
            GRID_AXIS_LOOP(xIdx, XThreads,x)
            {
                const int xoffset=xIdx*YThreads;
            
                GRID_AXIS_LOOP(yIdx, YThreads,y)
                {
                    const int yoffset=yIdx*Channels[j];
                    const int destoffset=deststride+xoffset+yoffset+destchanneloffset;
                    const int srcoffset=srcstride+xoffset+yoffset;
                    for (int k=0;k<Channels[j];k++)
                    {
                        dest[destoffset+k]  = srcs[srcoffset+k];
                    }
                }
            }
            destchanneloffset+=Channels[j];
        }
    }
}
extern "C" __global__ void ConcatBackwardNHWCEX( const int XThreads,
                                                const int YThreads,
                                                const int Batches,
                                                const int ConcatBatchVolume,
                                                const int *Channels,
                                                 float *srcs, 
                                                const int *vols, 
                                                const int length,
                                                const float *dest)
{
    for (int i = 0;i<Batches;i++)
    {
        const int deststride= Batches*(ConcatBatchVolume);
        int destchanneloffset = 0;
        for (int j=0;j<length;j++)
        {
             const int srcstride=vols[j]*i;
        
            GRID_AXIS_LOOP(xIdx, XThreads,x)
            {
                const int xoffset=xIdx*YThreads;
            
                GRID_AXIS_LOOP(yIdx, YThreads,y)
                {
                    const int yoffset=yIdx*Channels[j];
                    const int destoffset=deststride+xoffset+yoffset+destchanneloffset;
                    const int srcoffset=srcstride+xoffset+yoffset;
                    for (int k=0;k<Channels[j];k++)
                    {
                      srcs[srcoffset+k]=dest[destoffset+k];
                    }
                }
            }
            destchanneloffset+=Channels[j];
        }
    }
}
extern "C" __global__ void ConcatForwardNCHWEX( const int XThreads,
                                              const int YThreads,
                                              const int Batches,
                                              const int ConcatBatchVolume,
                                              const int *Channels,
                                              const float *srcs, 
                                              const int *vols, 
                                              const int numofsrcs,
                                              float *dest)
{
    for (int i = 0;i<Batches;i++)
    {
        const int destbatchstride= Batches*(ConcatBatchVolume);
        int srcoffsetindest =0;
        for (int j=0;j<numofsrcs;j++)
        {
         
             const int srcbatchstride=vols[j]*i;
            for (int k=0;k<Channels[j];k++)
            {
            GRID_AXIS_LOOP(xIdx, XThreads,x)
            {
            GRID_AXIS_LOOP(yIdx, YThreads,y)
            {
                    dest[destbatchstride+srcoffsetindest+
                    (k*XThreads*YThreads)+(xIdx*YThreads)+yIdx]  =
                    srcs[srcbatchstride+(k*XThreads*YThreads)+
                    (xIdx*YThreads)+yIdx];
            }
            }
            }
              srcoffsetindest+=vols[j];
        }
    }
}
extern "C" __global__ void ConcatBackwardNCHWEX( const int XThreads,
                                                 const int YThreads,
                                                 const int Batches,
                                                 const int ConcatBatchVolume,
                                                 const int *Channels,
                                                 float *srcs, 
                                                 const int *vols, 
                                                 const int numofsrcs,
                                                 const float *dest)
{
    for (int i = 0;i<Batches;i++)
    {
        const int destbatchstride= Batches*(ConcatBatchVolume);
             int srcoffsetindest =0;
        for (int j=0;j<numofsrcs;j++)
        {
         
             const int srcbatchstride=vols[j]*i;
            for (int k=0;k<Channels[j];k++)
            {
            GRID_AXIS_LOOP(xIdx, XThreads,x)
            {
            GRID_AXIS_LOOP(yIdx, YThreads,y)
            {
                
                  srcs[srcbatchstride+(k*XThreads*YThreads)+(xIdx*YThreads)+yIdx] =
                  dest[destbatchstride+srcoffsetindest+(k*XThreads*YThreads)+(xIdx*YThreads)+yIdx];
                   
                }
            }
            }
              srcoffsetindest+=vols[j];
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
            GRID_LOOP_X(xIdx, XThreads)
            {
           dest[Stride+(j*XThreads)+xIdx]  = Src1[src1batchstride+(j*XThreads)+xIdx];
            }
        }
        for (int j=0;j<Channels2;j++){
            GRID_LOOP_X(xIdx, XThreads)
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
            GRID_LOOP_X(xIdx, XThreads)
            {
                 Src1[src1batchstride+(j*XThreads)+xIdx]=  dest[Stride+(j*XThreads)+xIdx];  
            }
        }
        for (int j=0;j<Channels2;j++){
            GRID_LOOP_X(xIdx, XThreads)
            {
                Src2[src2batchstride+(j*XThreads)+xIdx]  = dest[Stride+(j*XThreads)+src1vol+xIdx];  
            }
        }
    }
}
extern "C" __global__ void ConcatForwardNCHWhalf( const int XThreads,
                                              const int Batches,
                                              const int Channels1,
                                              const int src1vol,
                                              const __half *Src1,
                                              const int Channels2,
                                              const int src2vol,
                                              const __half *Src2,
                                              __half *dest)
{
    for (int i = 0;i<Batches;i++)
    {
        const int Stride= Batches*(src1vol+src2vol);
        const int src1batchstride=src1vol*i;
        const int src2batchstride=src2vol*i;
        for (int j=0;j<Channels1;j++)
        {
            GRID_LOOP_X(xIdx, XThreads)
            {
           dest[Stride+(j*XThreads)+xIdx]  = Src1[src1batchstride+(j*XThreads)+xIdx];
            }
        }
        for (int j=0;j<Channels2;j++){
            GRID_LOOP_X(xIdx, XThreads)
            {
           dest[Stride+(j*XThreads)+src1vol+xIdx]  = Src2[src2batchstride+(j*XThreads)+xIdx];
            }
        }
    }
}
extern "C" __global__ void ConcatBackwardNCHWhalf( const int XThreads,
                                                   const int Batches,
                                                   const int Channels1,
                                                   const int src1vol,
                                               __half *Src1,
                                              const int Channels2,
                                              const int src2vol,
                                               __half *Src2,
                                              const __half *dest)
{
    for (int i = 0;i<Batches;i++)
    {
        const int Stride= Batches*(src1vol+src2vol);
        const int src1batchstride=src1vol*i;
        const int src2batchstride=src2vol*i;
        for (int j=0;j<Channels1;j++)
        {
            GRID_LOOP_X(xIdx, XThreads)
            {
                 Src1[src1batchstride+(j*XThreads)+xIdx]=  dest[Stride+(j*XThreads)+xIdx];  
            }
        }
        for (int j=0;j<Channels2;j++){
            GRID_LOOP_X(xIdx, XThreads)
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
            GRID_LOOP_X(xIdx, XThreads)
            {
               dest[(i*batchsize)+(j*XThreads)+xIdx]=Srcs[(j*XThreads)+xIdx];
            }
        }
    
    }
}