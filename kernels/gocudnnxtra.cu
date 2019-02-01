

#define CUDA_GRID_LOOP_X(i, n)                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

#define CUDA_GRID_AXIS_LOOP(i, n, axis)                                 \
    for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n; \
         i += blockDim.axis * gridDim.axis)


extern "C" __global__ void Int8ToFloat32(const int XThreads, const signed char *src, float *dest){
    CUDA_GRID_LOOP_X(xIdx,XThreads)
    {
        dest[xIdx]= (float) src[xIdx];
    }
}
extern "C" __global__ void Int8ToFloat32Normalize(const int XThreads,const signed char *src, float *dest){
    CUDA_GRID_LOOP_X(xIdx,XThreads)
    {
        dest[xIdx]= ((float) src[xIdx])/255.0;
    }
}
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
     
            
                CUDA_GRID_LOOP_X(xIdx, xThreads)
                { 
                    const float swapper =  t1[(i*BVol)+(xIdx)];
                    t1[(i*BVol) +xIdx]=t2[(i*BVol)+xIdx];
                    t2[(i*BVol)+xIdx]=swapper;
                }

            
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
}
extern "C" __global__ void SwapEveryOtherInt8(
    const int xThreads, 
    const int totalbatches,
   signed char *t1,
   signed  char *t2,
    const int start,
    const int stride)
{
const int BVol = xThreads;

 
        for (int i =start;i<totalbatches;i+=stride)
        {
     
            
                CUDA_GRID_LOOP_X(xIdx, xThreads)
                { 
                    const signed char swapper =  t1[(i*BVol)+(xIdx)];
                    t1[(i*BVol) +xIdx]=t2[(i*BVol)+xIdx];
                    t2[(i*BVol)+xIdx]=swapper;
                }

            
        }
        
           
    }

        



//SwapUpperLower will swap either the upper or lower batches
//Right Now inverse doesn't do anything
extern "C" __global__ void SwapUpperLowerInt8(
    const int xThreads, //batchsize
    const int yThreads, //batchvol
    signed char *t1,
    signed char *t2,
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
    const int S2B)
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
            CUDA_GRID_AXIS_LOOP(xIdx, xThreads, x)
            {
                CUDA_GRID_AXIS_LOOP(yIdx, yThreads, y)
                {
                    CUDA_GRID_AXIS_LOOP(zIdx, zThreads, z)
                    {

                        int oh = (hstride * i) + xIdx;
                        int ow = (wstride * j) + yIdx;

                        if (S2B > 0)
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
    const int S2B)
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
            CUDA_GRID_AXIS_LOOP(xIdx, xThreads, x)
            {
                CUDA_GRID_AXIS_LOOP(yIdx, yThreads, y)
                {
                    CUDA_GRID_AXIS_LOOP(zIdx, zThreads, z)
                    {

                        int oh = (hstride * i) + yIdx;
                        int ow = (wstride * j) + zIdx;

                        if (S2B > 0)
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

extern "C" __global__ void nearestneighborNHWC(
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
    CUDA_GRID_LOOP_X(i, threads)
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

//When calling this function it will have to do the stuff indexes on the destination
extern "C" __global__ void nearestneighborv2NCHW(
    const int xThreads,
    const int yThreads,
    const int zThreads,
    const int batches,
    const float *src,
    const int src_height,
    const int src_width,
    // const int dest_height,
    //  const int dest_width,
    const float hratio,
    const float wratio,
    float *dest)
{
    const int dbatchslide = xThreads * yThreads * zThreads;
    const int dchanslide = yThreads * zThreads;
    const int dhslide = zThreads;
    const int schanslide = src_height * src_width;
    const int sbatchslide = schanslide * xThreads;
    for (int i = 1; i < batches; i++)
    {

        CUDA_GRID_AXIS_LOOP(xIdx, xThreads, x)
        {
            CUDA_GRID_AXIS_LOOP(yIdx, yThreads, y)
            {
                CUDA_GRID_AXIS_LOOP(zIdx, zThreads, z)
                {
                    float ph = floorf(yIdx * hratio);
                    float pw = floorf(zIdx * wratio);
                    dest[(i * dbatchslide) + (xIdx * dchanslide) + (yIdx * dhslide) + zIdx] =
                        src[(int)((i * sbatchslide) + (schanslide * xIdx) + (ph * src_height) + pw)];
                }
            }
        }
    }
}
//When calling this function it will have to do the stuff indexes on the destination
extern "C" __global__ void nearestneighborv2NCHWAddGradient(
    const int xThreads,
    const int yThreads,
    const int zThreads,
    const int batches,
    const float *src,
    const int src_height,
    const int src_width,
    // const int dest_height,
    //  const int dest_width,
    const float hratio,
    const float wratio,
    float *dest)
{
    const int dbatchslide = xThreads * yThreads * zThreads;
    const int dchanslide = yThreads * zThreads;
    const int dhslide = zThreads;
    const int schanslide = src_height * src_width;
    const int sbatchslide = schanslide * xThreads;
    for (int i = 1; i < batches; i++)
    {

        CUDA_GRID_AXIS_LOOP(xIdx, xThreads, x)
        {
            CUDA_GRID_AXIS_LOOP(yIdx, yThreads, y)
            {
                CUDA_GRID_AXIS_LOOP(zIdx, zThreads, z)
                {
                    float ph = floorf(yIdx * hratio);
                    float pw = floorf(zIdx * wratio);
                    dest[(i * dbatchslide) + (xIdx * dchanslide) + (yIdx * dhslide) + zIdx] +=
                        src[(int)((i * sbatchslide) + (schanslide * xIdx) + (ph * src_height) + pw)];
                }
            }
        }
    }
}
extern "C" __global__ void nearestneighborNCHW(
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
    CUDA_GRID_LOOP_X(i, threads)
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
extern "C" __global__ void nearestneighborNCHWBack(
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
    CUDA_GRID_LOOP_X(i, threads)
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
extern "C" __global__ void nearestneighborNHWCBack(
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
    CUDA_GRID_LOOP_X(i, threads)
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

extern "C" __global__ void adagradfloat(const int length,
                                        float *weights,   //weights input and output
                                        float *dw,        //input and will have to set to zero
                                        float *gsum,      //storage
                                        const float rate, //input
                                        const float eps)
{ //input
    CUDA_GRID_LOOP_X(cell, length)
    {

        int holder = gsum[cell];
        gsum[cell] = holder + (dw[cell] * dw[cell]);
        weights[cell] = -(rate * dw[cell]) / (sqrtf(gsum[cell]) + eps);
        dw[cell] = 0.0;
    }
}

extern "C" __global__ void adamfloat(const int length,
                                     float *w,
                                     float *gsum,
                                     float *xsum,
                                     float *dw,
                                     const float rate,
                                     const float beta1,
                                     const float beta2,
                                     const float eps,
                                     const float counter)
{

    CUDA_GRID_LOOP_X(i, length)
    {

        gsum[i] = (beta1 * gsum[i]) + ((1.0 - beta1) * dw[i]);
        float gsumt = gsum[i] / (1.0 - powf(beta1, counter));
        xsum[i] = (beta2 * xsum[i]) + ((1.0 - beta2) * (dw[i] * dw[i]));
        float xsumt = xsum[i] / (1.0 - powf(beta2, counter));
        w[i] += -(rate * gsumt) / (sqrtf(xsumt) + eps);
        dw[i] = 0.0;
    }
}

extern "C" __global__ void adadeltafloat(const int length,
                                         float *weights,   //weights input and output
                                         float *gsum,      //storage
                                         float *xsum,      //storage
                                         float *dw,        //input and will have to set to zero
                                         const float rate, //input
                                         const float eps)
{

    CUDA_GRID_LOOP_X(cell, length)
    {

        gsum[cell] = gsum[cell] + (dw[cell] * dw[cell]);
        weights[cell] = -(rate * dw[cell]) / (sqrtf(gsum[cell]) + eps);
        dw[cell] = 0.0;
    }
}

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

extern "C" __global__ void l1l2regularizationfloat(
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
            CUDA_GRID_LOOP_X(xIdx,XThreads)
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

extern "C" __global__ void ThreshBackward(const int XThreads,
                                          const int batchsize,
                                          const float *x,
                                          float *dx,
                                          const float *dy,
                                          const float *negcoefs,
                                          float *dnegcoefs,
                                          const float *threshhold,
                                          const float *poscoefs,
                                          float *dposcoefs)
{

    for (int i=0;i<batchsize;i++)
    {
        int stride=XThreads*i;
            CUDA_GRID_LOOP_X(xIdx,XThreads)
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

extern "C" __global__ void forwardleakyfloat(const int length,
                                             const float *x,
                                             float *y,
                                             const float coef,
                                             const int PropNan)
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
                                              const float coef,
                                              const int PropNan)
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

extern "C" __global__ void MSELoss(const int length, float *errors, const float *target, const float *networkout, float *loss)
{

    CUDA_GRID_LOOP_X(i, length)
    {
        const float y = networkout[i] - target[i];
        errors[i] = y;
        atomicAdd(loss, (y * y) / 2);
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