

#define CUDA_GRID_LOOP_X(i,n)\
for (int i=blockIdx.x *blockDim.x+threadIdx.x;i<n;\
    i +=blockDim.x*gridDim.x)\

#define CUDA_GRID_AXIS_LOOP(i,n,axis)\
for (int i=blockIdx.axis *blockDim.axis+threadIdx.axis;i<n;\
    i +=blockDim.axis*gridDim.axis)\

#define BLOCK4D_DIMS 2

extern "C" __global__ 
void Transpose(int numthreads,
              const float *src,
              const int *buf,
              const int ndims,
              float *dest){
const int* src_strides=buf;
const int* dest_strides=&buf[ndims];
const int* perm=&buf[ndims*2];

CUDA_GRID_LOOP_X(destIdx,numthreads){
    int srcIdx=0;
    int t=destIdx;
         for (int i=0;i<ndims;++i){
           const int ratio=t/dest_strides[i];
             t-= ratio * dest_strides[i];
             srcIdx+= (ratio *src_strides[perm[i]]);
         }
         dest[destIdx]=src[srcIdx];
    }
}
extern "C" __global__ 
void ShapetoBatch4DNHWC(
                    const int xThreads,
                    const int yThreads,
                    const int zThreads,    
                    const int hSize,
                    const int wSize,          
                    const int BatchOffset,
                    const int ShapeOffset,
                    const int N1,
                    const int N2,
                    float *shape,
                    float *batch,
                    int S2B){
                       int batch0= N2*xThreads*yThreads*zThreads;
                       int batch1=  xThreads*yThreads*zThreads;
                       int batch2= yThreads*zThreads;
                       int batch3= zThreads;
                       for (int i=0;i<N1;i++){
                        for (int j=0;j<N2;j++){
                                CUDA_GRID_AXIS_LOOP(xIdx,xThreads,x){
                                    CUDA_GRID_AXIS_LOOP(yIdx,yThreads,y){
                                        CUDA_GRID_AXIS_LOOP(zIdx,zThreads,z){
                                    

                                            int oh=(xThreads*i)+xIdx;
                                            int ow=(yThreads*j)+yIdx;
                                       
                                 
                                            if (S2B>0){
                                                if (oh<hSize && ow<wSize){
                                                batch[BatchOffset+(i*batch0)+(j*batch1)+(xIdx*batch2)+(yIdx*batch3)+zIdx]=
                                               shape[ShapeOffset+(oh*hSize*zThreads)+(ow*zThreads)+zIdx];
                                                }else{
                                                    batch[BatchOffset+(i*batch0)+(j*batch1)+(xIdx*batch2)+(yIdx*batch3)+zIdx]=0;
                                                }
                                        }else{
                                            shape[ShapeOffset+(oh*hSize*zThreads)+(ow*zThreads)+zIdx]=
                                            batch[BatchOffset+(i*batch0)+(j*batch1)+(xIdx*batch2)+(yIdx*batch3)+zIdx];
                                            
                                        }
                                         
                                      }

                                      }
                                        
                                    }
                                }



                            }
                        }
                    

 extern "C" __global__ 
 void ShapetoBatch4DNCHW(
                     const int xThreads,
                     const int yThreads,
                     const int zThreads,    
                     const int hSize,
                     const int wSize,            
                     const int BatchOffset,
                     const int ShapeOffset,
                     const int N1,
                     const int N2,
                     float *shape,
                     float *batch,
                     int S2B){
                        int batch0= N2*xThreads*yThreads*zThreads;
                        int batch1=  xThreads*yThreads*zThreads;
                        int batch2= xThreads*yThreads;
                        int batch3=  yThreads;
                        for (int i=0;i<N1;i++){
                         for (int j=0;j<N2;j++){
                                 CUDA_GRID_AXIS_LOOP(xIdx,xThreads,x){
                                     CUDA_GRID_AXIS_LOOP(yIdx,yThreads,y){
                                         CUDA_GRID_AXIS_LOOP(zIdx,zThreads,z){
                                     
 
                                             int oh=(yThreads*i)+yIdx;
                                             int ow=(zThreads*j)+zIdx;
                                        
                                  
                                             if (S2B>0){
                                                 if (oh<hSize && ow<wSize){
                                                 batch[BatchOffset+(i*batch0)+(j*batch1)+(xIdx*batch2)+(yIdx*batch3)+zIdx]=
                                                shape[ShapeOffset+(xIdx*wSize*hSize)+(oh*wSize)+ow];
                                                 }else{
                                                     batch[BatchOffset+(i*batch0)+(j*batch1)+(xIdx*batch2)+(yIdx*batch3)+zIdx]=0;
                                                 }
                                         }else{
                                             shape[ShapeOffset+(xIdx*wSize*hSize)+(oh*wSize)+ow]=
                                             batch[BatchOffset+(i*batch0)+(j*batch1)+(xIdx*batch2)+(yIdx*batch3)+zIdx];
                                             
                                         }
                                       
                                   }
                                 }
                                                           
                              }
                            }
                          }
                      }

extern "C" __global__
void ShapetoBatch4Dold(         const int *yNHWC,//[BLOCK4D_DIMS+2],
                             const int *xHW,//[BLOCK4D_DIMS],
                             const int *BlockHW,//[BLOCK4D_DIMS],
                             const int numthrds,
                             const int xN,
                             const int B2S,
                              float *xMem,
                              float *yMem){
CUDA_GRID_LOOP_X(yIdx,numthrds){
int yIdxRemainder = yIdx;

int  batchPos[BLOCK4D_DIMS+2];

for (int dim= BLOCK4D_DIMS+1;dim>=1;--dim){
    batchPos[dim] = yIdxRemainder % yNHWC[dim];
    yIdxRemainder/=yNHWC[dim];
}
batchPos[0]=yIdxRemainder;
int blockIdxRemainder = batchPos[0]/xN;
int xIdx = batchPos[BLOCK4D_DIMS+1];
int xStride =xHW[BLOCK4D_DIMS+1];

const int xyposition=batchPos[0]%xN;
for (int block_Dim=BLOCK4D_DIMS-1;block_Dim>=0;--block_Dim){
    int offset=blockIdxRemainder;
if( block_Dim>0){
    offset %=BlockHW[block_Dim];
}
int xPos=batchPos[block_Dim+1]*BlockHW[block_Dim]+offset;
if (xPos<0||xPos>=xHW[block_Dim]){
    if (B2S<1){
        yMem[yIdx]=0;
    }
    break;
}
xIdx+=xStride*xPos;
xStride*=xHW[block_Dim];
if (block_Dim==0){
    xIdx+=xStride*xyposition;
    if (B2S<1){
        yMem[yIdx]= xMem[xIdx];
    }else{
        xMem[xIdx]= yMem[yIdx];
    }
}
blockIdxRemainder/=BlockHW[block_Dim];
}
}


}

/*
                             const int *yNHWC,//[BLOCK4D_DIMS+2],
                             const int *xHW,//[BLOCK4D_DIMS],
                             const int *BlockHW,//[BLOCK4D_DIMS],
                             const int numthrds,
                             const int xN,
                             const int B2S,
                              float *xMem,
                              float *yMem){



*/
/*
#define NUM_BLOCK_DIMS 2
extern "C"__global__ 
void ShapetoBatch4D( const int args.batch_tensor_shape
    const int32 nthreads, T* space_tensor_ptr,
                    S2BParameters<NUM_BLOCK_DIMS> args, T* batch_tensor_ptr) {
  CUDA_1D_KERNEL_LOOP(batch_tensor_idx, nthreads) {
    int32 remaining_batch_tensor_idx = batch_tensor_idx;

    int32 batch_tensor_pos[NUM_BLOCK_DIMS + 2];

    for (int dim = NUM_BLOCK_DIMS + 1; dim >= 1; --dim) {
      batch_tensor_pos[dim] =
          remaining_batch_tensor_idx % args.batch_tensor_shape[dim];
      remaining_batch_tensor_idx /= args.batch_tensor_shape[dim];
    }
    batch_tensor_pos[0] = remaining_batch_tensor_idx;

    int32 remaining_block_idx = batch_tensor_pos[0] / args.space_tensor_batch;
    int32 space_tensor_idx = batch_tensor_pos[NUM_BLOCK_DIMS + 1];
    int32 space_tensor_stride = args.batch_tensor_shape[NUM_BLOCK_DIMS + 1];
    const int32 space_tensor_batch_pos =
        batch_tensor_pos[0] % args.space_tensor_batch;
    for (int block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; --block_dim) {
      int32 offset = remaining_block_idx;
      if (block_dim > 0) {
        offset %= args.block_shape[block_dim];
      }
      int32 space_tensor_pos =
          batch_tensor_pos[block_dim + 1] * args.block_shape[block_dim] +
          offset - args.pad_start[block_dim];
      if (space_tensor_pos < 0 ||
          space_tensor_pos >= args.space_tensor_spatial_shape[block_dim]) {
        if (B2S == false) {
          // In the space-to-batch case, write zero padding.
          batch_tensor_ptr[batch_tensor_idx] = static_cast<T>(0);
        }
        break;
      }
      space_tensor_idx += space_tensor_stride * space_tensor_pos;
      space_tensor_stride *= args.space_tensor_spatial_shape[block_dim];
      if (block_dim == 0) {
        space_tensor_idx += space_tensor_stride * space_tensor_batch_pos;
        if (B2S == false) {
          batch_tensor_ptr[batch_tensor_idx] =
              ldg(space_tensor_ptr + space_tensor_idx);
        } else {
          space_tensor_ptr[space_tensor_idx] =
              ldg(batch_tensor_ptr + batch_tensor_idx);
        }
      }
      remaining_block_idx /= args.block_shape[block_dim];
    }
  }
}

*/
extern "C" __global__
void adagradfloat(const int length,
                  float *weights, //weights input and output
                  float *dw, //input and will have to set to zero
                  float *gsum, //storage
                  const float rate, //input
                  const float eps){ //input
                
 
    int section = blockIdx.x;
    int index = threadIdx.x;
    int cell = section*blockDim.x +index;
    if (cell<length){
        int holder = gsum[cell];
        gsum[cell]= holder +(dw[cell]*dw[cell]);
        weights[cell]= -(rate*dw[cell])/(sqrtf(gsum[cell])+eps);
        dw[cell]=0.0;
    }  

}


extern "C" __global__
void adamfloat(const int length,
               float *w,
               float *gsum,
               float *xsum,
               float *dw,
               const float rate,
               const float beta1,
               const float beta2,
               const float eps,
               const float counter){

    
    int i = (blockIdx.y*gridDim.x*blockDim.x) + (blockIdx.x*blockDim.x) +  threadIdx.x;

if (i<length){
     gsum[i]=(beta1*gsum[i]) +((1.0-beta1)*dw[i]);
    float gsumt = gsum[i]/(1.0- powf(beta1,counter));
     xsum[i]= (beta2*xsum[i])+((1.0 -beta2)*(dw[i]*dw[i]));
    float xsumt = xsum[i]/(1.0 - powf(beta2,counter));
    w[i] += -(rate*gsumt)/(sqrtf(xsumt)+eps);  
    dw[i]=0.0;

}
    
}


extern "C" __global__
void adadeltafloat( const int length,
                    float *weights, //weights input and output
                    float *gsum, //storage
                    float *xsum, //storage
                    float *dw, //input and will have to set to zero
                    const float rate, //input
                    const float eps){



            int section = blockIdx.x;
            int index = threadIdx.x;
            int cell = section*blockDim.x +index;
if(cell<length){
    gsum[cell]= gsum[cell]+(dw[cell]*dw[cell]);
    weights[cell]= -(rate*dw[cell])/(sqrtf(gsum[cell])+eps);
    dw[cell]=0.0;


}

}

extern "C" __global__
void l1regularizationfloat(const int length,
                           float *dw, //input and output
                           float *w,  //input
                           float *l1, //output
                           float *l2, //output
                           const float batch, // should be an int but just send it as a float
                           const float decay1,
                           const float decay2){

        int section = blockIdx.x;
        int index = threadIdx.x;
        int cell = section*blockDim.x+index;
        float decay = decay1;
        if (cell<length){
            if (dw[cell]<0){
                decay=-decay;
            }
            atomicAdd(l1,w[cell]*decay);
            dw[cell]= (dw[cell]/batch) +decay;


       }

    
}

//This is paired with the host
extern "C" __global__
void Segment1stDim(const int start_index, const float *src,float *dst ,const int size){
    int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
    int start_location=start_index*size;
    if (i<size){
        dst[i]=src[start_location+i];
    }
}



extern "C" __global__
void l2regularizationfloat(
    const int length,
    float *dw, //input and output
    float *w,  //input
    float *l1, //output
    float *l2, //output
    const float batch, // should be an int but just send it as a float
    const float decay1,
    const float decay2){
        int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
    
    if (i<length){
    atomicAdd(l2,(w[i]*w[i]*decay2)/2.0);
    dw[i]= (dw[i]/batch) + w[i]*decay2;
    }
 
}  
extern "C" __global__
void batchregfloat(
    const int length,
    float *dw, //input and output
    const float batch) {// should be an int but just send it as a float
        int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
    if (i<length){
   
    dw[i]/=batch;
    }
 
}  
extern "C" __global__
void l1l2regularizationfloat(
    const int length,
    float *dw, //input and output
    float *w,  //input needs to ba an array
   // int values, //number of values
    float *l1, //output set to zero
    float *l2, //output set to zero
    const float batch, // should be an int but just send it as a float
    const float decay1, //input
    const float decay2){ //input
 
int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
   int decay =decay1 ;
    if (i<length){
        
        if (dw[i]<0){
            decay=-decay;
        }

        atomicAdd(l1,w[i]*decay); 
        atomicAdd(l2,(w[i]*w[i]*decay2)/2.0);
        dw[i]= (dw[i]/batch) + (w[i]*decay2) +decay1;
     }

}  


extern "C" __global__
void forwardParametricfloat(const int length, const int alphalength ,float *x,float *y,  float *alpha){
   int xsize = gridDim.x*blockDim.x;
   int i= blockIdx.x*blockDim.x+threadIdx.x;
   int j = xsize*blockIdx.y+i;

  
if (j<length){
    if (i<alphalength){
        if (x[j]>0.0){
            y[j]=x[j];
        }else{
            y[j]=alpha[i]*x[j];
        }


    }
    
    
}

}

//NHCW, NCWH only matters on the batch channel so for this to work alpha and dalpha are going to have to be the size of 
// HCW.  
extern "C" __global__  
void backwardParametricfloat(const int length, const int alphalength ,float *x, float *dx,float *dy,  float *alpha, float *dalpha){

    int xsize = gridDim.x*blockDim.x;
    int i= blockIdx.x*blockDim.x+threadIdx.x;
    int j = xsize*blockIdx.y+i;
 
if (j<length){
    if (i<alphalength){
    if (x[j]>0.0){
        dx[j]=dy[j];
    }else{
        dx[j]=alpha[i]*dy[j];
        atomicAdd(&dalpha[i],x[j]*dy[j]);
  
    }   
 
}
}
}
extern "C" __global__
void forwardleakyfloat(const int length,float *x,float *y, const float alpha){

int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
if (i<length){
    if (x[i]>0.0){
        y[i]=x[i];
    }else{
        y[i]=alpha*x[i];
    }
    
}

}   

/*
extern "C" __global__
void concatforwardleakyfloatNCHW(const int length, const int batch, const int xAlength, const int xBlength, const int ylength, float *xA, float xB, float *y, const float alpha){

int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
if (i<xAlength){
    if (xA[i*batch]>0.0){
        y[i]=xA[i*batch];
    }else{
        y[i*batch*(ylength)]=alpha*xA[i*batch];
    }
    
}
if (i<xBlength){
    if (xA[i*batch]>0.0){
        y[i]=xB[i*batch];
    }else{
        y[i*batch*(ylength+xAlength)]=alpha*xB[i*batch];
    }
    
}
}   
*/

extern "C" __global__
void backwardleakyfloat(const int length,float *x, float *dx,float *dy, const float alpha){
int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
if (i<length){
    if (x[i]>0.0){
        dx[i]=dy[i];
    }else{
        dx[i]=alpha*dy[i];
    }
    
}

}  
// Doesn't Work and probably wont be used

/*  
extern "C" __global__
void NHWCSegmentedFrom1NHWC(  const int ChannelIndex,
                              const int ChannelLength,
                              const int OriginalXSize,
                              const int OriginalYSize,
                              float *oMem,
                              float *nMem){

              
int y = (blockIdx.y*blockDim.y +threadIdx.y);//Where y is in the memory    (y is the row)                                                  
int x=  (blockIdx.x*blockDim.x +threadIdx.x);
int ylength= blockDim.y*gridDim.y;
int xlength =(blockDim.x*gridDim.x);
int stridex=gridDim.x*xlength;
int stridey=gridDim.y*ylength;
int OriginalY=stridey+y;
int OriginalX=stridex+x;
__shared__ float *SharedMem;
            if (y<ylength&&x<xlength){
                if  (OriginalX<OriginalXSize && OriginalY<OriginalYSize ){
              SharedMem[x*ylength+y] =  oMem[(ChannelIndex*OriginalXSize*OriginalYSize)+(OriginalX*OriginalYSize)+OriginalY];  
            }else{
                SharedMem[x*ylength+y] =0.0;
            }

                
            }
            __syncthreads();
            nMem[(x*gridDim.y*ChannelLength*ylength*xlength)+(y*ChannelLength*ylength*xlength)+(ChannelIndex*ylength*xlength)+(x*ylength)+y]=  SharedMem[x*ylength+y] ;
          
        } 
        
*/        
/*
extern "C" __global__
void NHWCSegmentedFrom1NHWC(  const int N1index,
                              const int N2index,
                              const int N2length,
                              const int ChannelIndex,
                              const int ChannelLength,
                              const int OriginalXSize,
                              const int OriginalYSize,
                              float *oMem,
                              float *nMem){

              
int y = (blockIdx.y*blockDim.y +threadIdx.y);//Where y is in the memory    (y is the row)                                                  
int x=  (blockIdx.x*blockDim.x +threadIdx.x);
int ylength= blockDim.y*gridDim.y;
int xlength =(blockDim.x*gridDim.x);
int stridex=N1index*xlength;
int stridey=N2index*ylength;
int OriginalY=stridey+y;
int OriginalX=stridex+x;
__shared__ float *SharedMem;
            if (y<ylength&&x<xlength){
                if  (OriginalX<OriginalXSize && OriginalY<OriginalYSize ){
              SharedMem[x*ylength*y] =  oMem[(ChannelIndex*OriginalXSize*OriginalYSize)+(OriginalX*OriginalYSize)+OriginalY];  
            }else{
                SharedMem[x*ylength*y] =0.0;
            }

                
            }
            __syncthreads();
            nMem[(x*N2length*ChannelLength*ylength*xlength)+(y*ChannelLength*ylength*xlength)+(ChannelIndex*ylength*xlength)+(x*ylength)+y]=  SharedMem[x*ylength*y] ;
          
        } 
       



  */     