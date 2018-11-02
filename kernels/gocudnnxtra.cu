

#define CUDA_GRID_LOOP_X(i,n)\
for (int i=blockIdx.x *blockDim.x+threadIdx.x;i<n;\
    i +=blockDim.x*gridDim.x)\

#define CUDA_GRID_AXIS_LOOP(i,n,axis)\
for (int i=blockIdx.axis *blockDim.axis+threadIdx.axis;i<n;\
    i +=blockDim.axis*gridDim.axis)\



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
void nearestneighborNHWC(
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
        float *dest){
            CUDA_GRID_LOOP_X(i,threads){
                int n=i;
                int c = n%channels;
                n/=channels;
                int dest_x=n%dest_width;
                n/=dest_width;
                int dest_y=n%dest_height;
                n/=dest_height;
                const float *src_data_n=&src[n*channels*src_height*src_width];
                const int src_y=fminf((aligncorners) ? (roundf(dest_y*height_scale))
                                                     : (floorf(dest_y*height_scale)),
                src_height -1);

                const int src_x=fminf((aligncorners) ? (roundf(dest_x*width_scale))
                                                  : (floorf(dest_x*width_scale)),
                src_width -1);
                const int idx = (src_y*src_width+src_x)*channels+c;
                dest[i]=src_data_n[idx];
    }

}
extern "C" __global__
void nearestneighborNCHW(
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
        float *dest){
            CUDA_GRID_LOOP_X(i,threads){
                int n=i;
                int dest_x=n%dest_width;
                n/=dest_width;
                int dest_y=n%dest_height;
                n/=dest_height;
                int c = n%channels;
                n/=channels;
                const float *src_data_n=&src[n*channels*src_height*src_width];
                const int src_y=fminf((aligncorners) ? (roundf(dest_y*height_scale))
                                                     : (floorf(dest_y*height_scale)),
                src_height -1);

                const int src_x=fminf((aligncorners) ? (roundf(dest_x*width_scale))
                                                  : (floorf(dest_x*width_scale)),
                src_width -1);
                const int idx = (c*src_height*src_width)+(src_y*src_width)+src_x;
                dest[i]=src_data_n[idx];
    }

}
extern "C" __global__
void nearestneighborNCHWBack(
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
        float *dest){
            CUDA_GRID_LOOP_X(i,threads){
                int n=i;
                int src_x=n%src_width;
                n/=src_width;
                int src_y=n%src_height;
                n/=src_height;
                int c = n%channels;
                n/=channels;
                float *src_data_n=&src[n*channels*src_height*src_width];
                const int dest_y=fminf((aligncorners) ? (roundf(src_y*height_scale))
                                                     : (floorf(src_y*height_scale)),
                dest_height -1);

                const int dest_x=fminf((aligncorners) ? (roundf(src_x*width_scale))
                                                  : (floorf(src_x*width_scale)),
                dest_width -1);
                const int idx = (c*dest_width*dest_height)+(dest_y*dest_width)+dest_x;
                atomicAdd(&src_data_n[idx], dest[i]);  
    }

}
extern "C" __global__
void nearestneighborNHWCBack(
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
        float *dest){
            CUDA_GRID_LOOP_X(i,threads){
                int n=i;
                int c = n%channels;
                n/=channels;
                int src_x=n%src_width;
                n/=src_width;
                int src_y=n%src_height;
                n/=src_height;
                float *src_data_n=&src[n*channels*src_height*src_width];
                const int dest_y=fminf((aligncorners) ? (roundf(src_y*height_scale))
                                                     : (floorf(src_y*height_scale)),
                dest_height -1);

                const int dest_x=fminf((aligncorners) ? (roundf(src_x*width_scale))
                                                  : (floorf(src_x*width_scale)),
                dest_width -1);
                const int idx = (dest_y*dest_width+dest_x)*channels+c;
                atomicAdd(&src_data_n[idx], dest[i]);  
    }

}

extern "C" __global__
void adagradfloat(const int length,
                  float *weights, //weights input and output
                  float *dw, //input and will have to set to zero
                  float *gsum, //storage
                  const float rate, //input
                  const float eps){ //input
                    CUDA_GRID_LOOP_X(cell,length){

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

 CUDA_GRID_LOOP_X(i,length){ 



    gsum[i]=(beta1*gsum[i]) +((1.0-beta1)*dw[i]);
    xsum[i]= (beta2*xsum[i])+((1.0 -beta2)*(dw[i]*dw[i]));
    float gsumt = gsum[i]/(1.0- powf(beta1,counter));
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



 CUDA_GRID_LOOP_X(cell,length){

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
float decay = decay1;
CUDA_GRID_LOOP_X(cell,length){
        
    if (w[cell]<0){
         decay=-decay1;
    }else{
        decay=decay1;
    }
        atomicAdd(l1,w[cell]*decay);
        dw[cell]= (dw[cell] +decay)/batch;
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
CUDA_GRID_LOOP_X(i,length){ 
    atomicAdd(l2,(w[i]*w[i]*decay2)/2.0);
    dw[i]= (dw[i] + w[i]*decay2)/batch;
    }
}  

extern "C" __global__
void batchregfloat(
    const int length,
    float *dw, //input and output
    const float batch) {// should be an int but just send it as a float
CUDA_GRID_LOOP_X(i,length){ 
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
float decay =decay1 ;
CUDA_GRID_LOOP_X(i,length){
 

        
        if (w[i]<0){
            decay=-decay1;
        }else{
            decay=decay1;
        }

        atomicAdd(l1,w[i]*decay); 
        atomicAdd(l2,(w[i]*w[i]*decay2)/2.0);
        dw[i]= (dw[i] + (w[i]*decay2) +decay1)/batch;
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
void forwardleakyfloat(const int length,float *x,float *y, const float invalpha){
    CUDA_GRID_LOOP_X(i,length){
        if (x[i]>0.0){
            y[i]=x[i];
        }else{
            y[i]=x[i]/invalpha;
        }
    }  
    
}

  



extern "C" __global__
void backwardleakyfloat(const int length,float *x, float *dx,float *dy, const float invalpha){

CUDA_GRID_LOOP_X(i,length){
    if (x[i]>0.0){
        dx[i]=dy[i];
    }else{
        dx[i]=dy[i]/invalpha;
    }
    
}

}  

extern "C" __global__
void MSELoss(const int length ,float *target,float *networkout,float *errors,float *loss){
    CUDA_GRID_LOOP_X(i,length){
      float y = target[i]-networkout[i];
        errors[i]=y;
        atomicAdd(loss,(y*y)/2);
    }
}