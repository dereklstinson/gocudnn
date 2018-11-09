

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



 CUDA_GRID_LOOP_X(cell,length){

    gsum[cell]= gsum[cell]+(dw[cell]*dw[cell]);
    weights[cell]= -(rate*dw[cell])/(sqrtf(gsum[cell])+eps);
    dw[cell]=0.0;


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
/*
extern "C" __global__
void l1regularizationfloat(const int length,
                           float *dw, //input and output
                           float *w,  //input
                           float *l1, //output
                           const float batch, // should be an int but just send it as a float
                           const float decay1){
float decay = decay1;
CUDA_GRID_LOOP_X(i,length){
        
    if (w[i]<0){
         decay=-decay1;
    }else{
        decay=decay1;
    }
        atomicAdd(l1,w[i]*decay);
        dw[i]= (dw[i] +decay/batch);
    }
}

extern "C" __global__
void l2regularizationfloat(
    const int length,
    float *dw, //input and output
    float *w,  //input
    float *l2, //output
    const float batch, // should be an int but just send it as a float
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



*/
extern "C" __global__
void l1l2regularizationfloat(
    const int length,
    float *dw, //input and output
    float *w,  //input needs to ba an array
    float *l1, //output set to zero
    float *l2, //output set to zero
    const float batch, // should be an int but just send it as a float
    const float decay1, //input
    const float decay2){ //input
        float decay = decay1;
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
    /*
extern "C" __global__
void l1l2regularizationfloat(
    const int length,
    float *dw, //input and output
    float *w,  //input needs to ba an array
    float *l1, //output set to zero
    float *l2, //output set to zero
    const float batch, // should be an int but just send it as a float
    const float decay1, //input
    const float decay2){ //input
if (decay1 ==0 && decay2==0){
CUDA_GRID_LOOP_X(i,length){ 
        dw[i]/=batch;
}
}else if (decay1==0){
CUDA_GRID_LOOP_X(i,length){ 
        atomicAdd(l2,(w[i]*w[i]*decay2)/2.0);
        dw[i]= (dw[i] + w[i]*decay2)/batch;
}
}else if(decay2 == 0){
float decay = decay1;
CUDA_GRID_LOOP_X(i,length){
        if (w[i]<0){
             decay=-decay1;
        }else{
            decay=decay1;
        }
            atomicAdd(l1,w[i]*decay);
            dw[i]= (dw[i] +decay/batch);
}
}else {
float decay = decay1;
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
}
*/

extern "C" __global__
void AdvanceThreshRandomReluForward(const int length,
                                    const int batchindex,
                                    const float alpha,
                                    const float beta,
                                    const float *x,
                                          float *y,
                                    const float *coefs,
                                    const float *threshhold,
                                    const int   PropNan){
    int stride= length*batchindex;
   
    CUDA_GRID_LOOP_X(j,length){
        
    if (x[stride+j]>threshhold[j]){
        float value = (alpha*x[stride+j])+beta*y[stride +j];
        if (PropNan>0){
            y[stride+j]= value;
        }else{
            y[stride+j]=value*(!(isnan(value)==0));
        }
       
    }else{
        float value = (x[stride+j]*coefs[j]*alpha)+(beta*y[stride+j]);
        if (PropNan>0){
            y[stride+j]= value;
        }else{
            y[stride+j]=value*(!(isnan(value)==0));
        }  
    }   
}
}
extern "C" __global__
void AdvanceThreshRandomReluBackward(const int length, 
                                        const int batchindex,
                                        const float alpha,
                                        const float beta,        
                                        const float *x,
                                              float *dx,
                                        const float *dy, 
                                        const float *coefs,
                                        const float *threshhold,
                                        const int PropNan){
    int stride= length*batchindex;
   
    CUDA_GRID_LOOP_X(j,length){

    if (x[stride+j]>threshhold[j]){
       float value = (alpha*dy[stride+j])+(beta *dx[stride+j]);
            if (PropNan>0){
                dx[stride+j]=  value;
            }else{
                dx[stride+j]= value*(!(isnan(value)==0));
            }
    }else{
       float value =(alpha*dy[stride+j]*coefs[j])+(beta *dx[stride+j]);
            if (PropNan>0){
                dx[stride+j] =  value;
            }else{
                dx[stride+j]= value*(!(isnan(value)==0));
            }
    }   
}
}

extern "C" __global__
void forwardParametricfloatchannel(const int tx, 
                                   const int ty,
                                   const int tz,
                                   const int batchindex,
                                   const float alpha,
                                   const float beta,
                                   const float *xx,
                                         float *yy,  
                                   const float *coefs,
                                   const int NHWC,
                                   const int PropNan){


   const int stride= tx*ty*tz*batchindex;
   const int ofx = ty*tz;
   const int ofy = tz;
   if (NHWC>0){
        CUDA_GRID_AXIS_LOOP(i,tx,x){

            CUDA_GRID_AXIS_LOOP(j,ty,y){
            
            CUDA_GRID_AXIS_LOOP(k,tz,z){
                int xyindex = stride+(i*ofx)+(j*ofy)+k;
                
                        float value = (alpha *xx[xyindex]*(xx[xyindex]>0))+(alpha *xx[xyindex]*(xx[xyindex]<=0)*coefs[k]) + (beta*yy[xyindex]);
                        if (PropNan>0){
                            yy[xyindex]= value;
                        }else{
                            yy[xyindex]= value*(!(isnan(value)==0));
                                
                        }
                      
                     
                      
                    }
        }
        }
  
}else{
    CUDA_GRID_AXIS_LOOP(i,tx,x){

        CUDA_GRID_AXIS_LOOP(j,ty,y){
        
        CUDA_GRID_AXIS_LOOP(k,tz,z){
            int xyindex = stride+(i*ofx)+(j*ofy)+k;
                
            float value = (alpha *xx[xyindex]*(xx[xyindex]>0))+(alpha *xx[xyindex]*(xx[xyindex]<=0)*coefs[i]) + (beta*yy[xyindex]);
            if (PropNan>0){
                yy[xyindex]= value;
            }else{
                yy[xyindex]= value*(!(isnan(value)==0));
                    
            }
        }
    }
    }


}
                                           

                                           }
//backwardParametricfloat does the backprop of the parametric float

//f(x) = beta*Max(0,x)+alpha*Min(0,x)
extern "C" __global__  
void backwardParametricfloatchannel(const int tx,
                                    const int ty,
                                    const int tz,
                                    const int batchindex,
                                    const float alpha,
                                    const float beta,
                                    const float *xx,
                                          float *dx,
                                    const float *dy,
                                    const float *alphas ,
                                          float *dalphas,
                                    const int NHWC,
                                    const int PropNan){
    int stride= tx*ty*tz*batchindex;
    int ofx = ty*tz;
    int ofy = tz;
if (NHWC>0.0){

CUDA_GRID_AXIS_LOOP(i,tx,x){
    CUDA_GRID_AXIS_LOOP(j,ty,y){  
        CUDA_GRID_AXIS_LOOP(k,tz,z){
            int xyindex = stride+(i*ofx)+(j*ofy)+k;
            float value=(alpha *dy[xyindex]*(xx[xyindex]>0)) + ((xx[xyindex]<=0)*alphas[k]*alpha) +(beta*dx[xyindex]);
            if (PropNan>0){
                dx[xyindex]=value;
                float value2 = dy[xyindex]*xx[xyindex]*(xx[xyindex]<=0);
                atomicAdd(&dalphas[k],value2);
            }else {
                dx[xyindex]=value*(!(isnan(value)==0));
                float value2 = (dy[xyindex]*xx[xyindex]*(xx[xyindex]<=0))*(!(isnan(value)==0));
                atomicAdd(&dalphas[k],value2);
            }
         
          
         
            
        }
    }
}
}else{
    CUDA_GRID_AXIS_LOOP(i,tx,x){
        CUDA_GRID_AXIS_LOOP(j,ty,y){  
        CUDA_GRID_AXIS_LOOP(k,tz,z){
            int xyindex = stride+(i*ofx)+(j*ofy)+k;
            float value=(alpha *dy[xyindex]*(xx[xyindex]>0)) + ((xx[xyindex]<=0)*alphas[i]*alpha) +(beta*dx[xyindex]);
            if (PropNan>0){
                dx[xyindex]=value;
                float value2 = dy[xyindex]*xx[xyindex]*(xx[xyindex]<=0);
                atomicAdd(&dalphas[i],value2);
            }else {
                dx[xyindex]=value*(!(isnan(value)==0));
                float value2 = (dy[xyindex]*xx[xyindex]*(xx[xyindex]<=0))*(!(isnan(value)==0));
                atomicAdd(&dalphas[i],value2);
            }
               
        
            }
        }
        }
}


}



extern "C" __global__
void forwardleakyfloat(const int length,
                       const float alpha,
                       const float beta,
                       const float *x,
                             float *y,
                       const float coef,
                       const int PropNan){
    CUDA_GRID_LOOP_X(i,length){
        if (x[i]>0.0){
            float value = (alpha+x[i])+(beta*y[i]);
            if (PropNan>0){
                y[i]=value;
            }else{
                y[i]=value*(!(isnan(value)==0));
            }
           
         
        }else{
            float value = (x[i]*coef*alpha)+(beta*y[i]);
            if (PropNan>0){
                y[i]=value;
            }else{
                y[i]=value*(!(isnan(value)==0));
            }
          
        }
    }  
    
}

  



extern "C" __global__
void backwardleakyfloat(const int length,
                        const float alpha,
                        const float beta,
                        const float *x, 
                              float *dx,
                        const float *dy, 
                        const float coef,
                        const int PropNan){

CUDA_GRID_LOOP_X(i,length){
    if (x[i]>0.0){
        float value =(dy[i]*alpha)+(beta*dx[i]);
        if (PropNan>0){
            dx[i]=value;
        }else{
            dx[i]=value*(!(isnan(value)==0));
        }
     
    }else{
        float value =(dy[i]*coef*alpha)+(beta*dx[i]);
        if (PropNan>0){
            dx[i]=value;
        }else{
            dx[i]=value*(!(isnan(value)==0));
        }
      
    }
    
}

}  

extern "C" __global__
void MSELoss(const int length ,float *errors,float *target,float *networkout,float *loss){
    loss[0]=0;
    CUDA_GRID_LOOP_X(i,length){
      float y = networkout[i]-target[i];
        errors[i]=y;
        atomicAdd(loss,(y*y)/2);
    }
}