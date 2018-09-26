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
               const float beta1,
               const float beta2,
               const float eps,
               const float counter){

    
    int cell = (blockIdx.y*gridDim.x*blockDim.x) +
    (blockIdx.x*blockDim.x) + 
    threadIdx.x;

if (cell<length){
    float ghold=gsum[cell];
    gsum[cell]=beta1*ghold +(1.0-beta1)*dw[cell];
    float gsumt = 0;
    gsumt = gsum[cell]/(1.0- powf(beta1,counter));
    float xhold=xsum[cell];
    xsum[cell]= (beta2*xhold)+((1.0 -beta2)*dw[cell]*dw[cell]);
    float xsumt =0;
    xsumt = xsum[cell]/(1.0 - powf(beta2,counter));
    //float hw = w[cell];
    float wcellhold = w[cell];
    w[cell] = wcellhold -(eps*gsumt)/(sqrtf(xsumt)+eps);  
    __syncthreads();
    dw[cell]=0.0;

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
    int section = blockIdx.x;
    int index = threadIdx.x;
    int cell = section*blockDim.x+index;
    if (cell<length){
    atomicAdd(l2,(w[cell]*w[cell]*decay2)/2.0);
    dw[cell]= (dw[cell]/batch) + w[cell]*decay2;
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
    int section = blockIdx.x;
    int index = threadIdx.x;
    int cell = section*blockDim.x+index;
    float decay = decay1;
    
    if (cell<length){
        
        if (dw[cell]<0){
            decay=-decay;
        }

        atomicAdd(l1,w[cell]*decay); 
        atomicAdd(l2,(w[cell]*w[cell]*decay2)/2.0);
        dw[cell]= (dw[cell]/batch) + (w[cell]*decay2) +decay1;
     }

}  


extern "C"  __global__ 
void simpleadds(
    const int length,
    float *dw, //input and output
    float *w,  //input
    float *gsum, // should be an int but just send it as a float
    float *xsum){

    int section = blockIdx.x;
    int index = threadIdx.x;
    int cell = section*blockDim.x+index;
if (cell<length){
    gsum[cell]=dw[cell]+w[cell];
    xsum[cell]=dw[cell]+w[cell];
}
}

extern "C" __global__
void copyto(const int length,float *dest,float *src){

int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
if (i<length){
    dest[i]=src[i];
}

}
    
  
