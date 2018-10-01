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

    
    int i = (blockIdx.y*gridDim.x*blockDim.x) +
    (blockIdx.x*blockDim.x) + 
    threadIdx.x;

if (i<length){
     gsum[i]=(beta1*gsum[i]) +((1.0-beta1)*dw[i]);
    float gsumt = 0;
    gsumt = gsum[i]/(1.0- powf(beta1,counter));
     xsum[i]= (beta2*xsum[i])+((1.0 -beta2)*(dw[i]*dw[i]));
    float xsumt =0;
    xsumt = xsum[i]/(1.0 - powf(beta2,counter));
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
   
    dw[i]= dw[i]/batch;
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
void forwardParametricfloat(const int length ,float *x,float *y,  float *alpha){

int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
if (i<length){
    if (x[i]>0.0){
        y[i]=x[i];
    }else{
        y[i]=alpha[i]*x[i];
    }
    
}

}
extern "C" __global__
void backwardParametricfloat(const int length,float *dx,float *dy,  float *alpha, float *dalpha){

    int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;

if (i<length){
    if (dy[i]>0.0){
        dx[i]=1.0;
    }else{
        dx[i]=alpha[i];
        dalpha[i]+=alpha[i]*dy[i];
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
extern "C" __global__
void backwardleakyfloat(const int length,float *dx,float *dy, const float alpha){
int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
if (i<length){
    if (dy[i]>0.0){
        dx[i]=1.0;
    }else{
        dx[i]=alpha;
    }
    
}

}  
