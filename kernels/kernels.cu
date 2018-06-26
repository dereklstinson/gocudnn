

extern "C"__global__
void SoftMaxLossFloat(
     float *input,   // input sfrom the soft max function
     float *output,  // output going back through to the softmax function
     int size,       //
     int classlocation,
     float *loss){
    
    if (threadIdx.x<size){
    if (threadIdx.x==classlocation){
       output[threadIdx.x] =input[threadIdx.x]-1.0
       *loss= -__logf(input[threadIdx.x])
    }else{
        output[threadIdx.x]=input[threadIdx.x]
    }
}
}