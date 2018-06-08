# GoCudnn
Go Bindings for cuDNN 7.1 and Cuda 9.2

cuDNN found at or around https://developer.nvidia.com/cudnn

CUDA Toolkit found at or around https://developer.nvidia.com/cuda-downloads

Golang V1.10 found at or around  https://golang.org/dl/


For the least amount of hassle use Ubuntu 16.04.  This can work on Ubuntu 18.04, but for the time being there is no official Toolkit for 18.04.


Will need to set the environmental variables to something along the lines as below.  
'''
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64

export CUDA_PATH=/usr/local/cuda

export CPATH="$CUDA_PATH/include/"

export CGO_LDFLAGS="$CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so $CUDA_PATH/lib64/libcudnn.so"

export PATH=$PATH:/usr/local/cuda-9.2/bin
'''

version 0.0.1.
Pretty much all of the cudnn.h file has been converted to go.  There are still many things to do like tests.  Some of the tests can't be done until I get some sort of link with the device memory.


1)Needs a way to build memory on device.  
2)Callbacks at this time are not going to be implemented (maybe never)
3)Until this reaches version 1.0 functions are likely to change.  Especially the Flags.  Those are in the middle of being changed to be a little more user friendly/safe.
4)Algorithm will likely be changed.  I will have to see what it is like when I do testing. 
5)I might get rid of the handle methods to keep it in line with how Golang uses contexts. 






Documentation For cudnn can be found at https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html

Take a good look at chapter 2 to get an idea on how the cudnn library works.

The go bindings will be very similar to how cudnn is coded.

A few exceptions though.  Anything that passed a handle is now changed into a method for a handle.  

Anything that has to do with getting or setting up a certain type of descriptor will have that as a method for that descriptor.  

All of the "get" functions will return multiple values (some of them don't right now, but that will change).



#Thinking about doing

I am thinking about returning the Memer interface of the memory that gets updated in handle functions. I think it would make it easier to follow.

Also, I was thinking about adding the clips of the documentation commented out next to the functions. I think it would be easier to code if one is able to see what everything does without having to search for it in the documentation.  






