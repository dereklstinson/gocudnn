# GoCudnn
Go Bindings for cuDNN 7.1 and Cuda 9.2

cuDNN found at or around https://developer.nvidia.com/cudnn

CUDA Toolkit found at or around https://developer.nvidia.com/cuda-downloads

Golang V1.10 found at or around  https://golang.org/dl/


For the least amount of hassle use Ubuntu 16.04.  This can work on Ubuntu 18.04, but for the time being there is no official Toolkit for 18.04.


Will need to set the environmental variables to something along the lines as below.  

export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
export CUDA_PATH=/usr/local/cuda
export CPATH="$CUDA_PATH/include/"
export CGO_LDFLAGS="$CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so $CUDA_PATH/lib64/libcudnn.so"
export PATH=$PATH:/usr/local/cuda-9.2/bin




Sad to say, but this is going to be a save location for the time being.  I will update here on what is working and not.
The end is in sight though, Just finished the LRN portion of the handles.  




Documentation For cudnn can be found at https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html

Take a good look at chapter 2 to get an idea on how the cudnn library works.

The go bindings will be very similar to how cudnn is coded.

A few exceptions though.  Anything that passed a handle is now changed into a method for a handle.  

Anything that has to do with getting or setting up a certain type of descriptor will have that as a method for that descriptor.  

All of the "get" functions will return multiple values (some of them don't right now, but that will change).



#Thinking about doing

I am thinking about returning the Memer interface of the memory that gets updated in handle functions. I think it would make it easier to follow.

Also, I was thinking about adding the clips of the documentation commented out next to the functions. I think it would be easier to code if one is able to see what everything does without having to search for it in the documentation.  

Moving all of the enums/flags for each type of descpriptor to its own package.  I think it would be easier to program on something like vscode making the go extention work better/cleaner on stuff like Completion Lists.  The downside would be that a ton of packages would need to be imported, and I would have to edit a ton of code.  





