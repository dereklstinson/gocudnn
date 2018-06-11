# GoCudnn
Go Bindings for cuDNN 7.1 and Cuda 9.2

cuDNN found at or around https://developer.nvidia.com/cudnn

CUDA Toolkit found at or around https://developer.nvidia.com/cuda-downloads

Golang V1.10 found at or around  https://golang.org/dl/


For the least amount of hassle use Ubuntu 16.04.  This can work on Ubuntu 18.04, but for the time being there is no official Toolkit for 18.04.


Will need to set the environmental variables to something along the lines as below.  

```
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64

export CUDA_PATH=/usr/local/cuda

export CPATH="$CUDA_PATH/include/"

export CGO_LDFLAGS="$CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so $CUDA_PATH/lib64/libcudnn.so"

export PATH=$PATH:/usr/local/cuda-9.2/bin
```

I would also like to get this to work on windows, also, but I am finding that windows,go,cuda don't like to mesh together so well, at least not as intuitive as linux,go,cuda.

version 0.0.2.
Pretty much all of the cudnn.h file has been converted to go.  There are still many things to do like tests.


1. Needs a way to build memory on device.  
2. allbacks at this time are not going to be implemented (maybe never)
3. Until this reaches version 1.0 functions are likely to change.  
4. Algorithm will likely be changed.  I will have to see what it is like when I do testing. 
5. I might get rid of the handle methods to keep it in line with how Golang uses contexts. 

Documentation For cudnn can be found at https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html

Take a good look at chapter 2 to get an idea on how the cudnn library works.

The go bindings will be very similar to how cudnn is coded.

A few exceptions though:.  
1. Most descriptors will be handled with methods after they are created.
2. Flags are handled by empty structs. Those structs will pass the value of the flag.  
3. All of the "get" functions will return multiple values (some of them don't right now, but that will change).

##A little more on flag handling
So, lets say there is a flag that uses type ConvBwdFilterPref. In order to use those flags you would do something like ConvBwdFilterPref(x), where x would be 0,1, or 2. (I wanted to still be able to have an easy route for automation). Or if you are a human you could use the ConvBwdFilterPrefFlag struct. Lets say you make a ConvBwdFilterPrefFlag called  filtpref (var filtpref gocudnn.ConvBwdFilterPrefFlag) Then you could use the methods for that struct like filtpref.NoWorkspace(), filtpref.PrefFastest(), and filtpref.SpecifyWorkspaceLimit().  The thought process for doing this is, because I love intellisense, and it is really hard to use it when only flags show up on it.

I might end up putting the flags into seperate kind of like how I seperated the handle and desc parts of the funcs.  Just so it is easier to find. I might change the order too.  Instead of leading with desc I might end with it.  I don't know which would be easier to read.  


##Note on Handles.  
I am really stuck on how much I should enforce host thread use on this. So I am not going to enforce anything. That will be up to you.  I have a pre-prototype example in the file HostThread.go, but I haven't used it at all yet.  

Here is a little info on contexts/handles (correct me if I am wrong on this too).  
1. Cuda and cudnn like to have one context assigned to one host thread.  Basically that creates a little world where only that host thread and context can do work together. 
2. When creating a goroutine you will have to use the runtime.LockOSThread() function, and create a Handle on that. Then you can build everthing within that locked thread.
3. You will not be able to build any memory for that context unless it is built on the thread that is hosting that context. 
4. You will not be able have multiple goroutines send info to that context unless you use channels.   
5. You cannot share memory from one context to another unless you do some sort of memcopy.   
6. In order to use multiple GPUs you will have to set that device on a new thread and create a new handle on that thread. (I think I haven't included that    function yet.)

##Thinking about doing

I am thinking about returning the Memer interface of the memory that gets updated in handle functions. I think it would make it easier to follow.

Also, I was thinking about adding the clips of the documentation commented out next to the functions. I think it would be easier to code if one is able to see what everything does without having to search for it in the documentation.  






