# GoCudnn

Go Bindings for cuDNN 7.1 using Cuda 9.2 \(Just some Cuda 9.2\)

Currently this is pre alpha. Functions can and will change. Almost on a daily basis.

## Setup

cuDNN found at or around [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

CUDA Toolkit found at or around [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Golang V1.10 found at or around [https://golang.org/dl/](https://golang.org/dl/)

For the least amount of hassle use Ubuntu 16.04. This can work on Ubuntu 18.04, but for the time being there is no official Toolkit for 18.04.

Will need to set the environmental variables to something along the lines as below.

```text
export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PATH=$PATH:/usr/local/go/bin


export CUDA_PATH=/usr/local/cuda

export CPATH="$CUDA_PATH/include/"

export CGO_LDFLAGS="$CUDA_PATH/lib64/libcudnn.so $CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so"
```

I would also like to get this to work on windows, also, but I am finding that windows,go,cuda don't like to mesh together so well, at least not as intuitive as linux,go,cuda.

## Warnings/Notes

1. Callbacks at this time are not going to be implemented \(maybe never\)
2. Until this reaches version 1.0 functions are likely to change.  
3. Algorithm will likely be changed.  I will have to see what it is like when I do testing. 
4. I might get rid of the handle methods to keep it in line with how Golang uses contexts. 

Documentation For cudnn can be found at [https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html)

Take a good look at chapter 2 to get an idea on how the cudnn library works.

The go bindings will be very similar to how cudnn is coded.

A few exceptions though:.  
1. Most descriptors will be handled with methods after they are created. 2. Flags are handled by empty structs. Those structs will pass the value of the flag.  
3. All of the "get" functions will return multiple values \(some of them don't right now, but that will change\).

## A little more on flag handling

So, lets say there is a flag that uses type ConvBwdFilterPref. In order to use those flags you would do something like:

```text
ConvBwdFilterPref(x) //Where x = 0,1 or 2.
```

The reason this is still visable is if you wanted to have a front end and wanted to have this stuff easily automated.

Or if you are directly building the networks you could use the ConvBwdFilterPrefFlag struct. Lets say you make a ConvBwdFilterPrefFlag called filtpref

```text
var filtpref gocudnn.ConvBwdFilterPrefFlag
```

Then you could use the methods for that struct like:

```text
filtpref.NoWorkspace()
filtpref.PrefFastest()
filtpref.SpecifyWorkspaceLimit()
```

I think this makes it more intellisense friendly, and a little safer than putting the methods directly on ConvBwdFilterPref.

Also, I might end up putting the flags into seperate kind of like how I seperated the handle and desc parts of the funcs. Just so it is easier to find. I might change the order too. Instead of leading with desc I might end with it. I don't know which would be easier to read.

## Note on Handles.

I am really stuck on how much I should enforce host thread use on this. So I am not going to enforce anything. That will be up to you. I have a pre-prototype example in the file HostThread.go, but I haven't used it at all yet.

Here is a little info on contexts/handles \(correct me if I am wrong on this too\).  
1. Cuda and cudnn like to have one context/host assigned to one host thread.  
2. When creating a goroutine you will have to use the runtime.LockOSThread\(\) function, and create a Handle on that. Then you should build everthing within that locked thread. 3. You will not be able to build any memory for that context unless it is built on the thread that is hosting that context. 4. Multiple Host threads can be used by using streams. I am assuming that can sort of be implimented using goroutines that get locked on a stream, but I haven't played around with it yet.  
5. You cannot share memory from one context to another unless you do some sort of memcopy. 6. It is best practice to have one context per GPU. As of right now gocudnn doesn't support multiple gpus. It will in the future.

## Thinking about doing

I am thinking about returning the Memer interface of the memory that gets updated in handle functions. I think it would make it easier to follow.

Also, I was thinking about adding the clips of the documentation commented out next to the functions. I think it would be easier to code if one is able to see what everything does without having to search for it in the documentation.

## Other Notes

1. I took errors.go from unixpickle/cuda.  I really didn't want to have to rewrite that error stuff from the cuda runtime api. 

