# GoCudnn [![Coverage Status](https://coveralls.io/repos/github/dereklstinson/GoCudnn/badge.svg?branch=master)](https://coveralls.io/github/dereklstinson/GoCudnn?branch=master)
<p><img alt="Gopher" title="GoCu" src="GoCu.png" width="500"/></p>

V0.1_75_101 is compiling.  It is cudnn 7.5 w/ cuda 10.1, There might be bugs. Send me a pull request. 

I made a BatchNormalD descriptor and BatchNormDEx descriptor.  You will call this with a "Create" function. and set it like the other descriptors.  

I also made a deconvoltuion descriptor.  It should work.  At least I don't receive any errors when doing the operations.  Deconvolution works like a convolution except backward data is forward and forward is backward data.  
The thing with a deconvolution is that the filter channels will be the output channel, and the filter neurons must match the input channels.

Convolution(Input{N,C,H,W}, Filter{P,C,R,S},Output{N,P,_,_}) 

Deconvolution(Input{N,C,H,W}, Filter{C,Q,R,S}, Output{N,Q,_,_})

## gocu folder

The gocu folder contains interfaces that interconnect the different sub packages.  
To help parallelize your code use the type Worker.
It contains the method work. Where it takes a function at sends it to to be worked 
on a dedicated thread host thread.  Like if you wanted to make a new Context to handle gpu management.

```text
    type GPUcontext struct{
        w *gocu.Worker
        a *crtutil.Allocator
    }
    
    func CreateGPUcontext(dev gocu.Device,s gocu.Streamer)(g *GPUcontext,err error){
        g= new(GPUcontext)
        g.w = new(gocu.Worker)
        err = g.w.Work(func()error{
             g.a = crtutil.CreateAllocator(s)
             return nil
        })
      return err
    }


    func (g *GPUcontext)AllocateMemory(size uint)(c cutil.Mem,err  error){
      err=  g.w.Work(func()error{
            c,err=g.a.AllocateMemory(size)
            return err
        })
    return c,err
    }
    

```
## cudart/crtutil folder

This folder has a ReadWriter in it.  That fulfills the io.Reader and io.Writer interface.

## Beta

I don't forsee any code breaking changes.  Any changes will be new functions.  There will be bugs.  Report them or send me a pull request.

## Some required packages

```text
go get github.com/dereklstinson/half
go get github.com/dereklstinson/cutil
```
If I ever go to modules. These will be placed in there.

## Setup

cuDNN 7.5 found at or around [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

CUDA 10.1 Toolkit found at or around [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Golang V1.13 found at or around [https://golang.org/dl/](https://golang.org/dl/)


Will need to set the environmental variables to something along the lines as below.

```text
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PATH=$PATH:/usr/local/go/bin

```

I would also like to get this to work on windows, also, but I am finding that windows, go, and cuda don't like to mesh together so well, at least not as intuitive as linux, go, and cuda.

## Warnings/Notes

Documentation For cudnn can be found at [https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html)

Take a good look at chapter 2 to get an idea on how the cudnn library works.

The go bindings will be very similar to how cudnn is coded.

A few exceptions though:.  
1. Most descriptors will be handled with methods after they are created.
2. All of the "get" functions will return multiple values.

## A little more on flag handling

Flags are handled through methods.  You must be careful. The methods used with flags will change the flag value. 
If you don't set the flag with a method. It will default with the initialized value (0). That may or may not be a flag option with cudnn or any of the other packages.


## Note on Handles.

CreateHandle() is not thread safe.  Lock the thread using runtime.LockOSThread().  If you get this running on a mac. Then your functions will need to be sent to the main thread.

CreateHandleEX() is designed for multiple gpu use.  It takes a gocu.Worker and any function that takes the handle will pass that function to the worker.  This is still not thread safe, because any
gpu memory that the functions use (for the most part) need to be created on that worker.  Also, before any memory is created the handle needs to be made.  

To parallelize gpus you will need separate handles.  Check out parallel_test.go

## TensorD FilterD NHWC

I found out the for cudnn it always took NCHW as the dims even if the format was NHWC (oof). To me that didn't seem intuitive.  Especially, since it is barely mentioned in the documentation.  I am going to to have it so that if the format is chosen to be NHWC then you need to put the dims as NHWC.  We will see how that works. 


## CUBLAS and CUDA additions

## Other Notes

1. I took errors.go from unixpickle/cuda.  I really didn't want to have to rewrite that error stuff from the cuda runtime api. 

