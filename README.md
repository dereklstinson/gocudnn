# GoCudnn [![Coverage Status](https://coveralls.io/repos/github/dereklstinson/GoCudnn/badge.svg?branch=master)](https://coveralls.io/github/dereklstinson/GoCudnn?branch=master)
<p><img alt="Gopher" title="GoCu" src="GoCu.png" width="500"/></p>

V0.1_75_101 is compiling.  It is cudnn 7.5 w/ cuda 10.1, There might be bugs. Let me know.  

This is an api breaking update.  Flags are being used completely differently.  Now flags have methods that change value of the type, but also return that changed value.  Also, I am trimming out all non "ND" functions.
This will make less under the hood things that I was adding to get this binding to work. I noticed no performance hit when everything was switched to ND.

That being said. It shouldn't be a much of a change since I was using arrays as inputs to the functions even if it was a 4D functions. 

I got rid of New....Descriptor.  It is now Create...Descriptor.  The descriptors will now need to be set. with (type)Set(....flags). I tried to change every GetDescriptor() to Get(). So, that it will be streamlined.

Got got rid (or getting rid) of the empty structs used to handle the operations.  I moved the methods of those empty structs (or will move them if I find any more) to the appropriate descriptor. Even if the method didn't use the descriptor.
extra cudnn:

I made a BatchNormalD descriptor and BatchNormDEx descriptor.  You will call this with a "Create" function. and set it like the other descriptors.  

I also made a deconvoltuion descriptor.  It hasn't been tested yet.  Deconvolution works like a convolution except backward data is forward and forward is backward data.  
I tried giving giving them there own algo finders and all.  The thing with a deconvolution is that the filter channels will be the output channel, and the filter neurons must match the input channels.


## gocu folder

The gocu folder contains interfaces that interconnect the different sub packages.  To help parallelize your code use the type Worker.  It contains the method work. Where it takes a function at sends it to to be worked on a dedicated thread host thread.  Like if you wanted to make a new Context to handle gpu management.

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

## Back into alpha

I am making the code more like an actual binding. I have separated the different libraries into their own packages.  
I would have liked to not use cuda.h, but it is needed to run the kernels.  Its that or you would have to make a shared library every time you make a new kernel.  
I am adding some NPP library stuff where it would be appropriate with gocudnn.
I've added nvjpeg, but that was before the 10.1 update.  So, I will upgrade to 10.1 (cuda) and 7.5 (cudnn) and start a new branch.
Any subpackage library bindings I include will most likely only be made to supplement cudnn.

## Some required packages

```text
go get github.com/dereklstinson/half
go get github.com/dereklstinson/cutil
```
If I ever go to modules. These will be placed in there.

## Setup

cuDNN 7.5.0 found at or around [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

CUDA 10.1 Toolkit found at or around [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Golang V1.12 found at or around [https://golang.org/dl/](https://golang.org/dl/)


Will need to set the environmental variables to something along the lines as below.

```text
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PATH=$PATH:/usr/local/go/bin

```

I would also like to get this to work on windows, also, but I am finding that windows, go, and cuda don't like to mesh together so well, at least not as intuitive as linux, go, and cuda.

## Warnings/Notes

1. Callbacks at this time are not going to be implemented \(maybe never\)


Documentation For cudnn can be found at [https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html)

Take a good look at chapter 2 to get an idea on how the cudnn library works.

The go bindings will be very similar to how cudnn is coded.

A few exceptions though:.  
1. Most descriptors will be handled with methods after they are created.
2. All of the "get" functions will return multiple values \(some of them don't right now, but that will change\).

## A little more on flag handling

Flags are handled through methods.  You must be careful. The methods used with flags will change the flag value.  
If you don't set the flag with a method. It will default with the initialized value (0). That may or may not be a flag option with cudnn or any of the other packages.


## Note on Handles.

This is not thread safe.  You have to lock the host thread in any go routine that you use.  If you get this running on a mac. Then your functions will need to be sent to the main thread.  

## CUBLAS and CUDA additions

There will be no cublas. Fully connected neural networks can be made when the weights feature map dims == the input feature map dims.
I added nvjpeg, and some npp.  If DALI ever gets a C api then I will add that too, but I don't see that happening in the near future.


## Other Notes

1. I took errors.go from unixpickle/cuda.  I really didn't want to have to rewrite that error stuff from the cuda runtime api. 

