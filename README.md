# GoCudnn [![Coverage Status](https://coveralls.io/repos/github/dereklstinson/GoCudnn/badge.svg?branch=master)](https://coveralls.io/github/dereklstinson/GoCudnn?branch=master)
<p><img alt="Gopher" title="GoCu" src="GoCu.png" /></p>



Go Bindings for cuDNN 7.4  using Cuda 10.0 \(Just some Cuda 10.0\) 
V0.1 is cudnn 7.1 with cuda 10.0

 V0.1_75_101 is currently broken while I fix things up

In a few weeks I will add the 7.5 functions, and moving to cuda 10.1, because I think they added some functions
to nvjpeg and npp.  
7.5 is an api breaking update.  Flags are being used completely differently.  Now flags have methods that change value of the type, but also return that changed value.  Also, I am trimming out all non "ND" functions.
This will make less under the hood things that I was adding to get this binding to work. I was using 4d tensors in GoCuNets.  If there is a significant hit to performance when I conform to the new bindings. I will add it back to the bindings.  
That being said. It shouldn't be a much of a change since I was using arrays as inputs to the functions even if it was a 4D functions. 

I got rid of New....Descriptor.  It is now Create...Descriptor.  The descriptors will now need to be set. with (type)Set(....flags). I tried to change every GetDescriptor() to Get(). So, that it will be streamlined.

extra cudnn:

I made a BatchNormalD descriptor and BatchNormDEx descriptor.  You will call this with a "Create" function. and set it like the other descriptors.  



# Back into alpha

I am making the code more like an actual binding. I have seperated the different libraries into their own packages.  
I would have liked to not use cuda.h, but it is needed to run the kernels.  Its that or you would have to make a shared library every time you make a new kernel.  
I am adding some nppi.h to the mix as a subpackage.
I've added nvjpeg, but that was before the 10.1 update.  So, I will upgrade to 10.1 (cuda) and 7.5 (cudnn) and start a new branch.
Any subpackage library bindings I include will most likely only be made to suppliment cudnn.


# SoftMax 

Note on how cudnn uses softmax, because to be honest it isn't entirely clear in the documentation how it works.  
You place what I like to call the answers in the dy part of the function. y is the output of the softmaxforward function.  dx is the gradient going backward.
A simplfied look at how this softmax function works for cudnn is dx = [(alpha * y) + dy] + beta *dx (the bracket is the operation). 
Most texts read the backwards operation to be dx=dy-y.  (dx,dy,and y are vectors, and where y is the one hot state vector (Sorry, Im a computer engineer)))
So make sure beta = 0 and alpha is -1 to get the typical softmax backprop.

Example
```text
func (soft SoftMaxFuncs) SoftMaxBackward(
	handle *Handle,  //cudnn handle
	algo SoftMaxAlgorithm, // a flag
	mode SoftMaxMode,  // a flag
	alpha CScalar,  // Set to -1.0
	yD *TensorD,    // Network Ouput/Answer
	y gocu.Mem,        //   ''
	dyD *TensorD,  // Desired Answer / Label
	dy gocu.Mem,     //   '' 
	beta CScalar,  // Set to 0
	dxD *TensorD,  // Gradient for previous layer
	dx gocu.Mem,      //  ''
)
```


## Setup

cuDNN found at or around [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

CUDA Toolkit found at or around [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Golang V1.10 found at or around [https://golang.org/dl/](https://golang.org/dl/)


Will need to set the environmental variables to something along the lines as below.

```text
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64\
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



## Note on Handles.

This is not thread safe.  You have to lock the host thread in any go routine that you use.  If you get this running on a mac. Then your functions will need to be sent to the main thread.  

## CUBLAS and CUDA additions

There will be no cublas.  I added nvjpeg, and some npp  If DALI ever gets a C api then I will add that too, but I don't see that happening in the near future.


## Other Notes

1. I took errors.go from unixpickle/cuda.  I really didn't want to have to rewrite that error stuff from the cuda runtime api. 

