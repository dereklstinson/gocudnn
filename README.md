# GoCudnn [![Coverage Status](https://coveralls.io/repos/github/dereklstinson/GoCudnn/badge.svg?branch=master)](https://coveralls.io/github/dereklstinson/GoCudnn?branch=master)
<p><img alt="Gopher" title="GoCu" src="GoCu.png" /></p>
Go Bindings for cuDNN 7.1 using Cuda 9.2 \(Just some Cuda 9.2\)

# Now In Alpha Stage
To be honest I haven't tested all the functions.  If there is a bug. Please let me know. Even better make a pull request. I will only be testing the functions that are being used for my thesis.
I don't plan on changing any of the function names.  I don't plan on deleting any functions.  I do plan on adding functions if needed.  Like maybe some cuda_runtime_api module stuff so that more than 2 types
of trainers can be used.  I might add the jpeg image stuff, when that gets out of the beta phase.  
I was able to get MNIST to work, and it is fast. 70,000 images in a few seconds.  WOW!

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
	y Memer,        //   ''
	dyD *TensorD,  // Desired Answer / Label
	dy Memer,     //   '' 
	beta CScalar,  // Set to 0
	dxD *TensorD,  // Gradient for previous layer
	dx Memer,      //  ''
)
```

Currently partially tested files

cudaMemory.go  -Ive been using unified memory mostly.  
cudnnActivation.go - Seems to work
cudnnPooling.go -seems to work too
cudnnSoftMax.go -work in my mnist.  
cudnnOpTensor.go - I know the add works others should work too
cudnnConvolution.go - Worked for the mnist test I had
cudnnStatus.go -This works
cudnnFilter.go -This works
cudnnTensor.go -This works


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

export CGO_LDFLAGS="$CUDA_PATH/lib64/libcudnn.so $CUDA_PATH/lib64/libcublas.so $CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so"
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

Also, I might end up putting the flags into seperate kind of like how I seperated the handle and desc parts of the funcs. Just so it is easier to find. I might change the order too. Instead of leading with desc I might end with it. I don't know which would be easier to read.

## Note on Handles.

I am really stuck on how much I should enforce host thread use on this. So I am not going to enforce anything. That will be up to you. I have a pre-prototype example in the file HostThread.go, but I haven't used it at all yet.

Here is a little info on contexts/handles \(correct me if I am wrong on this too\).  
1. Cuda and cudnn like to have one context/host assigned to one host thread.  
2. When creating a goroutine you will have to use the runtime.LockOSThread\(\) function, and create a Handle on that. Then you should build everthing within that locked thread. 
3. You will not be able to build any memory for that context unless it is built on the thread that is hosting that context. 
4. Multiple Host threads can be used by using streams. I am assuming that can sort of be implimented using goroutines that get locked on a stream, but I haven't played around with it yet.  
5. You cannot share memory from one context to another unless you do some sort of memcopy. 
6. It is best practice to have one context per GPU. As of right now gocudnn doesn't support multiple gpus. It will in the future.



## CUBLAS and CUDA additions

As far as I can tell cudnn doesn't have/support training and loss functions.  I am going to add at least the training part. Loss is a different story. So that being said cuda will have its own context.  Sharing memory between contexts shouldn't be a problem if it is on one graphics card because only one context can run on a device at a time (multiple devices could pose a problem if I am not mistaken).  I will create a subpackage for the kernels that are going to be used for training.  

I started to make a cublas section for fully connected neural networks, but I think I will not do that.  You can create a fully connected neuron, by setting the tensor filter size to be the size of the input dims.  Then don't have padding on the convolution. Make sure the dilation is set to zero  


## Other Notes

1. I took errors.go from unixpickle/cuda.  I really didn't want to have to rewrite that error stuff from the cuda runtime api. 

