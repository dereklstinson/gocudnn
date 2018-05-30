package main

import (
	"fmt"

	"github.com/dereklstinson/cuda/cudnn"
)

func main() {
	handle := cudnn.NewHandle()
	fmt.Println("version:", cudnn.GetVersion())
	fmt.Println("cudaArtVersion:", cudnn.GetCudaartVersion())
	tens, err := cudnn.NewTensor(cudnn.DataTypeFloat, cudnn.TensorFormatNHWC, cudnn.Shape(1, 3, 32, 32), nil)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(tens)
	filts, err := cudnn.CreateFilterDescriptor(cudnn.DataTypeFloat, cudnn.TensorFormatNHWC, cudnn.Shape(3, 3, 5, 5))
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(filts)

	convd, err := cudnn.CreateConvolutionDescriptor(cudnn.Convolution, cudnn.DataTypeFloat,
		cudnn.Pads(1, 1), cudnn.Strides(1, 1), cudnn.Dialation(1, 1))
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(convd)
	dims, err := convd.GetConvolution2dForwardOutputDim(&tens, &filts)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(dims)
	tensout, err := cudnn.NewTensor(cudnn.DataTypeFloat, cudnn.TensorFormatNCHW, dims, nil)
	if err != nil {
		fmt.Println(err)
	}

	if err != nil {
		fmt.Println(err)
	}
	querystat, err := handle.QueryRuntimeError(cudnn.ErrQueryBlocking, nil)
	fmt.Println("QueryStatus:", querystat.GetErrorString())
	//stream, err := cudnn.CreateStream()

	//handle.SetStream(&stream)

	maxCount, err := handle.GetConvolutionForwardAlgorithmMaxCount()
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("MAXCOUNT", maxCount)
	top5performers, err := handle.FindConvolutionForwardAlgorithm(&tens, &filts, &convd, &tensout, 5)
	if err != nil {
		fmt.Println(err)
	}
	for i := 0; i < len(top5performers); i++ {
		top5performers[i].PrintReadable(i)

	}
	x, err := handle.GetConvolutionForwardAlgorithm(&tens, &filts, &convd, &tensout, cudnn.ConvolutionFwdNoWorkSpace, cudnn.SizeT(0))
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(x)

}
