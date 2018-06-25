package main

/*
#include <cudnn.h>
*/
import "C"

import (
	"fmt"

	"github.com/dereklstinson/GoCudnn"
)

func main() {
	var Convolution gocudnn.Convolution
	var Tensor gocudnn.Tensor
	Float := Tensor.Flgs.Data.Float()
	NHWC := Tensor.Flgs.Format.NCHW()
	var Filter gocudnn.Filter
	handle := gocudnn.NewHandle()

	//Group1
	tens, err := Tensor.NewTensor4dDescriptor(Float, NHWC, gocudnn.Shape(1, 3, 32, 32))
	if err != nil {
		fmt.Println(err)
	}
	//Group 2
	filts, err := Filter.NewFilter4dDescriptor(Float, NHWC, gocudnn.Shape(3, 3, 5, 5))
	if err != nil {
		fmt.Println(err)
	}

	//Group 3
	ConvMode := Convolution.Flgs.Mode.CrossCorrelation()
	convd, err := Convolution.NewConvolution2dDescriptor(ConvMode, Float,
		gocudnn.Pads(1, 1), gocudnn.Shape(1, 1), gocudnn.Dialation(1, 1))
	if err != nil {
		fmt.Println(err)
	}
	dims, err := convd.GetConvolution2dForwardOutputDim(tens, filts)
	if err != nil {
		fmt.Println(err)
	}

	//Group4
	tensout, err := Tensor.NewTensor4dDescriptor(Float, NHWC, dims)
	if err != nil {
		fmt.Println(err)
	}

	//	querystat, err := handle.QueryRuntimeError(gocudnn.ErrQueryBlocking, nil)
	//	fmt.Println("QueryStatus:", querystat.GetErrorString())
	//stream, err := cudnn.CreateStream()

	//handle.SetStream(&stream)
	/*
		maxCount, err := handle.GetConvolutionForwardAlgorithmMaxCount()
		if err != nil {
			fmt.Println(err)
		}
		fmt.Println("MAXCOUNT", maxCount)
	*/
	//Group 5
	top5performers, err := Convolution.Funcs.Fwd.FindConvolutionForwardAlgorithm(handle, tens, filts, convd, tensout, 5)
	if err != nil {
		fmt.Println(err)
	}
	for i := 0; i < len(top5performers); i++ {
		top5performers[i].PrintReadable(i)

	}
	var convfwd gocudnn.ConvolutionFwdPrefFlag
	x, err := Convolution.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, tens, filts, convd, tensout, convfwd.NoWorkSpace(), gocudnn.SizeT(0))
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(x)
	var actflag gocudnn.ActivationModeFlag
	var propflag gocudnn.PropagationNANFlag
	var Activation gocudnn.Activation
	activation, err := Activation.NewActivationDescriptor(actflag.Relu(), propflag.NotPropagateNan(), gocudnn.CDouble(4))
	fmt.Println(activation)
}
