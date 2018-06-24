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
	var Conv gocudnn.Convolution
	var Activation gocudnn.Activation
	//Group 0
	var datatypeflag gocudnn.DataTypeFlag
	var tffunctionflag gocudnn.TensorFormatFlag
	handle := gocudnn.NewHandle()
	DataTypeFlag := datatypeflag.Float()
	TensorFormatFlag := tffunctionflag.NCHW()
	//Group1
	tens, err := gocudnn.NewTensor4dDescriptor(DataTypeFlag, TensorFormatFlag, gocudnn.Shape(1, 3, 32, 32))
	if err != nil {
		fmt.Println(err)
	}
	//Group 2
	filts, err := gocudnn.NewFilter4dDescriptor(DataTypeFlag, TensorFormatFlag, gocudnn.Shape(3, 3, 5, 5))
	if err != nil {
		fmt.Println(err)
	}

	//Group 3
	var convmodef gocudnn.ConvolutionModeFlag
	ConvMode := convmodef.CrossCorrelation()
	convd, err := Conv.NewConvolution2dDescriptor(ConvMode, DataTypeFlag,
		gocudnn.Pads(1, 1), gocudnn.Strides(1, 1), gocudnn.Dialation(1, 1))
	if err != nil {
		fmt.Println(err)
	}
	dims, err := convd.GetConvolution2dForwardOutputDim(tens, filts)
	if err != nil {
		fmt.Println(err)
	}

	//Group4
	tensout, err := gocudnn.NewTensor4dDescriptor(DataTypeFlag, TensorFormatFlag, dims)
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
	top5performers, err := Conv.Funcs.Fwd.FindConvolutionForwardAlgorithm(handle, tens, filts, convd, tensout, 5)
	if err != nil {
		fmt.Println(err)
	}
	for i := 0; i < len(top5performers); i++ {
		top5performers[i].PrintReadable(i)

	}
	var convfwd gocudnn.ConvolutionFwdPrefFlag
	x, err := Conv.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, tens, filts, convd, tensout, convfwd.NoWorkSpace(), gocudnn.SizeT(0))
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(x)
	var actflag gocudnn.ActivationModeFlag
	var propflag gocudnn.PropagationNANFlag
	activation, err := Activation.NewActivationDescriptor(actflag.Relu(), propflag.NotPropagateNan(), gocudnn.CDouble(4))
	fmt.Println(activation)
}
