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
	/*
		fmt.Println("version:", gocudnn.GetVersion())
		fmt.Println("cudaArtVersion:", gocudnn.GetCudaartVersion())

		var handler C.cudnnHandle_t
		stat := (C.cudnnCreate(&handler))
		fmt.Println("Created Handle:", gocudnn.Status(stat).GetErrorString())

		//Group 1
		var tensor1 C.cudnnTensorDescriptor_t
		stat = C.cudnnCreateTensorDescriptor(&tensor1)
		fmt.Println("Created Tensor1:", gocudnn.Status(stat).GetErrorString())

		shape := []C.int{1, 3, 32, 32}
		stat = C.cudnnSetTensor4dDescriptor(tensor1, C.CUDNN_TENSOR_NCHW, C.CUDNN_DATA_FLOAT, shape[0], shape[1], shape[2], shape[3])
		fmt.Println("Set Tensor Descriptor:", gocudnn.Status(stat).GetErrorString())

		//Group 2
		var filter C.cudnnFilterDescriptor_t
		stat = C.cudnnCreateFilterDescriptor(&filter)
		fmt.Println("Created Filter", gocudnn.Status(stat).GetErrorString())

		shape = []C.int{3, 3, 3, 3}
		stat = C.cudnnSetFilter4dDescriptor(filter, C.CUDNN_DATA_FLOAT, C.CUDNN_TENSOR_NCHW, shape[0], shape[1], shape[2], shape[3])
		fmt.Println("Set Filter Descriptor", gocudnn.Status(stat).GetErrorString())

		//Group 3
		var conv C.cudnnConvolutionDescriptor_t
		stat = C.cudnnCreateConvolutionDescriptor(&conv)
		fmt.Println("Created Convolution Filter", gocudnn.Status(stat).GetErrorString())

		//stat = C.cudnnSetConvolutionMathType(conv, C.CUDNN_DEFAULT_MATH)//
		//fmt.Println("Set Convolution Math Type", gocudnn.Status(stat).GetErrorString())
		pad := []C.int{1, 1}
		stride := []C.int{1, 1}
		dila := []C.int{1, 1}
		stat = C.cudnnSetConvolution2dDescriptor(conv, pad[0], pad[1], stride[0], stride[1], dila[0], dila[1], C.CUDNN_CONVOLUTION, C.CUDNN_DATA_FLOAT)
		fmt.Println("Set Convolution Filter", gocudnn.Status(stat).GetErrorString())

		//Group4
		var tensor2 C.cudnnTensorDescriptor_t
		stat = C.cudnnCreateTensorDescriptor(&tensor2)
		fmt.Println("Created tensor 2", gocudnn.Status(stat).GetErrorString())

		shape = []C.int{1, 3, 32, 32}
		stat = C.cudnnSetTensor4dDescriptor(tensor2, C.CUDNN_TENSOR_NCHW, C.CUDNN_DATA_FLOAT, shape[0], shape[1], shape[2], shape[3])
		fmt.Println("Set Tensor 2", gocudnn.Status(stat).GetErrorString())

		//Group5
		var algo C.cudnnConvolutionFwdAlgo_t
		stat = C.cudnnGetConvolutionForwardAlgorithm(handler, tensor1, filter, conv, tensor2, C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, C.size_t(2048000000), &algo)
		fmt.Println("finding output algo", gocudnn.Status(stat).GetErrorString())

		fmt.Println(algo)
		algoamount := C.int(5)
		algoactual := C.int(0)
		algoPerflist := make([]C.cudnnConvolutionFwdAlgoPerf_t, 5)
		stat = C.cudnnFindConvolutionForwardAlgorithm(handler, tensor1, filter, conv, tensor2, algoamount, &algoactual, &algoPerflist[0])
		fmt.Println("finding output algos", gocudnn.Status(stat).GetErrorString())

		for i := C.int(0); i < algoactual; i++ {
			fmt.Println(algoPerflist[i])
		}
	*/
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
	convd, err := gocudnn.NewConvolution2dDescriptor(ConvMode, DataTypeFlag,
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
	top5performers, err := handle.FindConvolutionForwardAlgorithm(tens, filts, convd, tensout, 5)
	if err != nil {
		fmt.Println(err)
	}
	for i := 0; i < len(top5performers); i++ {
		top5performers[i].PrintReadable(i)

	}
	x, err := handle.GetConvolutionForwardAlgorithm(tens, filts, convd, tensout, gocudnn.ConvolutionFwdNoWorkSpace, gocudnn.SizeT(0))
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(x)
	var actflag gocudnn.ActivationModeFlag
	var propflag gocudnn.PropagationNANFlag
	activation, err := gocudnn.NewActivationDescriptor(actflag.Relu(), propflag.NotPropagateNan(), gocudnn.CDouble(4))
	fmt.Println(activation)
}
