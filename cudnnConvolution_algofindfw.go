package gocudnn

/*
#include <cudnn.h>


void MakeAlgorithmforFWD(cudnnAlgorithm_t *input,cudnnConvolutionFwdAlgo_t algo ){
	input->algo.convFwdAlgo=algo;
}
*/
import "C"
import (
	"fmt"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//Algo returns an Algorithm Struct
func (c ConvFwdAlgo) Algo() Algorithm {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforFWD(&algorithm, c.c())
	return Algorithm(algorithm)
}

//GetForwardAlgorithmMaxCount returns the max number of Algorithm
func (c *ConvolutionD) getForwardAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionForwardAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionForwardAlgorithmMaxCount")

	return int32(count), x

}

//FindForwardAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (c *ConvolutionD) FindForwardAlgorithm(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	yD *TensorD,
) ([]ConvFwdAlgoPerformance, error) {
	requestedAlgoCount, err := c.getForwardAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnFindConvolutionForwardAlgorithm(handle.x, xD.descriptor, wD.descriptor, c.descriptor, yD.descriptor, C.int(requestedAlgoCount), &actualalgocount, &perfResults[0])).error("FindConvolutionForwardAlgorithm")

	results := make([]ConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindForwardAlgorithmEx finds some algorithms with memory
func (c *ConvolutionD) FindForwardAlgorithmEx(
	handle *Handle,
	xD *TensorD,
	x gocu.Mem,
	wD *FilterD,
	w gocu.Mem,
	yD *TensorD,
	y gocu.Mem,
	wspace gocu.Mem,
	wspacesize uint) ([]ConvFwdAlgoPerformance, error) {
	reqAlgoCount, err := c.getForwardAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int

	if wspace == nil {
		err = Status(C.cudnnFindConvolutionForwardAlgorithmEx(handle.x, xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(), c.descriptor, yD.descriptor, y.Ptr(), C.int(reqAlgoCount), &actualalgocount, &perfResults[0], nil, C.size_t(0))).error("FindConvolutionForwardAlgorithmEx")

	} else {
		err = Status(C.cudnnFindConvolutionForwardAlgorithmEx(handle.x, xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(), c.descriptor, yD.descriptor, y.Ptr(), C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), C.size_t(wspacesize))).error("FindConvolutionForwardAlgorithmEx")

	}

	results := make([]ConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetForwardAlgorithm gives a good algo with the limits given to it
func (c *ConvolutionD) GetForwardAlgorithm(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	yD *TensorD,
	pref ConvolutionForwardPref,
	wsmemlimit uint) (ConvFwdAlgo, error) {
	var algo C.cudnnConvolutionFwdAlgo_t
	err := Status(C.cudnnGetConvolutionForwardAlgorithm(handle.x, xD.descriptor, wD.descriptor, c.descriptor, yD.descriptor, pref.c(), C.size_t(wsmemlimit), &algo)).error("GetConvolutionForwardAlgorithm")

	return ConvFwdAlgo(algo), err
}

//GetForwardAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (c *ConvolutionD) GetForwardAlgorithmV7(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	yD *TensorD,
) ([]ConvFwdAlgoPerformance, error) {
	requestedAlgoCount, err := c.getForwardAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnGetConvolutionForwardAlgorithm_v7(handle.x, xD.descriptor, wD.descriptor, c.descriptor, yD.descriptor, C.int(requestedAlgoCount), &actualalgocount, &perfResults[0])).error("FindConvolutionForwardAlgorithm")

	results := make([]ConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

func (c ConvFwdAlgo) toString() string {
	var x string
	switch c {
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM):
		x = "Implicit Gemm"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM):
		x = "Implicit Precomp Gemm"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM):
		x = "Gemm"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT):
		x = "Direct"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT):
		x = "FFT"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING):
		x = "FFT Tiling"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD):
		x = "WinoGrad"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED):
		x = "WinoGradNonFused"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT):
		x = "Count"
	default:
		x = "not supported algo --  to be honest ... I don't know how you got here"

	}
	return x
}

//Print prints a human readable copy of the algorithm
func (algoPerf ConvFwdAlgoPerformance) Print() {
	fmt.Println("Convolution Forward Filter Algorithm Performance")
	fmt.Println("-------------------------------------------------")
	ConvBwdFiltAlgo(algoPerf.Algo).print()
	fmt.Println("Status:", Status(algoPerf.Algo).GetErrorString())
	fmt.Println("Time:", algoPerf.Time)
	fmt.Println("Memory:", algoPerf.Memory)
	fmt.Println("Determinism:", algoPerf.Determinism)
	fmt.Println("MathType:", algoPerf.MathType)
}

//ConvFwdAlgoPerformance is a struct that holds the performance of the algorithm
type ConvFwdAlgoPerformance struct {
	Algo        ConvFwdAlgo `json:"algo,omitempty"`
	Status      Status      `json:"status,omitempty"`
	Time        float32     `json:"time,omitempty"`
	Memory      uint        `json:"memory,omitempty"`
	Determinism Determinism `json:"determinism,omitempty"`
	MathType    MathType    `json:"math_type,omitempty"`
}

func convertConvFwdAlgoPerformance(input C.cudnnConvolutionFwdAlgoPerf_t) ConvFwdAlgoPerformance {
	var x ConvFwdAlgoPerformance
	x.Algo = ConvFwdAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)

	return x
}
