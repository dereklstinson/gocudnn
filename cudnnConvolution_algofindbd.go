package gocudnn

/*
#include <cudnn.h>


void MakeAlgorithmforBWDData(cudnnAlgorithm_t *input,cudnnConvolutionBwdDataAlgo_t algo ){
	input->algo.convBwdDataAlgo=algo;
}

*/
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//Algo returns an Algorithm struct
func (c ConvBwdDataAlgo) Algo() Algorithm {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforBWDData(&algorithm, c.c())
	return Algorithm(algorithm)

}

//GetBackwardDataAlgorithmMaxCount returns the max number of Algorithm
func (c *ConvolutionD) getBackwardDataAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionBackwardDataAlgorithmMaxCount")

	return int32(count), x

}

//FindBackwardDataAlgorithm will find the top performing algoriths and return the best algorithms in accending order.
func (c *ConvolutionD) FindBackwardDataAlgorithm(
	handle *Handle,
	w *FilterD,
	dy *TensorD,
	dx *TensorD,
) ([]ConvBwdDataAlgoPerformance, error) {
	requestedAlgoCount, err := c.getBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnFindConvolutionBackwardDataAlgorithm(
		handle.x,
		w.descriptor,
		dy.descriptor,
		c.descriptor,
		dx.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0],
	)).error("FindConvolutionBackwardDataAlgorithm")

	results := make([]ConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdDataAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindBackwardDataAlgorithmEx finds some algorithms with memory
func (c *ConvolutionD) FindBackwardDataAlgorithmEx(
	handle *Handle,
	wD *FilterD, w cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	dxD *TensorD, dx cutil.Mem,
	wspace cutil.Mem, wspacesize uint) ([]ConvBwdDataAlgoPerformance, error) {
	reqAlgoCount, err := c.getBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnFindConvolutionBackwardDataAlgorithmEx(
		handle.x,
		wD.descriptor, w.Ptr(),
		dyD.descriptor, dy.Ptr(),
		c.descriptor,
		dxD.descriptor, dx.Ptr(),
		C.int(reqAlgoCount), &actualalgocount,
		&perfResults[0], wspace.Ptr(), C.size_t(wspacesize))).error("cudnnFindConvolutionBackwardDataAlgorithmEx")

	results := make([]ConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdDataAlgoPerformance(perfResults[i])

	}

	return results, err
}

//FindBackwardDataAlgorithmExUS is just like FindBackwardDataAlgorithmEx but uses unsafe.Pointer instead of cutil.Mem
func (c *ConvolutionD) FindBackwardDataAlgorithmExUS(
	handle *Handle,
	wD *FilterD, w unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	dxD *TensorD, dx unsafe.Pointer,
	wspace unsafe.Pointer, wspacesize uint) ([]ConvBwdDataAlgoPerformance, error) {
	reqAlgoCount, err := c.getBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnFindConvolutionBackwardDataAlgorithmEx(
		handle.x,
		wD.descriptor, w,
		dyD.descriptor, dy,
		c.descriptor,
		dxD.descriptor, dx,
		C.int(reqAlgoCount), &actualalgocount,
		&perfResults[0], wspace, C.size_t(wspacesize))).error("cudnnFindConvolutionBackwardDataAlgorithmEx")

	results := make([]ConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdDataAlgoPerformance(perfResults[i])

	}

	return results, err
}

//GetBackwardDataAlgorithm - This function serves as a heuristic for obtaining the best suited algorithm for (*ConvolutionD)BackwardData() for the given layer specifications.
//Based on the input preference, this function will either return the fastest algorithm or the fastest algorithm within a given memory limit.
//For an exhaustive search for the fastest algorithm, please use  (*ConvolutionD)FindBackwardDataAlgorithm().
//
//Parameters:
//	----
//	handle(input):
//	Handle to a previously created cuDNN context.
//	----
//	---
//	wD(input):
//	Handle to a previously initialized filter descriptor
//	---
//	----
//	dyD(input):
//	Handle to the previously initialized input differential tensor descriptor.
//	----
//	---
//	dxD(input):
//	Handle to the previously initialized output tensor descriptor.
//	---
//	----
//	pref(input):
//	Enumerant to express the preference criteria in terms of memory requirement and speed.
//	----
//	---
//	wspaceSIBlimit(input):
//	It is to specify the maximum amount of GPU memory the user is willing to use as a workspace.
//	This is currently a placeholder and is not used
//	---
//	----
//	returns:
//	ConvBwdDataAlgo and error.
//	----
//
//Possible Error Returns:
//	nil:
//
//	The function launched successfully.
//
//	CUDNN_STATUS_BAD_PARAM:
//
//	At least one of these conditions are met:
//	1) The numbers of feature maps of the input tensor and output tensor differ.
//	2) The DataType of the tensor descriptors or the filter are different.
func (c *ConvolutionD) GetBackwardDataAlgorithm(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	dxD *TensorD,
	pref ConvBwdDataPref, wspaceSIBlimit uint) (ConvBwdDataAlgo, error) {
	var algo C.cudnnConvolutionBwdDataAlgo_t
	err := Status(C.cudnnGetConvolutionBackwardDataAlgorithm(
		handle.x,
		wD.descriptor,
		dyD.descriptor,
		c.descriptor,
		dxD.descriptor,
		pref.c(), (C.size_t)(wspaceSIBlimit), &algo)).error("GetConvolutionBackwardDataAlgorithm")

	return ConvBwdDataAlgo(algo), err
}

//GetBackwardDataAlgorithmV7 - This function serves as a heuristic for obtaining the best suited algorithm for cudnnConvolutionBackwardData for the given layer specifications.
//This function will return all algorithms (including (MathType where available) sorted by expected (based on internal heuristic)
//relative performance with fastest being index 0 of perfResults.
//For an exhaustive search for the fastest algorithm, please use (*ConvolutionD)FindBackwardDataAlgorithm().
func (c *ConvolutionD) GetBackwardDataAlgorithmV7(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	dxD *TensorD,
) ([]ConvBwdDataAlgoPerformance, error) {
	requestedAlgoCount, err := c.getBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnGetConvolutionBackwardDataAlgorithm_v7(
		handle.x,
		wD.descriptor,
		dyD.descriptor,
		c.descriptor,
		dxD.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0])).error("GetConvolutionBackwardDataAlgorithmV7")
	results := make([]ConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdDataAlgoPerformance(perfResults[i])

	}

	return results, err
}

//ConvBwdDataAlgoPerf is used to find the best/fastest algorithms
//type ConvBwdDataAlgoPerformance C.cudnnConvolutionBwdDataAlgoPerf_t

//ConvBwdDataAlgoPerformance is the return struct in the finding algorithm funcs
type ConvBwdDataAlgoPerformance struct {
	Algo        ConvBwdDataAlgo `json:"algo,omitempty"`
	Status      Status          `json:"status,omitempty"`
	Time        float32         `json:"time,omitempty"`
	Memory      uint            `json:"memory,omitempty"`
	Determinism Determinism     `json:"determinism,omitempty"`
	MathType    MathType        `json:"math_type,omitempty"`
}

func convertConvBwdDataAlgoPerformance(input C.cudnnConvolutionBwdDataAlgoPerf_t) ConvBwdDataAlgoPerformance {
	var x ConvBwdDataAlgoPerformance
	x.Algo = ConvBwdDataAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)
	return x
}

func (c ConvBwdDataAlgo) print() {
	switch c {
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0):
		fmt.Println("ConvBwdDataAlgo0")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1):
		fmt.Println("ConvBwdDataAlgo1")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT):
		fmt.Println("ConvBwdDataAlgoFFT")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING):
		fmt.Println("ConvBwdDataAlgoFFTTiling")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD):
		fmt.Println("ConvBwdDataAlgoWinograd")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED):
		fmt.Println("ConvBwdDataAlgoWinoGradNonFused")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT):
		fmt.Println("ConvBwdDataAlgoCount")

	default:
		fmt.Println("Not supported")
	}
}

//Print prints a human readable copy of the algorithm
func (cbd ConvBwdDataAlgoPerformance) Print() {
	fmt.Println("Convolution Backward Data Algorithm Performance")
	fmt.Println("-------------------------------------------------")
	ConvBwdFiltAlgo(cbd.Algo).print()
	fmt.Println("Status:", Status(cbd.Algo).GetErrorString())
	fmt.Println("Time:", cbd.Time)
	fmt.Println("Memory:", cbd.Memory)
	fmt.Println("Determinism:", cbd.Determinism)
	fmt.Println("MathType:", cbd.MathType)
}
