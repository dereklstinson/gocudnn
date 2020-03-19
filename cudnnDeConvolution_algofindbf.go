package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//Algo returns an Algorithm Struct
func (c DeConvBwdFiltAlgo) Algo() Algorithm {
	return makealgorithmforbwdfilter(c.c())
}

//GetBackwardFilterAlgorithmMaxCount returns the max number of Algorithm
func (c *DeConvolutionD) getBackwardFilterAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionForwardAlgorithmMaxCount")
	return int32(count), x
}

//FindBackwardFilterAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (c *DeConvolutionD) FindBackwardFilterAlgorithm(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	dwD *FilterD,
) ([]DeConvBwdFiltAlgoPerformance, error) {
	requestedAlgoCount, err := c.getBackwardFilterAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnFindConvolutionBackwardFilterAlgorithm(
		handle.x,
		xD.descriptor,
		dyD.descriptor,
		c.descriptor,
		dwD.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0],
	)).error("FindConvolutionBackwardFilterAlgorithm")

	results := make([]DeConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindBackwardFilterAlgorithmEx finds some algorithms with memory
func (c *DeConvolutionD) FindBackwardFilterAlgorithmEx(
	handle *Handle,
	xD *TensorD, x cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	dwD *FilterD, dw cutil.Mem,
	wspace cutil.Mem, wspacesize uint) ([]DeConvBwdFiltAlgoPerformance, error) {
	reqAlgoCount, err := c.getBackwardFilterAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnFindConvolutionBackwardFilterAlgorithmEx(
		handle.x,
		xD.descriptor, x.Ptr(),
		dyD.descriptor, dy.Ptr(),
		c.descriptor,
		dwD.descriptor, dw.Ptr(),
		C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), C.size_t(wspacesize))).error("FindConvolutionBackwardFilterAlgorithmEx")

	results := make([]DeConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindBackwardFilterAlgorithmExUS is just like FindBackwardFilterAlgorithmEx but uses unsafe.Pointer instead of cutil.Mem
func (c *DeConvolutionD) FindBackwardFilterAlgorithmExUS(
	handle *Handle,
	xD *TensorD, x unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	dwD *FilterD, dw unsafe.Pointer,
	wspace unsafe.Pointer, wspacesize uint) ([]DeConvBwdFiltAlgoPerformance, error) {
	reqAlgoCount, err := c.getBackwardFilterAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnFindConvolutionBackwardFilterAlgorithmEx(
		handle.x,
		xD.descriptor, x,
		dyD.descriptor, dy,
		c.descriptor,
		dwD.descriptor, dw,
		C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace, C.size_t(wspacesize))).error("FindConvolutionBackwardFilterAlgorithmEx")

	results := make([]DeConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetBackwardFilterAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (c *DeConvolutionD) GetBackwardFilterAlgorithmV7(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	dwD *FilterD,
) ([]DeConvBwdFiltAlgoPerformance, error) {
	requestedAlgoCount, err := c.getBackwardFilterAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnGetConvolutionBackwardFilterAlgorithm_v7(
		handle.x,
		xD.descriptor,
		dyD.descriptor,
		c.descriptor,
		dwD.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0])).error("GetConvolutionBackwardFilterAlgorithm_v7")
	results := make([]DeConvBwdFiltAlgoPerformance, int32(actualalgocount))

	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetBackwardFilterAlgorithm gives a good algo with the limits given to it
func (c *DeConvolutionD) GetBackwardFilterAlgorithm(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	dwD *FilterD,
	pref DeConvBwdFilterPref, wsmemlimit uint) (DeConvBwdFiltAlgo, error) {
	var algo C.cudnnConvolutionBwdFilterAlgo_t
	err := Status(C.cudnnGetConvolutionBackwardFilterAlgorithm(
		handle.x,
		xD.descriptor,
		dyD.descriptor,
		c.descriptor,
		dwD.descriptor,
		pref.c(), C.size_t(wsmemlimit), &algo)).error("GetConvolutionBackwardFilterAlgorithm")

	return DeConvBwdFiltAlgo(algo), err
}
func (c DeConvBwdFiltAlgo) String() string {
	var x string
	switch c {
	case DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0):
		x = "DeConvBwdFiltAlgo0"
	case DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1):
		x = "DeConvBwdFiltAlgo1"
	case DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT):
		x = "DeConvBwdFiltAlgoFFT"
	case DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3):
		x = "DeConvBwdFiltAlgo3"
	case DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD):
		x = "DeConvBwdFiltAlgoWinGrad"
	case DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED):
		x = "DeConvBwdFiltAlgoNonFused"
	case DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING):
		x = "DeConvBwdFiltAlgoFFTTiling"
	case DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT):
		x = "DeConvBwdFiltAlgoCount"
	default:
		x = "Unsupported Flag"
	}
	return "DeConvBwdFiltAlgo" + x
}

//DeConvBwdFiltAlgoPerformance is the return struct in the finding algorithm funcs
type DeConvBwdFiltAlgoPerformance struct {
	Algo        DeConvBwdFiltAlgo `json:"algo,omitempty"`
	Status      Status            `json:"status,omitempty"`
	Time        float32           `json:"time,omitempty"`
	Memory      uint              `json:"memory,omitempty"`
	Determinism Determinism       `json:"determinism,omitempty"`
	MathType    MathType          `json:"math_type,omitempty"`
}

func (cb DeConvBwdFiltAlgoPerformance) String() string {
	return fmt.Sprintf("DeConvBwdFiltAlgoPerformance{\n%v,\n%v,\nTime: %v,\nMemory: %v,\n%v,\n%v,\n}\n", cb.Algo, cb.Status, cb.Time, cb.Memory, cb.Determinism, cb.MathType)
}

func convertDeConvBwdFiltAlgoPerformance(input C.cudnnConvolutionBwdFilterAlgoPerf_t) DeConvBwdFiltAlgoPerformance {
	var x DeConvBwdFiltAlgoPerformance
	x.Algo = DeConvBwdFiltAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)
	return x
}
