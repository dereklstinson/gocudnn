package gocudnn

/*
#include <cudnn.h>
void MakeAlgorithmforBWDFilter(cudnnAlgorithm_t *input,cudnnConvolutionBwdFilterAlgo_t algo ){
	input->algo.convBwdFilterAlgo=algo;
}

*/
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//ConvBwdFiltAlgoPerformance is the return struct in the finding algorithm funcs
type ConvBwdFiltAlgoPerformance struct {
	Algo        ConvBwdFiltAlgo `json:"algo,omitempty"`
	Status      Status          `json:"status,omitempty"`
	Time        float32         `json:"time,omitempty"`
	Memory      uint            `json:"memory,omitempty"`
	Determinism Determinism     `json:"determinism,omitempty"`
	MathType    MathType        `json:"math_type,omitempty"`
}

func (cb ConvBwdFiltAlgoPerformance) String() string {
	return fmt.Sprintf("ConvBwdFiltAlgoPerformance{\n%v,\n%v,\nTime: %v,\nMemory: %v,\n%v,\n%v,\n}\n", cb.Algo, cb.Status, cb.Time, cb.Memory, cb.Determinism, cb.MathType)
}

func convertConvBwdFiltAlgoPerformance(input C.cudnnConvolutionBwdFilterAlgoPerf_t) ConvBwdFiltAlgoPerformance {
	var x ConvBwdFiltAlgoPerformance
	x.Algo = ConvBwdFiltAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)
	return x
}

//Algo returns an Algorithm Struct
func (c ConvBwdFiltAlgo) Algo() Algorithm {
	return makealgorithmforbwdfilter(c.c())
}
func makealgorithmforbwdfilter(convbwddataalgo C.cudnnConvolutionBwdFilterAlgo_t) Algorithm {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforBWDFilter(&algorithm, convbwddataalgo)
	return Algorithm(algorithm)
}

//GetBackwardFilterAlgorithmMaxCount returns the max number of Algorithm
func (c *ConvolutionD) getBackwardFilterAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle.x, &count)).error("(c *ConvolutionD) getBackwardFilterAlgorithmMaxCount")

		})
	} else {
		err = Status(C.cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle.x, &count)).error("(c *ConvolutionD) getBackwardFilterAlgorithmMaxCount")
	}
	return int32(count), err
}

//FindBackwardFilterAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (c *ConvolutionD) FindBackwardFilterAlgorithm(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	dwD *FilterD,
) ([]ConvBwdFiltAlgoPerformance, error) {
	requestedAlgoCount, err := c.getBackwardFilterAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int

	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnFindConvolutionBackwardFilterAlgorithm(
				handle.x,
				xD.descriptor,
				dyD.descriptor,
				c.descriptor,
				dwD.descriptor,
				C.int(requestedAlgoCount),
				&actualalgocount,
				&perfResults[0],
			)).error("(c *ConvolutionD) FindBackwardFilterAlgorithm")
		})
	} else {
		err = Status(C.cudnnFindConvolutionBackwardFilterAlgorithm(
			handle.x,
			xD.descriptor,
			dyD.descriptor,
			c.descriptor,
			dwD.descriptor,
			C.int(requestedAlgoCount),
			&actualalgocount,
			&perfResults[0],
		)).error("(c *ConvolutionD) FindBackwardFilterAlgorithm")
	}

	results := make([]ConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindBackwardFilterAlgorithmEx finds some algorithms with memory
func (c *ConvolutionD) FindBackwardFilterAlgorithmEx(
	handle *Handle,
	xD *TensorD, x cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	dwD *FilterD, dw cutil.Mem,
	wspace cutil.Mem, wspacesize uint) ([]ConvBwdFiltAlgoPerformance, error) {
	reqAlgoCount, err := c.getBackwardFilterAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnFindConvolutionBackwardFilterAlgorithmEx(
				handle.x,
				xD.descriptor, x.Ptr(),
				dyD.descriptor, dy.Ptr(),
				c.descriptor,
				dwD.descriptor, dw.Ptr(),
				C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), C.size_t(wspacesize))).error("(c *ConvolutionD) FindBackwardFilterAlgorithmEx")
		})
	} else {
		err = Status(C.cudnnFindConvolutionBackwardFilterAlgorithmEx(
			handle.x,
			xD.descriptor, x.Ptr(),
			dyD.descriptor, dy.Ptr(),
			c.descriptor,
			dwD.descriptor, dw.Ptr(),
			C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), C.size_t(wspacesize))).error("(c *ConvolutionD) FindBackwardFilterAlgorithmEx")

	}

	results := make([]ConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindBackwardFilterAlgorithmExUS is just like FindBackwardFilterAlgorithmEx but uses unsafe.Pointer instead of cutil.Mem
func (c *ConvolutionD) FindBackwardFilterAlgorithmExUS(
	handle *Handle,
	xD *TensorD, x unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	dwD *FilterD, dw unsafe.Pointer,
	wspace unsafe.Pointer, wspacesize uint) ([]ConvBwdFiltAlgoPerformance, error) {
	reqAlgoCount, err := c.getBackwardFilterAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnFindConvolutionBackwardFilterAlgorithmEx(
				handle.x,
				xD.descriptor, x,
				dyD.descriptor, dy,
				c.descriptor,
				dwD.descriptor, dw,
				C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace, C.size_t(wspacesize))).error("(c *ConvolutionD) FindBackwardFilterAlgorithmExUS")
		})
	} else {
		err = Status(C.cudnnFindConvolutionBackwardFilterAlgorithmEx(
			handle.x,
			xD.descriptor, x,
			dyD.descriptor, dy,
			c.descriptor,
			dwD.descriptor, dw,
			C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace, C.size_t(wspacesize))).error("(c *ConvolutionD) FindBackwardFilterAlgorithmExUS")

	}

	results := make([]ConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetBackwardFilterAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (c *ConvolutionD) GetBackwardFilterAlgorithmV7(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	dwD *FilterD,
) ([]ConvBwdFiltAlgoPerformance, error) {
	requestedAlgoCount, err := c.getBackwardFilterAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetConvolutionBackwardFilterAlgorithm_v7(
				handle.x,
				xD.descriptor,
				dyD.descriptor,
				c.descriptor,
				dwD.descriptor,
				C.int(requestedAlgoCount),
				&actualalgocount,
				&perfResults[0])).error("(c *ConvolutionD) GetBackwardFilterAlgorithmV7")
		})
	} else {
		err = Status(C.cudnnGetConvolutionBackwardFilterAlgorithm_v7(
			handle.x,
			xD.descriptor,
			dyD.descriptor,
			c.descriptor,
			dwD.descriptor,
			C.int(requestedAlgoCount),
			&actualalgocount,
			&perfResults[0])).error("(c *ConvolutionD) GetBackwardFilterAlgorithmV7")
	}

	results := make([]ConvBwdFiltAlgoPerformance, int32(actualalgocount))

	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetBackwardFilterAlgorithm gives a good algo with the limits given to it
func (c *ConvolutionD) GetBackwardFilterAlgorithm(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	dwD *FilterD,
	pref ConvBwdFilterPref, wsmemlimit uint) (ConvBwdFiltAlgo, error) {
	var algo C.cudnnConvolutionBwdFilterAlgo_t
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetConvolutionBackwardFilterAlgorithm(
				handle.x,
				xD.descriptor,
				dyD.descriptor,
				c.descriptor,
				dwD.descriptor,
				pref.c(), C.size_t(wsmemlimit), &algo)).error("(c *ConvolutionD) GetBackwardFilterAlgorithm")
		})
	} else {
		err = Status(C.cudnnGetConvolutionBackwardFilterAlgorithm(
			handle.x,
			xD.descriptor,
			dyD.descriptor,
			c.descriptor,
			dwD.descriptor,
			pref.c(), C.size_t(wsmemlimit), &algo)).error("(c *ConvolutionD) GetBackwardFilterAlgorithm")

	}

	return ConvBwdFiltAlgo(algo), err
}

func (c ConvBwdFiltAlgo) String() string {
	var x string
	switch c {
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0):
		x = "ConvBwdFiltAlgo0"
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1):
		x = "ConvBwdFiltAlgo1"
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT):
		x = "ConvBwdFiltAlgoFFT"
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3):
		x = "ConvBwdFiltAlgo3"
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD):
		x = "ConvBwdFiltAlgoWinGrad"
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED):
		x = "ConvBwdFiltAlgoNonFused"
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING):
		x = "ConvBwdFiltAlgoFFTTiling"
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT):
		x = "ConvBwdFiltAlgoCount"
	default:
		x = "Unsupported Flag"
	}
	return "ConvBwdFiltAlgo: " + x
}
