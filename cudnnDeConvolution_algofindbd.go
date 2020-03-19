package gocudnn

//#include <cudnn.h>
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//DeConvBwdDataAlgoPerformance is a new stuct that is made for deconvolution performance
type DeConvBwdDataAlgoPerformance struct {
	Algo        DeConvBwdDataAlgo `json:"algo,omitempty"`
	Status      Status            `json:"status,omitempty"`
	Time        float32           `json:"time,omitempty"`
	Memory      uint              `json:"memory,omitempty"`
	Determinism Determinism       `json:"determinism,omitempty"`
	MathType    MathType          `json:"math_type,omitempty"`
}

func convertDeConvBwdDataAlgoPerformance(input C.cudnnConvolutionFwdAlgoPerf_t) DeConvBwdDataAlgoPerformance {
	var x DeConvBwdDataAlgoPerformance
	x.Algo = DeConvBwdDataAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)
	return x
}

//Algo returns an Algorithm struct
func (c DeConvBwdDataAlgo) Algo() Algorithm {
	return makealgorithmforfwd(c.c())

}

//GetBackwardDataAlgorithmMaxCount returns the max number of Algorithm
func (c *DeConvolutionD) getBackwardDataAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionForwardAlgorithmMaxCount(handle.x, &count)).error("(c *DeConvolutionD) getBackwardDataAlgorithmMaxCount(handle *Handle)")

	return int32(count), x

}

//FindBackwardDataAlgorithm will find the top performing algoriths and return the best algorithms in accending order.
func (c *DeConvolutionD) FindBackwardDataAlgorithm(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	dxD *TensorD,
) ([]DeConvBwdDataAlgoPerformance, error) {
	requestedAlgoCount, err := c.getBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnFindConvolutionForwardAlgorithm(handle.x,
		dyD.descriptor,
		wD.descriptor,
		c.descriptor,
		dxD.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0])).error("(c *DeConvolutionD) FindBackwardDataAlgorithm")

	results := make([]DeConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvBwdDataAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindBackwardDataAlgorithmEx finds some algorithms with memory
func (c *DeConvolutionD) FindBackwardDataAlgorithmEx(
	handle *Handle,
	wD *FilterD, w cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	dxD *TensorD, dx cutil.Mem,
	wspace cutil.Mem, wspacesize uint) ([]DeConvBwdDataAlgoPerformance, error) {
	reqAlgoCount, err := c.getBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	if wspace == nil {
		err = Status(C.cudnnFindConvolutionForwardAlgorithmEx(
			handle.x,
			dyD.descriptor,
			dy.Ptr(),
			wD.descriptor,
			w.Ptr(),
			c.descriptor,
			dxD.descriptor,
			dx.Ptr(),
			C.int(reqAlgoCount),
			&actualalgocount,
			&perfResults[0],
			nil, C.size_t(0))).error("(c *DeConvolutionD) FindBackwardDataAlgorithmEx")

	} else {
		err = Status(C.cudnnFindConvolutionForwardAlgorithmEx(
			handle.x,
			dyD.descriptor,
			dy.Ptr(),
			wD.descriptor,
			w.Ptr(),
			c.descriptor,
			dxD.descriptor,
			dx.Ptr(),
			C.int(reqAlgoCount),
			&actualalgocount,
			&perfResults[0],
			wspace.Ptr(), C.size_t(wspacesize))).error("(c *DeConvolutionD) FindBackwardDataAlgorithmEx")

	}
	results := make([]DeConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvBwdDataAlgoPerformance(perfResults[i])

	}

	return results, err
}

//FindBackwardDataAlgorithmExUS is just like FindBackwardDataAlgorithmEx but uses unsafe.Pointer instead of cutil.Mem
func (c *DeConvolutionD) FindBackwardDataAlgorithmExUS(
	handle *Handle,
	wD *FilterD, w unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	dxD *TensorD, dx unsafe.Pointer,
	wspace unsafe.Pointer, wspacesize uint) ([]DeConvBwdDataAlgoPerformance, error) {
	reqAlgoCount, err := c.getBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnFindConvolutionForwardAlgorithmEx(
		handle.x,
		dyD.descriptor, dy,
		wD.descriptor, w,
		c.descriptor,
		dxD.descriptor, dx,
		C.int(reqAlgoCount), &actualalgocount,
		&perfResults[0],
		wspace, C.size_t(wspacesize))).error("(c *DeConvolutionD) FindBackwardDataAlgorithmExUS")

	results := make([]DeConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvBwdDataAlgoPerformance(perfResults[i])

	}

	return results, err
}

//GetBackwardDataAlgorithm gets the fastest backwards data algorithm with parameters that are passed.
func (c *DeConvolutionD) GetBackwardDataAlgorithm(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	dxD *TensorD,
	pref DeConvBwdDataPref, wspaceSIBlimit uint) (DeConvBwdDataAlgo, error) {
	var algo C.cudnnConvolutionFwdAlgo_t
	err := Status(C.cudnnGetConvolutionForwardAlgorithm(
		handle.x,
		dyD.descriptor,
		wD.descriptor,
		c.descriptor,
		dxD.descriptor,
		pref.c(),
		C.size_t(wspaceSIBlimit), &algo)).error("(c *DeConvolutionD) GetBackwardDataAlgorithm")

	return DeConvBwdDataAlgo(algo), err
}

//GetBackwardDataAlgorithmV7 - This function serves as a heuristic for obtaining the best suited algorithm for the given layer specifications.
//This function will return all algorithms (including (MathType where available) sorted by expected (based on internal heuristic)
//relative performance with fastest being index 0 of perfResults.
func (c *DeConvolutionD) GetBackwardDataAlgorithmV7(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	dxD *TensorD,
) ([]DeConvBwdDataAlgoPerformance, error) {
	requestedAlgoCount, err := c.getBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err = Status(C.cudnnGetConvolutionForwardAlgorithm_v7(
		handle.x,
		dyD.descriptor,
		wD.descriptor,
		c.descriptor,
		dxD.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0])).error("(c *ConvolutionD) GetBackwardDataAlgorithmV7")
	results := make([]DeConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvBwdDataAlgoPerformance(perfResults[i])

	}

	return results, err
}

func (cb DeConvBwdDataAlgoPerformance) String() string {
	return fmt.Sprintf("DeConvBwdDataAlgoPerformance{\n%v,\n%v,\nTime: %v,\nMemory: %v,\n%v,\n%v,\n}\n", cb.Algo, cb.Status, cb.Time, cb.Memory, cb.Determinism, cb.MathType)
}
func (c DeConvBwdDataAlgo) String() string {
	var x string
	switch c {
	case DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM):
		x = "Implicit Gemm"
	case DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM):
		x = "Implicit Precomp Gemm"
	case DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM):
		x = "Gemm"
	case DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT):
		x = "Direct"
	case DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT):
		x = "FFT"
	case DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING):
		x = "FFT Tiling"
	case DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD):
		x = "WinoGrad"
	case DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED):
		x = "WinoGradNonFused"
	case DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT):
		x = "Count"
	default:
		x = "not supported algo --  to be honest ... I don't know how you got here"

	}
	return "DeConvBwdDataAlgo: " + x
}
