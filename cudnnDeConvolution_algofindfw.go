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

//DeConvFwdAlgoPerformance is a struct that holds the performance of the algorithm
type DeConvFwdAlgoPerformance struct {
	Algo        DeConvFwdAlgo `json:"algo,omitempty"`
	Status      Status        `json:"status,omitempty"`
	Time        float32       `json:"time,omitempty"`
	Memory      uint          `json:"memory,omitempty"`
	Determinism Determinism   `json:"determinism,omitempty"`
	MathType    MathType      `json:"math_type,omitempty"`
}

func (cb DeConvFwdAlgoPerformance) String() string {
	return fmt.Sprintf("DeConvFwdAlgoPerformance{\n%v,\n%v,\nTime: %v,\nMemory: %v,\n%v,\n%v,\n}\n", cb.Algo, cb.Status, cb.Time, cb.Memory, cb.Determinism, cb.MathType)
}

func convertDeConvFwdAlgoPerformance(input C.cudnnConvolutionBwdDataAlgoPerf_t) DeConvFwdAlgoPerformance {
	var x DeConvFwdAlgoPerformance
	x.Algo = DeConvFwdAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)

	return x
}

//Algo returns an Algorithm Struct
func (c DeConvFwdAlgo) Algo() Algorithm {
	return makealgorithmforbwddata(c.c())
}

//GetForwardAlgorithmMaxCount returns the max number of Algorithm
func (c *DeConvolutionD) getForwardAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle.x, &count)).error("(c *ConvolutionD) getForwardAlgorithmMaxCount(handle *Handle)")
		})
	} else {
		err = Status(C.cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle.x, &count)).error("(c *ConvolutionD) getForwardAlgorithmMaxCount(handle *Handle)")
	}

	return int32(count), err

}

//FindForwardAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (c *DeConvolutionD) FindForwardAlgorithm(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	yD *TensorD,
) ([]DeConvFwdAlgoPerformance, error) {
	requestedAlgoCount, err := c.getForwardAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnFindConvolutionBackwardDataAlgorithm(
				handle.x,
				wD.descriptor,
				xD.descriptor,
				c.descriptor,
				yD.descriptor,
				C.int(requestedAlgoCount),
				&actualalgocount,
				&perfResults[0],
			)).error("(c *ConvolutionD) FindBackwardDataAlgorithm")
		})
	} else {
		err = Status(C.cudnnFindConvolutionBackwardDataAlgorithm(
			handle.x,
			wD.descriptor,
			xD.descriptor,
			c.descriptor,
			yD.descriptor,
			C.int(requestedAlgoCount),
			&actualalgocount,
			&perfResults[0],
		)).error("(c *ConvolutionD) FindBackwardDataAlgorithm")
	}
	if err != nil {
		return nil, err
	}

	results := make([]DeConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindForwardAlgorithmEx finds some algorithms with memory
func (c *DeConvolutionD) FindForwardAlgorithmEx(
	handle *Handle,
	xD *TensorD,
	x cutil.Mem,
	wD *FilterD,
	w cutil.Mem,
	yD *TensorD,
	y cutil.Mem,
	wspace cutil.Mem,
	wspaceSIBlimit uint) ([]DeConvFwdAlgoPerformance, error) {
	reqAlgoCount, err := c.getForwardAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}

	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	if handle.w != nil {
		err = handle.w.Work(func() error {
			if wspace == nil {
				return Status(C.cudnnFindConvolutionBackwardDataAlgorithmEx(
					handle.x,
					wD.descriptor, w.Ptr(),
					xD.descriptor, x.Ptr(),
					c.descriptor,
					yD.descriptor, y.Ptr(),
					C.int(reqAlgoCount), &actualalgocount,
					&perfResults[0], nil, C.size_t(0))).error("(c *DeConvolutionD) FindForwardAlgorithmEx")

			}
			return Status(C.cudnnFindConvolutionBackwardDataAlgorithmEx(
				handle.x,
				wD.descriptor, w.Ptr(),
				xD.descriptor, x.Ptr(),
				c.descriptor,
				yD.descriptor, y.Ptr(),
				C.int(reqAlgoCount), &actualalgocount,
				&perfResults[0], wspace.Ptr(), C.size_t(wspaceSIBlimit))).error("(c *DeConvolutionD) FindForwardAlgorithmEx")

		})
	} else {
		if wspace == nil {
			err = Status(C.cudnnFindConvolutionBackwardDataAlgorithmEx(
				handle.x,
				wD.descriptor, w.Ptr(),
				xD.descriptor, x.Ptr(),
				c.descriptor,
				yD.descriptor, y.Ptr(),
				C.int(reqAlgoCount), &actualalgocount,
				&perfResults[0], nil, C.size_t(0))).error("(c *DeConvolutionD) FindForwardAlgorithmEx")

		} else {
			err = Status(C.cudnnFindConvolutionBackwardDataAlgorithmEx(
				handle.x,
				wD.descriptor, w.Ptr(),
				xD.descriptor, x.Ptr(),
				c.descriptor,
				yD.descriptor, y.Ptr(),
				C.int(reqAlgoCount), &actualalgocount,
				&perfResults[0], wspace.Ptr(), C.size_t(wspaceSIBlimit))).error("(c *DeConvolutionD) FindForwardAlgorithmEx")

		}
	}

	if err != nil {
		return nil, err
	}
	results := make([]DeConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindForwardAlgorithmExUS is like FindForwardAlgorithmEx but uses unsafe.Pointer instead of cutil.Mem
func (c *DeConvolutionD) FindForwardAlgorithmExUS(
	handle *Handle,
	xD *TensorD,
	x unsafe.Pointer,
	wD *FilterD,
	w unsafe.Pointer,
	yD *TensorD,
	y unsafe.Pointer,
	wspace unsafe.Pointer,
	wspaceSIBlimit uint) ([]DeConvFwdAlgoPerformance, error) {
	reqAlgoCount, err := c.getForwardAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int

	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnFindConvolutionBackwardDataAlgorithmEx(
				handle.x,
				wD.descriptor, w,
				xD.descriptor, x,
				c.descriptor,
				yD.descriptor, y,
				C.int(reqAlgoCount), &actualalgocount,
				&perfResults[0], wspace, C.size_t(wspaceSIBlimit))).error(" (c *DeConvolutionD) FindForwardAlgorithmExUS")
		})
	} else {
		err = Status(C.cudnnFindConvolutionBackwardDataAlgorithmEx(
			handle.x,
			wD.descriptor, w,
			xD.descriptor, x,
			c.descriptor,
			yD.descriptor, y,
			C.int(reqAlgoCount), &actualalgocount,
			&perfResults[0], wspace, C.size_t(wspaceSIBlimit))).error(" (c *DeConvolutionD) FindForwardAlgorithmExUS")

	}
	if err != nil {
		return nil, err
	}
	results := make([]DeConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetForwardAlgorithm gives a good algo with the limits given to it
func (c *DeConvolutionD) GetForwardAlgorithm(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	yD *TensorD,
	pref DeConvolutionForwardPref,
	wspaceSIBlimit uint) (DeConvFwdAlgo, error) {
	var algo C.cudnnConvolutionBwdDataAlgo_t
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetConvolutionBackwardDataAlgorithm(
				handle.x,
				wD.descriptor,
				xD.descriptor,
				c.descriptor,
				yD.descriptor,
				pref.c(), (C.size_t)(wspaceSIBlimit), &algo)).error("(c *DeConvolutionD) GetForwardAlgorithm")
		})
	} else {
		err = Status(C.cudnnGetConvolutionBackwardDataAlgorithm(
			handle.x,
			wD.descriptor,
			xD.descriptor,
			c.descriptor,
			yD.descriptor,
			pref.c(), (C.size_t)(wspaceSIBlimit), &algo)).error("(c *DeConvolutionD) GetForwardAlgorithm")
	}

	return DeConvFwdAlgo(algo), err
}

//GetForwardAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (c *DeConvolutionD) GetForwardAlgorithmV7(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	yD *TensorD,
) ([]DeConvFwdAlgoPerformance, error) {
	requestedAlgoCount, err := c.getForwardAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetConvolutionBackwardDataAlgorithm_v7(
				handle.x,
				wD.descriptor,
				xD.descriptor,
				c.descriptor,
				yD.descriptor,
				C.int(requestedAlgoCount),
				&actualalgocount,
				&perfResults[0])).error("(c *DeConvolutionD) GetForwardAlgorithmV7")
		})
	} else {
		err = Status(C.cudnnGetConvolutionBackwardDataAlgorithm_v7(
			handle.x,
			wD.descriptor,
			xD.descriptor,
			c.descriptor,
			yD.descriptor,
			C.int(requestedAlgoCount),
			&actualalgocount,
			&perfResults[0])).error("(c *DeConvolutionD) GetForwardAlgorithmV7")
	}

	if err != nil {
		return nil, err
	}
	results := make([]DeConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertDeConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

func (c DeConvFwdAlgo) String() string {

	switch c {
	case DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0):
		return fmt.Sprint("DeConvFwdAlgo0")
	case DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1):
		return fmt.Sprint("DeConvFwdAlgo1")
	case DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT):
		return fmt.Sprint("DeConvFwdAlgoFFT")
	case DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING):
		return fmt.Sprint("DeConvFwdAlgoFFTTiling")
	case DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD):
		return fmt.Sprint("DeConvFwdAlgoWinograd")
	case DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED):
		return fmt.Sprint("DeConvFwdAlgoWinoGradNonFused")
	case DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT):
		return fmt.Sprint("DeConvFwdAlgoCount")

	default:
		return fmt.Sprint("Not supported")
	}
}
