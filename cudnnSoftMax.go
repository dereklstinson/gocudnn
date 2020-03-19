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

//SoftMaxD holds the soft max flags and soft max funcs
type SoftMaxD struct {
	algo C.cudnnSoftmaxAlgorithm_t
	mode C.cudnnSoftmaxMode_t
}

//CreateSoftMaxDescriptor creates a gocudnn softmax descriptor.  It is not part of cudnn, but I wanted to make the library
//A little more stream lined after using it for a while
func CreateSoftMaxDescriptor() *SoftMaxD {
	return &SoftMaxD{}
}

//Set sets the soft max algos.
func (s *SoftMaxD) Set(algo SoftMaxAlgorithm, mode SoftMaxMode) error {

	s.algo = algo.c()
	s.mode = mode.c()
	return nil
}

//Get gets the softmax descriptor values
func (s *SoftMaxD) Get() (algo SoftMaxAlgorithm, mode SoftMaxMode, err error) {
	return SoftMaxAlgorithm(s.algo), SoftMaxMode(s.mode), nil
}
func (s *SoftMaxD) String() string {
	return fmt.Sprintf("SoftMaxD{\n%v,\n%v,\n}\n", s.algo, s.mode)
}

/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//Forward performs forward softmax
//
//Input/Output: y
func (s *SoftMaxD) Forward(
	handle *Handle,

	alpha float64,
	xD *TensorD, x cutil.Mem,
	beta float64,
	yD *TensorD, y cutil.Mem) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnSoftmaxForward(
		handle.x,
		s.algo,
		s.mode,
		a.CPtr(),
		xD.descriptor, x.Ptr(),
		b.CPtr(),
		yD.descriptor, y.Ptr(),
	)).error("SoftMaxForward")
}

//Backward performs the backward softmax
//
//Input/Output: dx
func (s *SoftMaxD) Backward(
	handle *Handle,
	alpha float64,
	yD *TensorD, y cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	beta float64,
	dxD *TensorD, dx cutil.Mem,
) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(dxD.dtype, beta)
	return Status(C.cudnnSoftmaxBackward(
		handle.x,
		s.algo,
		s.mode,
		a.CPtr(),
		yD.descriptor, y.Ptr(),
		dyD.descriptor, dy.Ptr(),
		b.CPtr(),
		dxD.descriptor, dx.Ptr(),
	)).error("SoftMaxBackward")
}

//ForwardUS is like Forward but uses unsafe.Pointer instead of cutil.Mem
func (s *SoftMaxD) ForwardUS(
	handle *Handle,

	alpha float64,
	xD *TensorD, x unsafe.Pointer,
	beta float64,
	yD *TensorD, y unsafe.Pointer) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnSoftmaxForward(
		handle.x,
		s.algo,
		s.mode,
		a.CPtr(),
		xD.descriptor, x,
		b.CPtr(),
		yD.descriptor, y,
	)).error("SoftMaxForward")
}

//BackwardUS is like Backward but uses unsafe.Pointer instead of cutil.Mem
func (s *SoftMaxD) BackwardUS(
	handle *Handle,
	alpha float64,
	yD *TensorD, y unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	beta float64,
	dxD *TensorD, dx unsafe.Pointer,
) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(dxD.dtype, beta)
	return Status(C.cudnnSoftmaxBackward(
		handle.x,
		s.algo,
		s.mode,
		a.CPtr(),
		yD.descriptor, y,
		dyD.descriptor, dy,
		b.CPtr(),
		dxD.descriptor, dx,
	)).error("SoftMaxBackward")
}

//SoftMaxAlgorithm is used for flags and are exposed through its methods
type SoftMaxAlgorithm C.cudnnSoftmaxAlgorithm_t

//Fast changes s to and returns SoftMaxAlgorithm(C.CUDNN_SOFTMAX_FAST)
func (s *SoftMaxAlgorithm) Fast() SoftMaxAlgorithm {
	*s = SoftMaxAlgorithm(C.CUDNN_SOFTMAX_FAST)
	return *s
}

//Accurate changes s to and returns SoftMaxAlgorithm(C.CUDNN_SOFTMAX_ACCURATE)
func (s *SoftMaxAlgorithm) Accurate() SoftMaxAlgorithm {
	*s = SoftMaxAlgorithm(C.CUDNN_SOFTMAX_ACCURATE)
	return *s
}

//Log changes s to and returns SoftMaxAlgorithm(C.CUDNN_SOFTMAX_LOG)
func (s *SoftMaxAlgorithm) Log() SoftMaxAlgorithm {
	*s = SoftMaxAlgorithm(C.CUDNN_SOFTMAX_LOG)
	return *s
}

func (s SoftMaxAlgorithm) c() C.cudnnSoftmaxAlgorithm_t { return C.cudnnSoftmaxAlgorithm_t(s) }
func (s SoftMaxAlgorithm) String() string {
	var x string
	f := s
	switch s {
	case f.Fast():
		x = "Fast"
	case f.Accurate():
		x = "Accurate"
	case f.Log():
		x = "Log"
	default:
		x = "Unsupported Flag"

	}
	return "SoftMaxAlgorithm: " + x
}

//SoftMaxMode is used for softmaxmode flags and are exposed through its methods
type SoftMaxMode C.cudnnSoftmaxMode_t

//Instance changes s to SoftMaxMode(C.CUDNN_SOFTMAX_MODE_INSTANCE) and returns changed value
func (s *SoftMaxMode) Instance() SoftMaxMode {
	*s = SoftMaxMode(C.CUDNN_SOFTMAX_MODE_INSTANCE)
	return *s
}

//Channel changes s to SoftMaxMode(C.CUDNN_SOFTMAX_MODE_CHANNEL) and returns changed value
func (s *SoftMaxMode) Channel() SoftMaxMode { *s = SoftMaxMode(C.CUDNN_SOFTMAX_MODE_CHANNEL); return *s }

func (s SoftMaxMode) c() C.cudnnSoftmaxMode_t { return C.cudnnSoftmaxMode_t(s) }

func (s SoftMaxMode) String() string {
	var x string
	f := s
	switch s{
	case f.Channel():
		x = "Channel"
	case f.Instance():
		x = "Instance"
	default:
		x = "Unsupported Flag"

	}
	return "SoftMaxMode: " + x
}
