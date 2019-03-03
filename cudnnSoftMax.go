package gocudnn

/*

#include <cudnn.h>
*/
import "C"
import "github.com/dereklstinson/GoCudnn/gocu"

//SoftMax holds the soft max flags and soft max funcs
type SoftMax struct {
	Flgs  SoftMaxFlags
	Funcs SoftMaxFuncs
}

//SoftMaxFuncs is a nil struct that is used to call SoftMax Functions
type SoftMaxFuncs struct {
}

/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//SoftMaxForward performs forward softmax
func (soft SoftMaxFuncs) SoftMaxForward(
	handle *Handle,
	algo SoftMaxAlgorithm,
	mode SoftMaxMode,
	alpha float64,
	xD *TensorD,
	x gocu.Mem,
	beta float64,
	yD *TensorD,
	y gocu.Mem) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnSoftmaxForward(
		handle.x,
		algo.c(),
		mode.c(),
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		b.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("SoftMaxForward")
}

//SoftMaxBackward performs the backward softmax
func (soft SoftMaxFuncs) SoftMaxBackward(
	handle *Handle,
	algo SoftMaxAlgorithm,
	mode SoftMaxMode,
	alpha float64,
	yD *TensorD,
	y gocu.Mem,
	dyD *TensorD,
	dy gocu.Mem,
	beta float64,
	dxD *TensorD,
	dx gocu.Mem,
) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(dxD.dtype, beta)
	return Status(C.cudnnSoftmaxBackward(
		handle.x,
		algo.c(),
		mode.c(),
		a.CPtr(),
		yD.descriptor,
		y.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		b.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
	)).error("SoftMaxBackward")
}

//SoftMaxFlags holds all the soft max flag
type SoftMaxFlags struct {
	Algo SoftMaxAlgorithmFlag
	Mode SoftMaxModeFlag
}

//SoftMaxAlgorithm is used for flags
type SoftMaxAlgorithm C.cudnnSoftmaxAlgorithm_t

//SoftMaxAlgorithmFlag used to pass SoftMaxAlgorithm flags through methods
type SoftMaxAlgorithmFlag struct {
}

//Fast returns SoftMaxAlgorithm(C.CUDNN_SOFTMAX_FAST)
func (s SoftMaxAlgorithmFlag) Fast() SoftMaxAlgorithm { /* straightforward implementation */
	return SoftMaxAlgorithm(C.CUDNN_SOFTMAX_FAST)
}

//Accurate returns SoftMaxAlgorithm(C.CUDNN_SOFTMAX_ACCURATE)
func (s SoftMaxAlgorithmFlag) Accurate() SoftMaxAlgorithm { /* subtract max from every point to avoid overflow */
	return SoftMaxAlgorithm(C.CUDNN_SOFTMAX_ACCURATE)
}

//Log returns SoftMaxAlgorithm(C.CUDNN_SOFTMAX_LOG)
func (s SoftMaxAlgorithmFlag) Log() SoftMaxAlgorithm {
	return SoftMaxAlgorithm(C.CUDNN_SOFTMAX_LOG)
}

func (sm SoftMaxAlgorithm) c() C.cudnnSoftmaxAlgorithm_t { return C.cudnnSoftmaxAlgorithm_t(sm) }

//SoftMaxMode is used for softmaxmode flags
type SoftMaxMode C.cudnnSoftmaxMode_t

//SoftMaxModeFlag passes SoftMaxMode flags through methods
type SoftMaxModeFlag struct {
}

//Instance returns SoftMaxMode(C.CUDNN_SOFTMAX_MODE_INSTANCE)
func (s SoftMaxModeFlag) Instance() SoftMaxMode { /* subtract max from every point to avoid overflow */
	return SoftMaxMode(C.CUDNN_SOFTMAX_MODE_INSTANCE)
}

//Channel returns SoftMaxMode(C.CUDNN_SOFTMAX_MODE_CHANNEL)
func (s SoftMaxModeFlag) Channel() SoftMaxMode {
	return SoftMaxMode(C.CUDNN_SOFTMAX_MODE_CHANNEL)
}

func (sm SoftMaxMode) c() C.cudnnSoftmaxMode_t { return C.cudnnSoftmaxMode_t(sm) }
