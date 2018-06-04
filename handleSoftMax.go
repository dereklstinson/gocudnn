package gocudnn

/*

#include <cudnn.h>
*/
import "C"

//SoftMaxAlgorithm is used for flags
type SoftMaxAlgorithm C.cudnnSoftmaxAlgorithm_t

//Flags for SoftMaxAlgorithm
const (
	SoftMaxFast     SoftMaxAlgorithm = C.CUDNN_SOFTMAX_FAST     /* straightforward implementation */
	SoftMaxAccurate SoftMaxAlgorithm = C.CUDNN_SOFTMAX_ACCURATE /* subtract max from every point to avoid overflow */
	SoftMaxLog      SoftMaxAlgorithm = C.CUDNN_SOFTMAX_LOG
)

func (sm SoftMaxAlgorithm) c() C.cudnnSoftmaxAlgorithm_t { return C.cudnnSoftmaxAlgorithm_t(sm) }

//SoftMaxMode is used for softmaxmode flags
type SoftMaxMode C.cudnnSoftmaxMode_t

//flags for softmaxmode
const (
	SoftMaxModeInstance SoftMaxMode = C.CUDNN_SOFTMAX_MODE_INSTANCE /* compute the softmax over all C, H, W for each N */
	SoftMaxModeChannel  SoftMaxMode = C.CUDNN_SOFTMAX_MODE_CHANNEL  /* compute the softmax over all C for each H, W, N */
)

func (sm SoftMaxMode) c() C.cudnnSoftmaxMode_t { return C.cudnnSoftmaxMode_t(sm) }

/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//SoftMaxForward performs forward softmax
func (handle *Handle) SoftMaxForward(
	algo SoftMaxAlgorithm,
	mode SoftMaxMode,
	alpha CScaler,
	xD *TensorD,
	x Memer,
	beta CScaler,
	yD *TensorD,
	y Memer) error {
	return Status(C.cudnnSoftmaxForward(
		handle.x,
		algo.c(),
		mode.c(),
		alpha.CPtr(),
		xD.descriptor,
		x.Ptr(),
		beta.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("SoftMaxForward")
}

//SoftMaxBackward performs the backward softmax
func (handle *Handle) SoftMaxBackward(
	algo SoftMaxAlgorithm,
	mode SoftMaxMode,
	alpha CScaler,
	yD *TensorD,
	y Memer,
	dyD *TensorD,
	dy Memer,
	beta CScaler,
	dxD *TensorD,
	dx Memer,
) error {
	return Status(C.cudnnSoftmaxBackward(
		handle.x,
		algo.c(),
		mode.c(),
		alpha.CPtr(),
		yD.descriptor,
		y.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		beta.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
	)).error("SoftMaxBackward")
}
