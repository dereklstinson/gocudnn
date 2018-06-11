package gocudnn

/*

#include <cudnn.h>
*/
import "C"

/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//SoftMaxForward performs forward softmax
func (handle *Handle) SoftMaxForward(
	algo SoftMaxAlgorithm,
	mode SoftMaxMode,
	alpha CScalar,
	xD *TensorD,
	x Memer,
	beta CScalar,
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
	alpha CScalar,
	yD *TensorD,
	y Memer,
	dyD *TensorD,
	dy Memer,
	beta CScalar,
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
