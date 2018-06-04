package gocudnn

/*

#include <cudnn.h>
*/
import "C"

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//PoolingForward does the poolingForward operation
func (handle *Handle) PoolingForward(
	p *PoolingD,
	alpha CScaler,
	xD *TensorD,
	x Memer,
	beta CScaler,
	yD *TensorD,
	y Memer,
) error {
	return Status(C.cudnnPoolingForward(
		handle.x,
		p.descriptor,
		alpha.CPtr(),
		xD.descriptor,
		x.Ptr(),
		beta.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("PoolingForward")
}

//PoolingBackward does the backward pooling operation
func (handle *Handle) PoolingBackward(
	p *PoolingD,
	alpha CScaler,
	yD *TensorD,
	y Memer,
	dyD *TensorD,
	dy Memer,
	xD *TensorD,
	x Memer,
	beta CScaler,
	dxD *TensorD,
	dx Memer,
) error {
	return Status(C.cudnnPoolingBackward(
		handle.x,
		p.descriptor,
		alpha.CPtr(),
		yD.descriptor,
		y.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		xD.descriptor,
		x.Ptr(),
		beta.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
	)).error("PoolingBackward")
}
