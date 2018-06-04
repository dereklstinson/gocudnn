package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//ActivationForward does the forward activation function yrtn is returned and changed.
func (handle *Handle) ActivationForward(
	aD *ActivationD,
	alpha CScaler,
	xD *TensorD,
	x Memer,
	beta CScaler,
	yD *TensorD,
	yrtn Memer) error {
	return Status(C.cudnnActivationForward(handle.x, aD.descriptor, alpha.CPtr(), xD.descriptor, x.Ptr(), beta.CPtr(), yD.descriptor, yrtn.Ptr())).error("ActivationForward")
}

//ActivationBackward does the activation backward method
func (handle *Handle) ActivationBackward(
	aD *ActivationD,
	alpha CScaler,
	yD *TensorD,
	y Memer,
	dyD *TensorD,
	dy Memer,
	xD *TensorD,
	x Memer,
	beta CScaler,
	dxD *TensorD,
	dx Memer) error {
	return Status(C.cudnnActivationBackward(handle.x, aD.descriptor, alpha.CPtr(), yD.descriptor, y.Ptr(), dyD.descriptor, dy.Ptr(), xD.descriptor, x.Ptr(), beta.CPtr(), dxD.descriptor, dx.Ptr())).error("ActivationBackward")
}
