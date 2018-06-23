package gocudnn

/*
#include <cudnn.h>
*/
import "C"

type Activation struct {
	Ops  ActivationOps
	Flgs ActivationModeFlag
}

//ActivationOperations makes
type ActivationOps struct {
}

//ActivationForward does the forward activation function yrtn is returned and changed.
func (op ActivationOps) ActivationForward(
	handle *Handle,
	aD *ActivationD,
	alpha CScalar,
	xD *TensorD,
	x Memer,
	beta CScalar,
	yD *TensorD,
	yrtn Memer) error {
	return Status(C.cudnnActivationForward(handle.x, aD.descriptor, alpha.CPtr(), xD.descriptor, x.Ptr(), beta.CPtr(), yD.descriptor, yrtn.Ptr())).error("ActivationForward")
}

//ActivationBackward does the activation backward method
func (op ActivationOps) ActivationBackward(
	handle *Handle,
	aD *ActivationD,
	alpha CScalar,
	yD *TensorD,
	y Memer,
	dyD *TensorD,
	dy Memer,
	xD *TensorD,
	x Memer,
	beta CScalar,
	dxD *TensorD,
	dx Memer) error {
	return Status(C.cudnnActivationBackward(handle.x, aD.descriptor, alpha.CPtr(), yD.descriptor, y.Ptr(), dyD.descriptor, dy.Ptr(), xD.descriptor, x.Ptr(), beta.CPtr(), dxD.descriptor, dx.Ptr())).error("ActivationBackward")
}
