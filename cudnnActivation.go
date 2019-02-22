package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"
)

//Activation is a helper func that is used is activation type processes
type Activation struct {
	Flgs ActivationModeFlag
}

//ActivationD is an opaque struct that holds the description of an activation operation.
type ActivationD struct {
	descriptor C.cudnnActivationDescriptor_t
}

//NewActivationDescriptor creates and sets and returns an activation descriptor in ActivationD and the error
func (new Activation) NewActivationDescriptor(mode ActivationMode, nan PropagationNAN, coef float64) (descriptor *ActivationD, err error) {

	var desc C.cudnnActivationDescriptor_t
	err = Status(C.cudnnCreateActivationDescriptor(&desc)).error("NewActivationDescriptor-create")
	if err != nil {
		return nil, err
	}

	err = Status(C.cudnnSetActivationDescriptor(desc, mode.c(), nan.c(), C.double(coef))).error("NewActivationDescriptor-set")
	if err != nil {
		return nil, err
	}
	descriptor = &ActivationD{
		descriptor: desc,
	}
	if setfinalizer {
		runtime.SetFinalizer(descriptor, destroyactivationdescriptor)
	}
	return descriptor, err
}
func (a *ActivationD) keepsalive() {
	runtime.KeepAlive(a)
}

//GetDescriptor gets the descriptor info for ActivationD
func (a *ActivationD) GetDescriptor() (ActivationMode, PropagationNAN, float64, error) {
	var coef C.double
	var mode C.cudnnActivationMode_t
	var nan C.cudnnNanPropagation_t
	err := Status(C.cudnnGetActivationDescriptor(a.descriptor, &mode, &nan, &coef)).error("GetDescriptor")
	if setkeepalive {
		a.keepsalive()
	}
	return ActivationMode(mode), PropagationNAN(nan), float64(coef), err
}

//DestroyDescriptor destroys the activation descriptor
func (a *ActivationD) DestroyDescriptor() error {
	return destroyactivationdescriptor(a)
}
func destroyactivationdescriptor(a *ActivationD) error {
	return Status(C.cudnnDestroyActivationDescriptor(a.descriptor)).error("DestroyDescriptor")
}

//Forward does the forward activation function yrtn is returned and changed.
func (a *ActivationD) Forward(
	handle *Handle,
	alpha CScalar,
	xD *TensorD,
	x *Malloced,
	beta CScalar,
	yD *TensorD,
	y *Malloced) error {
	if setkeepalive {
		keepsalivebuffer(a, handle, xD, x, yD, y)
	}
	return Status(C.cudnnActivationForward(handle.x, a.descriptor, alpha.CPtr(), xD.descriptor, x.Ptr(), beta.CPtr(), yD.descriptor, y.Ptr())).error("ActivationForward")
}

//Backward does the activation backward method
func (a *ActivationD) Backward(
	handle *Handle,
	alpha CScalar,
	yD *TensorD,
	y *Malloced,
	dyD *TensorD,
	dy *Malloced,
	xD *TensorD,
	x *Malloced,
	beta CScalar,
	dxD *TensorD,
	dx *Malloced) error {
	if setkeepalive {
		keepsalivebuffer(a, handle, xD, x, yD, y, dyD, dy, dxD, dx)
	}
	return Status(C.cudnnActivationBackward(handle.x, a.descriptor, alpha.CPtr(), yD.descriptor, y.Ptr(), dyD.descriptor, dy.Ptr(), xD.descriptor, x.Ptr(), beta.CPtr(), dxD.descriptor, dx.Ptr())).error("ActivationBackward")
}

//ActivationModeFlag is used to "safely" pass flags by the use of methods.
//If it was me I would call it like:
//var ActMode gocudnn.ActivationModeFlag
//Then use the methods like ActMode.Sigmoid() to pass the sigmoid flag.
type ActivationModeFlag struct {
}

//ActivationMode is used for activation discriptor flags use ActivationModeFlag struct to pass flags through its methods
type ActivationMode C.cudnnActivationMode_t

//Sigmoid returnsActivationMode(C.CUDNN_ACTIVATION_SIGMOID)
func (a ActivationModeFlag) Sigmoid() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_SIGMOID)
}

//Relu returns ActivationMode(C.CUDNN_ACTIVATION_RELU)
func (a ActivationModeFlag) Relu() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_RELU)
}

//Tanh returns ActivationMode(C.CUDNN_ACTIVATION_TANH)
func (a ActivationModeFlag) Tanh() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_TANH)
}

//ClippedRelu returns  ActivationMode(C.CUDNN_ACTIVATION_CLIPPED_RELU)
func (a ActivationModeFlag) ClippedRelu() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_CLIPPED_RELU)
}

//Elu returns  ActivationMode(C.CUDNN_ACTIVATION_ELU)
func (a ActivationModeFlag) Elu() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_ELU)
}

//Identity returns  ActivationMode(C.CUDNN_ACTIVATION_IDENTITY)
func (a ActivationModeFlag) Identity() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_IDENTITY)
}
func (a ActivationMode) c() C.cudnnActivationMode_t { return C.cudnnActivationMode_t(a) }
