package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//ActivationD is an opaque struct that holds the description of an activation operation.
type ActivationD struct {
	descriptor C.cudnnActivationDescriptor_t
}

//CreateActivationDescriptor creates an activation descriptor
func CreateActivationDescriptor() (*ActivationD, error) {
	desc := new(ActivationD)
	err = Status(C.cudnnCreateActivationDescriptor(desc.descriptor)).error("NewActivationDescriptor-create")
	if setfinalizer {
		runtime.SetFinalizer(desc, destroyactivationdescriptor)
	}
	return desc, err
}

func (a *ActivationD) Set(Mode ActivationMode, nan NANProp, ceof float64) error {
	return Status(C.cudnnSetActivationDescriptor(a.descriptor, mode.c(), nan.c(), C.double(coef))).error("NewActivationDescriptor-set")
}

//Get gets the descriptor descriptors values
func (a *ActivationD) Get() (mode ActivationMode, nan NANProp, coef float64, err error) {
	var coefd C.double
	err = Status(C.cudnnGetActivationDescriptor(a.descriptor, mode.cptr(), nan.cptr(), &coefd)).error("GetDescriptor")
	coef = (float64)(coefd)
	return (mode), (nan), (coef), err
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
	alpha float64,
	xD *TensorD,
	x gocu.Mem,
	beta float64,
	yD *TensorD,
	y gocu.Mem) error {

	a1 := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnActivationForward(handle.x, a.descriptor, a1.CPtr(), xD.descriptor, x.Ptr(), b.CPtr(), yD.descriptor, y.Ptr())).error("ActivationForward")
}

//Backward does the activation backward method
func (a *ActivationD) Backward(
	handle *Handle,
	alpha float64,
	yD *TensorD,
	y gocu.Mem,
	dyD *TensorD,
	dy gocu.Mem,
	xD *TensorD,
	x gocu.Mem,
	beta float64,
	dxD *TensorD,
	dx gocu.Mem) error {
	if setkeepalive {
		keepsalivebuffer(a, handle, xD, x, yD, y, dyD, dy, dxD, dx)
	}
	a1 := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnActivationBackward(handle.x, a.descriptor, a1.CPtr(), yD.descriptor, y.Ptr(), dyD.descriptor, dy.Ptr(), xD.descriptor, x.Ptr(), b.CPtr(), dxD.descriptor, dx.Ptr())).error("ActivationBackward")
}

//ActivationMode is used for activation discriptor flags flags are obtained through type's methods
type ActivationMode C.cudnnActivationMode_t

//Sigmoid sets a to ActivationMode(C.CUDNN_ACTIVATION_SIGMOID)and returns that value.
func (a *ActivationMode) Sigmoid() ActivationMode {
	*a = ActivationMode(C.CUDNN_ACTIVATION_SIGMOID)
	return *a
}

//Relu sets a to ActivationMode(C.CUDNN_ACTIVATION_RELU)and returns that value.
func (a *ActivationMode) Relu() ActivationMode {
	*a = ActivationMode(C.CUDNN_ACTIVATION_RELU)
	return *a
}

//Tanh sets a to ActivationMode(C.CUDNN_ACTIVATION_TANH)and returns that value.
func (a *ActivationMode) Tanh() ActivationMode {
	*a = ActivationMode(C.CUDNN_ACTIVATION_TANH)
	return *a
}

//ClippedRelu sets a to  ActivationMode(C.CUDNN_ACTIVATION_CLIPPED_RELU)and returns that value.
func (a *ActivationMode) ClippedRelu() ActivationMode {
	*a = ActivationMode(C.CUDNN_ACTIVATION_CLIPPED_RELU)
	return *a
}

//Elu sets a to ActivationMode(C.CUDNN_ACTIVATION_ELU) and returns that value.
func (a *ActivationMode) Elu() ActivationMode { *a = ActivationMode(C.CUDNN_ACTIVATION_ELU); return *a }

//Identity returns  ActivationMode(C.CUDNN_ACTIVATION_IDENTITY)
func (a *ActivationMode) Identity() ActivationMode {
	*a = ActivationMode(C.CUDNN_ACTIVATION_SIGMOID)
	return *a
}
func (a ActivationMode) c() C.cudnnActivationMode_t      { return C.cudnnActivationMode_t(a) }
func (a *ActivationMode) cptr() *C.cudnnActivationMode_t { (C.cudnnActivationMode_t)(a) }
