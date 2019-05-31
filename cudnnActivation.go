package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//ActivationD is an opaque struct that holds the description of an activation operation.
type ActivationD struct {
	descriptor C.cudnnActivationDescriptor_t
	gogc       bool
}

//CreateActivationDescriptor creates an activation descriptor
func CreateActivationDescriptor() (*ActivationD, error) {
	desc := new(ActivationD)

	err := Status(C.cudnnCreateActivationDescriptor(&desc.descriptor)).error("NewActivationDescriptor-create")
	if setfinalizer {
		desc.gogc = true
		runtime.SetFinalizer(desc, destroyactivationdescriptor)
	}
	return desc, err
}

//Set sets the activation operation according to the settings passed
func (a *ActivationD) Set(mode ActivationMode, nan NANProp, coef float64) error {
	return Status(C.cudnnSetActivationDescriptor(a.descriptor, mode.c(), nan.c(), C.double(coef))).error("NewActivationDescriptor-set")
}

//Get gets the descriptor descriptors values
func (a *ActivationD) Get() (mode ActivationMode, nan NANProp, coef float64, err error) {
	var coefd C.double
	err = Status(C.cudnnGetActivationDescriptor(a.descriptor, mode.cptr(), nan.cptr(), &coefd)).error("GetDescriptor")
	coef = (float64)(coefd)
	return (mode), (nan), (coef), err
}

//Destroy destroys the activation descriptor if GC is not set. if not set method will only return nil
//Currently GC is always set with no way of turning it off
func (a *ActivationD) Destroy() error {
	if setfinalizer || a.gogc {
		return nil
	}
	return destroyactivationdescriptor(a)
}
func destroyactivationdescriptor(a *ActivationD) error {
	return Status(C.cudnnDestroyActivationDescriptor(a.descriptor)).error("DestroyDescriptor")
}

//Forward does the forward activation function
//
//From deep learning sdk documentation (slightly modified for gocudnn):
//
//This routine applies a specified neuron activation function element-wise over each input value.
//
//Note: In-place operation is allowed for this routine; i.e., x and y cutil.Mem may be equal.
//However, this requires xD and yD descriptors to be identical
//(particularly, the strides of the input and output must match for in-place operation to be allowed).
//
//Note: All tensor formats are supported for 4 and 5 dimensions, however best performance is obtained
//when the strides of xD and yD are equal and HW-packed. For more than 5 dimensions
//the tensors must have their spatial dimensions packed.
//
//Parameters:
//
//	---
//	handle(input):
//
//	previously created Handle
//	---
//	----
//	alpha, beta(input):
//
//	Pointers to scaling factors (in host memory) used to blend the computation result with prior
//	value in the output layer as follows: dstValue = alpha[0]*result + beta[0]*priorDstValue.
//	----
//	---
//	xD(input):
//
//	Handle to the previously initialized input tensor descriptor.
//	---
//	----
//	x(input):
//
//	Data pointer to GPU memory associated with the tensor descriptor xD.
//
//	----
//	---
//	yD(input):
//
//	Handle to the previously initialized output tensor descriptor.
//	---
//	----
//	y(output):
//
//	Data pointer to GPU memory associated with the output tensor descriptor yDesc.
//	----
//
//Possible Error Returns
//
//	nil:
//
//	The function launched successfully.
//
//	CUDNN_STATUS_NOT_SUPPORTED:
//
//	The function does not support the provided configuration.
//
//	CUDNN_STATUS_BAD_PARAM:
//
//	At least one of the following conditions are met:
//
//	1)The parameter mode has an invalid enumerant value.
//	2)The dimensions n,c,h,w of the input tensor and output tensors differ.
//	3)The datatype of the input tensor and output tensors differs.
//	4)The strides nStride,cStride,hStride,wStride of the input tensor and output tensors differ and in-place operation is used (i.e., x and y pointers are equal).
//
//	CUDNN_STATUS_EXECUTION_FAILED:
//
//	The function failed to launch on the GPU.
//
func (a *ActivationD) Forward(
	handle *Handle,
	alpha float64,
	xD *TensorD, x cutil.Mem,
	beta float64,
	yD *TensorD, y cutil.Mem) error {
	a1 := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnActivationForward(handle.x, a.descriptor, a1.CPtr(), xD.descriptor, x.Ptr(), b.CPtr(), yD.descriptor, y.Ptr())).error("ActivationForward")
}

//ForwardUS is just like Forward but it takes unsafe.Pointers instead of cutil.Mem
func (a *ActivationD) ForwardUS(
	handle *Handle,
	alpha float64,
	xD *TensorD, x unsafe.Pointer,
	beta float64,
	yD *TensorD, y unsafe.Pointer) error {
	a1 := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnActivationForward(handle.x, a.descriptor, a1.CPtr(), xD.descriptor, x, b.CPtr(), yD.descriptor, y)).error("ActivationForward")
}

//Backward does the activation backward method
//
//From deep learning sdk documentation (slightly modified for gocudnn):
//
//This routine computes the gradient of a neuron activation function.
//
//Note: In-place operation is allowed for this routine; i.e., dx and dy cutil.Mem may be equal.
//However, this requires dxD and dyD descriptors to be identical
//(particularly, the strides of the input and output must match for in-place operation to be allowed).
//
//Note: All tensor formats are supported for 4 and 5 dimensions, however best performance is obtained
//when the strides of dxD and dyD are equal and HW-packed. For more than 5 dimensions
//the tensors must have their spatial dimensions packed.
//
//Parameters:
//
//	---
//	handle(input):
//
//	previously created Handle
//	---
//	----
//	alpha, beta(input):
//
//	Pointers to scaling factors (in host memory) used to blend the computation result with prior
//	value in the output layer as follows: dstValue = alpha[0]*result + beta[0]*priorDstValue.
//	----
//	---
//	xD(input):
//
//	Handle to the previously initialized input tensor descriptor.
//	---
//	----
//	x(input):
//
//	Data pointer to GPU memory associated with the tensor descriptor xD.
//	----
//	---
//	dxD(input):
//
//	Handle to the previously initialized input tensor descriptor.
//	---
//	----
//	dx(output):
//
//	Data pointer to GPU memory associated with the tensor descriptor dxD.
//	----
//	---
//	yD(input):
//
//	Handle to the previously initialized output tensor descriptor.
//	---
//	----
//	y(input):
//
//	Data pointer to GPU memory associated with the output tensor descriptor yD.
//	----
//	---
//	dyD(input):
//
//	Handle to the previously initialized output tensor descriptor.
//	---
//	----
//	dy(input):
//
//	Data pointer to GPU memory associated with the output tensor descriptor dyD.
//	----
//
//Possible Error Returns
//
//	nil:
//
//	The function launched successfully.
//
//	CUDNN_STATUS_NOT_SUPPORTED:
//
//	1) The dimensions n,c,h,w of the input tensor and output tensors differ.
//  2) The datatype of the input tensor and output tensors differs.
//  3) The strides nStride, cStride, hStride, wStride of the input tensor and the input differential tensor differ.
//	4) The strides nStride, cStride, hStride, wStride of the output tensor and the output differential tensor differ.
//
//	CUDNN_STATUS_BAD_PARAM:
//
//	At least one of the following conditions are met:
//
//	The strides nStride, cStride, hStride, wStride of the input differential tensor and output
//	differential tensors differ and in-place operation is used.
//
//	CUDNN_STATUS_EXECUTION_FAILED:
//
//	The function failed to launch on the GPU.
//
func (a *ActivationD) Backward(
	handle *Handle,
	alpha float64,
	yD *TensorD, y cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	xD *TensorD, x cutil.Mem,
	beta float64,
	dxD *TensorD, dx cutil.Mem) error {
	a1 := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnActivationBackward(handle.x, a.descriptor, a1.CPtr(), yD.descriptor, y.Ptr(), dyD.descriptor, dy.Ptr(), xD.descriptor, x.Ptr(), b.CPtr(), dxD.descriptor, dx.Ptr())).error("ActivationBackward")
}

//BackwardUS is just like Backward but it takes unsafe.Pointers instead of cutil.Mem
func (a *ActivationD) BackwardUS(
	handle *Handle,
	alpha float64,
	yD *TensorD, y unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	xD *TensorD, x unsafe.Pointer,
	beta float64,
	dxD *TensorD, dx unsafe.Pointer) error {
	a1 := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnActivationBackward(handle.x, a.descriptor, a1.CPtr(), yD.descriptor, y, dyD.descriptor, dy, xD.descriptor, x, b.CPtr(), dxD.descriptor, dx)).error("ActivationBackward")
}

//ActivationMode is used for activation discriptor flags flags are obtained through type's methods
type ActivationMode C.cudnnActivationMode_t

//Sigmoid sets a to ActivationMode(C.CUDNN_ACTIVATION_SIGMOID)and returns that value.
//
//Selects the sigmoid function.
func (a *ActivationMode) Sigmoid() ActivationMode {
	*a = ActivationMode(C.CUDNN_ACTIVATION_SIGMOID)
	return *a
}

//Relu sets a to ActivationMode(C.CUDNN_ACTIVATION_RELU)and returns that value.
//
//Selects the rectified linear function.
func (a *ActivationMode) Relu() ActivationMode {
	*a = ActivationMode(C.CUDNN_ACTIVATION_RELU)
	return *a
}

//Tanh sets a to ActivationMode(C.CUDNN_ACTIVATION_TANH)and returns that value.
//
//Selects the hyperbolic tangent function.
func (a *ActivationMode) Tanh() ActivationMode {
	*a = ActivationMode(C.CUDNN_ACTIVATION_TANH)
	return *a
}

//ClippedRelu sets a to  ActivationMode(C.CUDNN_ACTIVATION_CLIPPED_RELU)and returns that value.
//
//Selects the clipped rectified linear function.
func (a *ActivationMode) ClippedRelu() ActivationMode {
	*a = ActivationMode(C.CUDNN_ACTIVATION_CLIPPED_RELU)
	return *a
}

//Elu sets a to ActivationMode(C.CUDNN_ACTIVATION_ELU) and returns that value.
//
//Selects the exponential linear function.
func (a *ActivationMode) Elu() ActivationMode { *a = ActivationMode(C.CUDNN_ACTIVATION_ELU); return *a }

//Identity returns  ActivationMode(C.CUDNN_ACTIVATION_IDENTITY) (new for 7.1)
//
//Selects the identity function, intended for bypassing the activation step in (*Convolution)BiasActivationForward().
//(The Identity flag must use CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_​PRECOMP_GEMM, and only for (*Convolution)BiasActivationForward())
//Does not work with cudnnActivationForward() or cudnnActivationBackward().
func (a *ActivationMode) Identity() ActivationMode {
	*a = ActivationMode(C.CUDNN_ACTIVATION_SIGMOID)
	return *a
}
func (a ActivationMode) c() C.cudnnActivationMode_t      { return C.cudnnActivationMode_t(a) }
func (a *ActivationMode) cptr() *C.cudnnActivationMode_t { return (*C.cudnnActivationMode_t)(a) }
