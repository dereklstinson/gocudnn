package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//OpTensor is a struct that is used in making op tensors it holds the Funcs and Flgs for optensors
type OpTensor struct {
	Flgs OpTensorFlag
}

//OPTensorD holds OP Tensor information
type OPTensorD struct {
	descriptor C.cudnnOpTensorDescriptor_t
}

//NewOpTensorDescriptor creates and sets an OpTensor
func (op OpTensor) NewOpTensorDescriptor(opTensOp OpTensorOp, opTensorCompType DataType, opTensorNanOpt PropagationNAN) (descriptor *OPTensorD, err error) {
	var desc C.cudnnOpTensorDescriptor_t
	err = Status(C.cudnnCreateOpTensorDescriptor(&desc)).error("NewOpTensorDescriptor-Create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetOpTensorDescriptor(desc, opTensOp.c(), C.cudnnDataType_t(opTensorCompType), C.cudnnNanPropagation_t(opTensorNanOpt))).error("NewOpTensorDescriptor-set")
	if err != nil {
		return nil, err
	}
	descriptor = &OPTensorD{descriptor: desc}
	if setfinalizer {
		runtime.SetFinalizer(descriptor, destroyopdesc)
	}

	return descriptor, nil
}

//GetDescriptor returns the descriptor information with error
func (t *OPTensorD) GetDescriptor() (OpTensorOp, DataType, PropagationNAN, error) {
	var tensop C.cudnnOpTensorOp_t
	var datatype C.cudnnDataType_t
	var nanprop C.cudnnNanPropagation_t

	x := C.cudnnGetOpTensorDescriptor(t.descriptor, &tensop, &datatype, &nanprop)
	if setkeepalive {
		t.keepsalive()
	}
	return OpTensorOp(tensop), DataType(datatype), PropagationNAN(nanprop), Status(x).error("GetOpTensorDescriptor")

}
func destroyopdesc(t *OPTensorD) error {
	return Status(C.cudnnDestroyOpTensorDescriptor(t.descriptor)).error("destroyoptensor")
}
func (t *OPTensorD) keepsalive() {
	runtime.KeepAlive(t)
}

//DestroyDescriptor destroys the descriptor
func (t *OPTensorD) DestroyDescriptor() error {

	return destroyopdesc(t)
}

//OpTensor performs an operation on some tensors   C= operation( (alpha1 * A) , (alpha2 *B) ) + (beta *C)
func (t *OPTensorD) OpTensor(
	handle *Handle,
	alpha1 float64,
	aD *TensorD,
	A gocu.Mem,
	alpha2 float64,
	bD *TensorD,
	B gocu.Mem,
	beta float64,
	cD *TensorD,
	cmem gocu.Mem) error {
	a1 := cscalarbydatatype(aD.dtype, alpha1)
	a2 := cscalarbydatatype(bD.dtype, alpha2)
	b := cscalarbydatatype(cD.dtype, beta)
	x := C.cudnnOpTensor(
		handle.x,
		t.descriptor,
		a1.CPtr(),
		aD.descriptor,
		A.Ptr(),
		a2.CPtr(),
		bD.descriptor,
		B.Ptr(),
		b.CPtr(),
		cD.descriptor,
		cmem.Ptr())

	return Status(x).error("OpTensor")
}

//OpTensorFlag is used pass OpTensorOp Flags Semi Safely using methods
type OpTensorFlag struct {
}

//OpTensorOp is used for flags for the Optensor functions
type OpTensorOp C.cudnnOpTensorOp_t

//Add returns 	 OpTensorOp(C.CUDNN_OP_TENSOR_ADD)
func (o OpTensorFlag) Add() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_ADD)
}

//Mul returns  OpTensorOp(C.CUDNN_OP_TENSOR_MUL)
func (o OpTensorFlag) Mul() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_MUL)
}

//Min returns OpTensorOp(C.CUDNN_OP_TENSOR_MIN)
func (o OpTensorFlag) Min() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_MIN)
}

//Max returns OpTensorOp(C.CUDNN_OP_TENSOR_MAX)
func (o OpTensorFlag) Max() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_MAX)
}

//Sqrt returns OpTensorOp(C.CUDNN_OP_TENSOR_SQRT)
func (o OpTensorFlag) Sqrt() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_SQRT)
}

//Not returns OpTensorOp(C.CUDNN_OP_TENSOR_NOT)
func (o OpTensorFlag) Not() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_NOT)
}
func (o OpTensorOp) c() C.cudnnOpTensorOp_t { return C.cudnnOpTensorOp_t(o) }
