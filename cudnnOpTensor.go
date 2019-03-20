package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//OPTensorD holds OP Tensor information
type OPTensorD struct {
	descriptor C.cudnnOpTensorDescriptor_t
	gogc       bool
}

//CreateOpTensorDescriptor creates and sets an OpTensor
func CreateOpTensorDescriptor() (*OPTensorD, error) {
	desc := new(OPTensorD)
	err := Status(C.cudnnCreateOpTensorDescriptor(&desc.descriptor)).error("NewOpTensorDescriptor-Create")
	if setfinalizer {
		runtime.SetFinalizer(desc, destroyopdesc)
	}

	return descriptor, err
}

//Set sets the OPTensorD.
func (t *OPTensorD) Set(op OpTensorOp, dtype DataType, nan NANProp) error {
	err = Status(C.cudnnSetOpTensorDescriptor(t.descriptor, op.c(), dtype.c(), nan.c())).error("NewOpTensorDescriptor-set")

}

//Get returns the descriptor information with error
func (t *OPTensorD) Get() (op OpTensorOp, dtype DataType, nan NANProp, err error) {

	err = Status(C.cudnnGetOpTensorDescriptor(t.descriptor, op.cptr(), dtype.cptr(), nan.cptr())).error("GetOpTensorDescriptor")

	return op, dtype, nan, err

}
func destroyopdesc(t *OPTensorD) error {
	return Status(C.cudnnDestroyOpTensorDescriptor(t.descriptor)).error("destroyoptensor")
}

//DestroyDescriptor destroys the descriptor
func (t *OPTensorD) Destroy() error {
	if setfinalizer || t.gogc {
		return
	}
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

//OpTensorOp is used for flags for the Optensor functions
type OpTensorOp C.cudnnOpTensorOp_t

//Add sets o to OpTensorOp(C.CUDNN_OP_TENSOR_ADD) and returns the new value
func (o *OpTensorOp) Add() OpTensorOp { *o = OpTensorOp(C.CUDNN_OP_TENSOR_ADD); return *o }

//Mul sets o to OpTensorOp(C.CUDNN_OP_TENSOR_MUL) and returns the new value
func (o *OpTensorOp) Mul() OpTensorOp { *o = OpTensorOp(C.CUDNN_OP_TENSOR_MUL); return *o }

//Min sets o to OpTensorOp(C.CUDNN_OP_TENSOR_MIN)  and returns the new value
func (o *OpTensorOp) Min() OpTensorOp { *o = OpTensorOp(C.CUDNN_OP_TENSOR_MIN); return *o }

//Max sets o to OpTensorOp(C.CUDNN_OP_TENSOR_MAX) and returns the new value
func (o *OpTensorOp) Max() OpTensorOp { *o = OpTensorOp(C.CUDNN_OP_TENSOR_MAX); return *o }

//Sqrt sets o to OpTensorOp(C.CUDNN_OP_TENSOR_SQRT) and returns the new value
func (o *OpTensorOp) Sqrt() OpTensorOp { *o = OpTensorOp(C.CUDNN_OP_TENSOR_SQRT); return *o }

//Not returns OpTensorOp(C.CUDNN_OP_TENSOR_NOT) and returns the new value
func (o *OpTensorOp) Not() OpTensorOp { *o = OpTensorOp(C.CUDNN_OP_TENSOR_NOT); return *o }

func (o OpTensorOp) c() C.cudnnOpTensorOp_t      { return C.cudnnOpTensorOp_t(o) }
func (o *OpTensorOp) cptr() *C.cudnnOpTensorOp_t { return (*C.cudnnOpTensorOp_t)(o) }
