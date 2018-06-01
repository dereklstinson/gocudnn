package gocudnn

/*
#include <cudnn.h>
*/
import (
	"C"
)
import (
	"errors"
	"unsafe"
)

//OpTensorOp is used for flags for the Optensor functions
type OpTensorOp C.cudnnOpTensorOp_t

//These are direct flags of the enums used in OpTensor
const (
	OpTensorOpAdd  OpTensorOp = C.CUDNN_OP_TENSOR_ADD
	OpTensorOpMul  OpTensorOp = C.CUDNN_OP_TENSOR_MUL
	OpTensorOpMin  OpTensorOp = C.CUDNN_OP_TENSOR_MIN
	OpTensorOpMax  OpTensorOp = C.CUDNN_OP_TENSOR_MAX
	OpTensorOpSqrt OpTensorOp = C.CUDNN_OP_TENSOR_SQRT
	OpTensorOpNot  OpTensorOp = C.CUDNN_OP_TENSOR_NOT
)

func (o OpTensorOp) c() C.cudnnOpTensorOp_t { return C.cudnnOpTensorOp_t(o) }

//OPTensorD holds OP Tensor information
type OPTensorD struct {
	descriptor C.cudnnOpTensorDescriptor_t
}

//NewOpTensorDescriptor creates and sets an OpTensor
func NewOpTensorDescriptor(opTensOp OpTensorOp, opTensorCompType DataType, opTensorNanOpt PropagationNAN) (*OPTensorD, error) {
	var descriptor C.cudnnOpTensorDescriptor_t
	err := Status(C.cudnnCreateOpTensorDescriptor(&descriptor)).error("NewOpTensorDescriptor-Create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetOpTensorDescriptor(descriptor, opTensOp.c(), C.cudnnDataType_t(opTensorCompType), C.cudnnNanPropagation_t(opTensorNanOpt))).error("NewOpTensorDescriptor-set")
	if err != nil {
		return nil, err
	}
	return &OPTensorD{descriptor: descriptor}, nil
}

//GetDescriptor returns the descriptor information with error
func (t *OPTensorD) GetDescriptor() (OpTensorOp, OpTensorOp, PropagationNAN, error) {
	var tensop C.cudnnOpTensorOp_t
	var datatype C.cudnnDataType_t
	var nanprop C.cudnnNanPropagation_t

	x := C.cudnnGetOpTensorDescriptor(t.descriptor, &tensop, &datatype, &nanprop)
	return OpTensorOp(tensop), OpTensorOp(datatype), PropagationNAN(nanprop), Status(x).error("GetOpTensorDescriptor")
}

//OpTensor performs an operation on some tensors
func (h *Handle) OpTensor(data DataType, t *OPTensorD, alpha1 float64, aDesc TensorD, A Memer,
	alpha2 float64, bDesc TensorD, B Memer,
	beta float64, cDesc TensorD, Ce Memer) error {

	var alpha1u, alpha2u, betau unsafe.Pointer
	switch data {

	case DataTypeInt32:
		a1 := C.int(alpha1)
		a2 := C.int(alpha2)
		b := C.int(beta)
		alpha1u = unsafe.Pointer(&a1)
		alpha2u = unsafe.Pointer(&a2)
		betau = unsafe.Pointer(&b)
	case DataTypeFloat:

		a1 := C.float(alpha1)
		a2 := C.float(alpha2)
		b := C.float(beta)
		alpha1u = unsafe.Pointer(&a1)
		alpha2u = unsafe.Pointer(&a2)
		betau = unsafe.Pointer(&b)
	case DataTypeDouble:

		a1 := C.double(alpha1)
		a2 := C.double(alpha2)
		b := C.double(beta)
		alpha1u = unsafe.Pointer(&a1)
		alpha2u = unsafe.Pointer(&a2)
		betau = unsafe.Pointer(&b)
	default:
		return errors.New("Should have never reached this place we are in trouble")
	}
	x := C.cudnnOpTensor(h.x, t.descriptor, alpha1u, aDesc.descriptor, A.Ptr(), alpha2u, bDesc.descriptor, B.Ptr(), betau, cDesc.descriptor, Ce.Ptr())
	return Status(x).error("OpTensor")
}
