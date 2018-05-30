package cudnn

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

//OPTensorD holds OP Tensor information
type OPTensorD struct {
	opTensorDesc     C.cudnnOpTensorDescriptor_t
	opTensOp         C.cudnnOpTensorOp_t
	opTensorCompType C.cudnnDataType_t
	opTensorNanOpt   C.cudnnNanPropagation_t
}

//CreateOpTensorDescriptor creates the optensor descriptor
func CreateOpTensorDescriptor(opTensOp OpTensorOp, opTensorCompType DataType, opTensorNanOpt PropagationNAN) (OPTensorD, error) {
	var optensor OPTensorD
	optensor.opTensOp = C.cudnnOpTensorOp_t(opTensOp)
	optensor.opTensorCompType = C.cudnnDataType_t(opTensorCompType)
	optensor.opTensorNanOpt = C.cudnnNanPropagation_t(opTensorNanOpt)
	x := C.cudnnCreateOpTensorDescriptor(&optensor.opTensorDesc)
	return optensor, Status(x).error("CreateOpTensorDescriptor")
}

//SetOpTensorDescriptor sets the OPTensor Descriptor
func (t *OPTensorD) SetOpTensorDescriptor() error {
	x := C.cudnnSetOpTensorDescriptor(t.opTensorDesc, t.opTensOp, t.opTensorCompType, t.opTensorNanOpt)
	return Status(x).error("SetOpTensorDescriptor")
}

//GetOpTensorDescriptor returns a copy of the information of OPTensor ...(It uses the built in function and not the struct that was created on the go side)
func (t *OPTensorD) GetOpTensorDescriptor() (OPTensorD, error) {
	var some OPTensorD
	some.opTensorDesc = t.opTensorDesc

	x := C.cudnnGetOpTensorDescriptor(some.opTensorDesc, &some.opTensOp, &some.opTensorCompType, &some.opTensorNanOpt)
	return some, Status(x).error("GetOpTensorDescriptor")
}

//OpTensor performs an operation on some tensors
func (h *Handle) OpTensor(t *OPTensorD, alpha1 float64, aDesc TensorD, A Memer,
	alpha2 float64, bDesc TensorD, B Memer,
	beta float64, cDesc TensorD, Ce Memer) error {

	if DataType(aDesc.data) != DataType(bDesc.data) || DataType(aDesc.data) != DataType(cDesc.data) {
		return errors.New("The Data Types Don't Match in the TransformTensor")
	}
	var alpha1u, alpha2u, betau unsafe.Pointer
	switch DataType(aDesc.data) {

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
	x := C.cudnnOpTensor(h.x, t.opTensorDesc, alpha1u, aDesc.descriptor, A.Ptr(), alpha2u, bDesc.descriptor, B.Ptr(), betau, cDesc.descriptor, Ce.Ptr())
	return Status(x).error("OpTensor")
}
