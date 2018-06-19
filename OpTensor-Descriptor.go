package gocudnn

/*
#include <cudnn.h>
*/
import (
	"C"
)

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
func (t *OPTensorD) GetDescriptor() (OpTensorOp, DataType, PropagationNAN, error) {
	var tensop C.cudnnOpTensorOp_t
	var datatype C.cudnnDataType_t
	var nanprop C.cudnnNanPropagation_t

	x := C.cudnnGetOpTensorDescriptor(t.descriptor, &tensop, &datatype, &nanprop)
	return OpTensorOp(tensop), DataType(datatype), PropagationNAN(nanprop), Status(x).error("GetOpTensorDescriptor")
}

//DestroyDescriptor destroys the descriptor
func (t *OPTensorD) DestroyDescriptor() error {
	x := C.cudnnDestroyOpTensorDescriptor(t.descriptor)
	return Status(x).error("DestroyDescriptor")
}
