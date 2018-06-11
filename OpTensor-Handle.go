package gocudnn

/*
#include <cudnn.h>
*/
import (
	"C"
)

//OpTensor performs an operation on some tensors
func (handle *Handle) OpTensor(data DataType,
	t *OPTensorD,
	alpha1 CScalar,
	aDesc *TensorD,
	A Memer,
	alpha2 CScalar,
	bDesc *TensorD,
	B Memer,
	beta CScalar,
	cDesc *TensorD,
	Ce Memer) error {

	x := C.cudnnOpTensor(
		handle.x,
		t.descriptor,
		alpha1.CPtr(),
		aDesc.descriptor,
		A.Ptr(),
		alpha2.CPtr(),
		bDesc.descriptor,
		B.Ptr(),
		beta.CPtr(),
		cDesc.descriptor,
		Ce.Ptr())
	return Status(x).error("OpTensor")
}
