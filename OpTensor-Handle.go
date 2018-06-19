package gocudnn

/*
#include <cudnn.h>
*/
import (
	"C"
)

//OpTensor performs an operation on some tensors   C= operation( (alpha1 * A) , (alpha2 *B) ) + (beta *C)
func (handle *Handle) OpTensor(
	t *OPTensorD,
	alpha1 CScalar,
	aDesc *TensorD,
	A Memer,
	alpha2 CScalar,
	bDesc *TensorD,
	B Memer,
	beta CScalar,
	cDesc *TensorD,
	c Memer) error {

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
		c.Ptr())
	return Status(x).error("OpTensor")
}
