package gocudnn

/*

#include <cudnn.h>

*/
import "C"

//TransformTensor does something like this --> Tensor layout conversion helper (y = alpha * x + beta * y)
//Will have to play around with this layer to figure it out
func (h *Handle) TransformTensor(data DataType, alpha CScalar, tx *TensorD, x Memer, beta CScalar, ty *TensorD, y Memer) error {
	var s Status

	s = Status(C.cudnnTransformTensor(h.x, alpha.CPtr(), tx.descriptor, x.Ptr(), beta.CPtr(), ty.descriptor, y.Ptr()))
	return s.error("TransformTensor")
}

//AddTensor Tensor Bias addition : C = alpha * A + beta * C // c is both the input and output
/*From Documentation
This function adds the scaled values of a bias tensor to another tensor.
Each dimension of the bias tensor A must match the corresponding dimension of the destination tensor C or must be equal to 1.
In the latter case, the same value from the bias tensor for those dimensions will be used to blend into the C tensor.

**Note: Up to dimension 5, all tensor formats are supported. Beyond those dimensions, this routine is not supported
*/
func (h *Handle) AddTensor(data DataType, alpha CScalar, aD *TensorD, A Memer, beta CScalar, cD *TensorD, c Memer) error {

	s := Status(C.cudnnAddTensor(h.x, alpha.CPtr(), aD.descriptor, A.Ptr(), beta.CPtr(), cD.descriptor, c.Ptr()))
	return s.error("AddTensor")
}
