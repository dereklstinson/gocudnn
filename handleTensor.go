package gocudnn

/*

#include <cudnn.h>

*/
import "C"
import (
	"errors"
	"unsafe"
)

//TransformTensor does something like this --> Tensor layout conversion helper (y = alpha * x + beta * y)
//Will have to play around with this layer to figure it out
func (h *Handle) TransformTensor(data DataType, alpha Memer, tx TensorD, x Memer, beta Memer, ty TensorD, y Memer) error {
	var s Status
	/*	var alphau, betau unsafe.Pointer

		switch data {

		case DataTypeInt32:
			a := C.int(alpha)
			b := C.int(beta)
			alphau = unsafe.Pointer(&a)
			betau = unsafe.Pointer(&b)
		case DataTypeFloat:
			a := C.float(alpha)
			b := C.float(beta)
			alphau = unsafe.Pointer(&a)
			betau = unsafe.Pointer(&b)
		case DataTypeDouble:
			a := C.double(alpha)
			b := C.double(beta)
			alphau = unsafe.Pointer(&a)
			betau = unsafe.Pointer(&b)
		default:
			return errors.New("Should have never reached this place we are in trouble")
		}
	*/
	s = Status(C.cudnnTransformTensor(h.x, alpha.Ptr(), tx.descriptor, x.Ptr(), beta.Ptr(), ty.descriptor, y.Ptr()))
	return s.error("TransformTensor")
}

//AddTensor Tensor Bias addition : C = alpha * A + beta * C // c is both the input and output
/*From Documentation
This function adds the scaled values of a bias tensor to another tensor.
Each dimension of the bias tensor A must match the corresponding dimension of the destination tensor C or must be equal to 1.
In the latter case, the same value from the bias tensor for those dimensions will be used to blend into the C tensor.

**Note: Up to dimension 5, all tensor formats are supported. Beyond those dimensions, this routine is not supported
*/
func (h *Handle) AddTensor(data DataType, alpha float64, tx TensorD, x Memer, beta float64, tc TensorD, c Memer) error {

	var alphau, betau unsafe.Pointer

	switch data {

	case DataTypeInt32:
		a := C.int(alpha)
		b := C.int(beta)
		alphau = unsafe.Pointer(&a)
		betau = unsafe.Pointer(&b)
	case DataTypeFloat:
		a := C.float(alpha)
		b := C.float(beta)
		alphau = unsafe.Pointer(&a)
		betau = unsafe.Pointer(&b)
	case DataTypeDouble:
		a := C.double(alpha)
		b := C.double(beta)
		alphau = unsafe.Pointer(&a)
		betau = unsafe.Pointer(&b)
	default:
		return errors.New("Should have never reached this place we are in trouble")
	}
	s := Status(C.cudnnTransformTensor(h.x, alphau, tx.descriptor, x.Ptr(), betau, tc.descriptor, c.Ptr()))
	return s.error("TransformTensor")
}
