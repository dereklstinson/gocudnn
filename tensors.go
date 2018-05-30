package cudnn

/*

#include <cudnn.h>

*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

//PONDERING most of the tensor info stuff(like the stuff that says "Get.....") should probably be private.

//DataType is used for flags for the tensor layer structs
type DataType C.cudnnDataType_t

//There are more constants that are supported with cudnn but since these types are supported with go that is what I am going to use
const (
	DataTypeFloat  DataType = 0
	DataTypeDouble DataType = 1
	//DataTypeInt8   DataType = 3
	DataTypeInt32 DataType = 4
	//DataTypeUint8  DataType = 6
	//This is added in by me so for error checking
	DataTypeerror DataType = 100
)

//MathType are flags to set for cudnnMathType_t
type MathType C.cudnnMathType_t

//Flags for cudnnMathType
const (
	MathTypeDefault MathType = iota
	MathTypeTensorOP
)

func (math MathType) string() string {
	if math == MathTypeDefault {
		return "Math Type Default"
	}
	return "Math Type Tensor OP"
}

//PropagationNAN  is type for C.cudnnNanPropagation_t used for flags
type PropagationNAN C.cudnnNanPropagation_t

//Flags for cudnnNanPropagation
const (
	PropagateNanNot PropagationNAN = iota
	PropagateNan
)

//Determinism is the type for flags that set Determinism
type Determinism C.cudnnDeterminism_t

//Flags for Determinism
const (
	DeterministicNON Determinism = iota
	Deterministic
)

func (d Determinism) string() string {
	if d == DeterministicNON {
		return "Non Deterministic"
	}
	return "Deterministic "
}

//TensorFormat is the type used for flags to set tensor format
type TensorFormat C.cudnnTensorFormat_t

//Flags for TensorFormat
const (
	TensorFormatNCHW TensorFormat = iota
	TensorFormatNHWC
	TensorFormatNCHWVectC
)

type tensorDescriptor uint

const (
	t4d   tensorDescriptor = 1
	t4dEx tensorDescriptor = 2
	tnd   tensorDescriptor = 3
	tndex tensorDescriptor = 4
)

//TensorD holds the cudnnTensorDescriptor. Which is basically the tensor itself
type TensorD struct {
	descriptor C.cudnnTensorDescriptor_t
	flag       tensorDescriptor
	format     C.cudnnTensorFormat_t
	data       C.cudnnDataType_t
	size       C.size_t
	dims       C.int
	shape      []C.int
	stride     []C.int
	destroyed  bool
}

//Format returns the Format of TensorD
func (t *TensorD) Format() TensorFormat { return TensorFormat(t.format) }

//Data returns the data flag for tensor
func (t *TensorD) Data() DataType { return DataType(t.data) }

//SizeInBytes returns the tensors Size in bytes
func (t *TensorD) SizeInBytes() int { return int(t.size) }

//Dims returns the # of dims in TensorD
func (t *TensorD) Dims() int { return int(t.dims) }

//Shape returns the shape int array for tensorD
func (t *TensorD) Shape() []int { return cintToint(t.shape) }

//Stride returns the stride int array for tensorD
func (t *TensorD) Stride() []int { return cintToint(t.stride) }

//DimMax is the current max dims that this version can do
const DimMax int = 8

//Shape basically takes some arguments and makes a slice out of them. This is made out of convenience to the user when building a tensor. It will not return an error
func Shape(nums ...int) []int {
	return nums
}

//Strides takes some arguements and makes a slice out of them This is made out of convenience to the user when building a tensor. It will not return an error
func Strides(nums ...int) []int {
	return nums
}

//CreateTensorD Creates a tensor descriptor
func CreateTensorD() (*TensorD, error) {
	var descriptor C.cudnnTensorDescriptor_t
	err := Status(C.cudnnCreateTensorDescriptor(&descriptor)).error("CreateTensorD")
	retTens := &TensorD{
		descriptor: descriptor,
	}
	return retTens, err
}

//NewTensor returns a tensor and error. stride interface will only accept []int or nil.
//Note: The total size of a tensor including the potential padding between dimensions is limited to 2 Giga-elements of type datatype.
//Tensors are restricted to having at least 4 dimensions, and at most DimMax (8) dimensions (as of cudnn 7.2 Which is what this is based on.
//When working with lower dimensional data, it is recommended that the user create a 4D tensor, and set the size along unused dimensions to 1.
//Note: At present, some cuDNN routines have limited support for strides; Those routines will return CUDNN_STATUS_NOT_SUPPORTED if a Tensor4D object with an unsupported stride is used. cudnnTransformTensor can be used to convert the data to a supported layout.
//Note: The total size of a tensor including the potential padding between dimensions is limited to 2 Giga-elements of type datatype.
//Note: At present, some cuDNN routines have limited support for strides; Those routines will return CUDNN_STATUS_NOT_SUPPORTED if a Tensor4D object with an unsupported stride is used. cudnnTransformTensor can be used to convert the data to a supported layout.
//Note: The total size of a tensor including the potential padding between dimensions is limited to 2 Giga-elements of type datatype.
func NewTensor(data DataType, format TensorFormat, shape []int, stride interface{}) (TensorD, error) {

	var newtensor TensorD
	if len(shape) < 4 {
		return newtensor, errors.New("Length of Shape array must be at least 4")
	}
	switch x := stride.(type) {
	case []int:
		if len(x) < 4 || len(x) > DimMax || len(x) != len(shape) {
			return newtensor, errors.New("Length of stride array must be at least 4 and less than DimMax and len(shape)==len(stride)")
		}
		newtensor.stride = make([]C.int, len(x))
		for i := 0; i < len(x); i++ {
			newtensor.stride[i] = C.int(x[i])
		}
	case nil:

	default:
		return newtensor, errors.New("Unsuppored format for stride")

	}
	newtensor.data = C.cudnnDataType_t(data)
	newtensor.format = C.cudnnTensorFormat_t(format)
	newtensor.dims = C.int(len(shape))

	newtensor.shape = intTocint(shape)
	x := Status(C.cudnnCreateTensorDescriptor(&newtensor.descriptor)).error("NewTensor")
	if x != nil {
		return newtensor, x
	}

	if len(shape) < 5 {
		x = newtensor.set4d()
		return newtensor, x
	}
	x = newtensor.setnd()
	return newtensor, x
}

func (t *TensorD) setnd() error {
	fmt.Println("Going Ito SetNd")
	var x error
	if len(t.stride) < 1 {
		x = t.setTensorNdDescriptorEx()
		if x != nil {
			return x
		}
		x = t.setTensorSizeInBytes()
		return x
	}
	x = t.setTensor4dDescriptor()
	if x != nil {
		return x
	}
	x = t.setTensorSizeInBytes()
	return x

}

func (t *TensorD) set4d() error {

	var x error
	if len(t.stride) < 1 {
		x = t.setTensor4dDescriptor()
		if x != nil {
			return x
		}
		x = t.setTensorSizeInBytes()
		return x
	}
	x = t.setTensor4dDescriptorEx()
	if x != nil {
		return x
	}
	x = t.setTensorSizeInBytes()
	return x
}

/*
//AddStride allows you to add the stride if need be.
func (t *Tensor) AddStride(stride []int)error {

	t.stride = make([]C.int, len(stride))
	for i := 0; i < len(stride); i++ {
		t.stride[i] = C.int(stride[i])
	}
	return nil
}
*/

//SetTensor4dDescriptor sets the descriptor to a 4dtensor with the size given
func (t *TensorD) setTensor4dDescriptor() error {
	if len(t.shape) != 4 {
		return errors.New("SetTensor4dDescriptor:  len(t.shape)!=4 ")
	}
	x := C.cudnnSetTensor4dDescriptor(t.descriptor, t.format, t.data, t.shape[0], t.shape[1], t.shape[2], t.shape[3])
	return Status(x).error("SetTensor4dDescriptor")

}

//SetTensor4dDescriptorEx sets the descriptor to a 4dtensor with the size given and strides

func (t *TensorD) setTensor4dDescriptorEx() error {
	if len(t.stride) != 4 || len(t.shape) != 4 {
		return errors.New("SetTensor4dDescriptorEx: len(t.stride)!=4 || len(t.shape) !=4")
	}
	x := C.cudnnSetTensor4dDescriptorEx(t.descriptor, t.data, t.shape[0], t.shape[1], t.shape[2], t.shape[3], t.stride[0], t.stride[1], t.stride[2], t.stride[3])
	return Status(x).error("SetTensor4dDescriptorEx")

}

//GetTensor4dDescrptor this info should already be in the struct that is used to make this stuff, but I will put it here anyways
//This should probably be private
/*
func (t *TensorD) GetTensor4dDescrptor() (DataType, []C.int, []C.int, error) {
	shape := make([]C.int, 4)
	stride := make([]C.int, 4)
	var data C.cudnnDataType_t
	x := C.cudnnGetTensor4dDescriptor(t.descriptor, &data, &shape[0], &shape[1], &shape[2], &shape[3], &stride[0], &stride[1], &stride[2], &stride[3])

	return DataType(data), shape, stride, Status(x).error("SetTensor4dDescriptorEx")
}
*/
//SetTensorNdDescriptor is used to set multi dim tensor of dim n. max size is 8
func (t *TensorD) setTensorNdDescriptor() error {
	if len(t.stride) == 0 || len(t.shape) == 0 {
		return errors.New("SetTensorNdDescriptor: len(t.stride)==0 || len(t.shape) ==0")
	}
	x := C.cudnnSetTensorNdDescriptor(t.descriptor, t.data, t.dims, &t.shape[0], &t.stride[0])
	return Status(x).error("SetTensorNdDescriptor")
}
func (t *TensorD) setTensorNdDescriptorEx() error {
	x := C.cudnnSetTensorNdDescriptorEx(t.descriptor, t.format, t.data, t.dims, &t.shape[0])
	return Status(x).error("setTensorNdDescriptorEx")
}

/*
//GetTensorNdDescriptor this info should already be in the struct that is used to make this stuff, but I will put it here anyways
//This should probably be private.
func (t *TensorD) GetTensorNdDescriptor(dimrequest int) (DataType, C.int, []C.int, []C.int, error) {
	shape := make([]C.int, t.dims)
	stride := make([]C.int, t.dims)

	var dims C.int
	var data C.cudnnDataType_t
	x := C.cudnnGetTensorNdDescriptor(t.descriptor, C.int(dimrequest), &data, &dims, &shape[0], &stride[0])

	return DataType(data), dims, shape, stride, Status(x).error("GetTensorNdDescriptor")
}
*/
//GetTensorSizeInBytes returns the C.size)t and Status
//This Should probably be private.
func (t *TensorD) setTensorSizeInBytes() error {
	var sizebytes C.size_t
	x := C.cudnnGetTensorSizeInBytes(t.descriptor, &sizebytes)
	t.size = sizebytes
	return Status(x).error("GetTensorNdDescriptor")
}

//IsDestroyed checks if the tensor is destroyed.  It will return a true if it is destroyed. If it is then this can be used again.

//DestroyTensorD destroys the tensor
func (t *TensorD) DestroyTensorD() error {
	x := Status(C.cudnnDestroyTensorDescriptor(t.descriptor))

	return x.error("GetTensorNdDescriptor")
}

//
//TransformTensor does something like this --> Tensor layout conversion helper (y = alpha * x + beta * y)
//Will have to play around with this layer to figure it out
func (h *Handle) TransformTensor(alpha float64, tx TensorD, x Memer, beta float64, ty TensorD, y Memer) error {
	var s Status
	var alphau, betau unsafe.Pointer
	if DataType(tx.data) != DataType(ty.data) {
		return errors.New("The Data Types Don't Match in the TransformTensor")
	}
	switch DataType(tx.data) {

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
	s = Status(C.cudnnTransformTensor(h.x, alphau, tx.descriptor, x.Ptr(), betau, ty.descriptor, y.Ptr()))
	return s.error("TransformTensor")
}

//AddTensor Tensor Bias addition : C = alpha * A + beta * C // c is both the input and output
/*From Documentation
This function adds the scaled values of a bias tensor to another tensor.
Each dimension of the bias tensor A must match the corresponding dimension of the destination tensor C or must be equal to 1.
In the latter case, the same value from the bias tensor for those dimensions will be used to blend into the C tensor.

**Note: Up to dimension 5, all tensor formats are supported. Beyond those dimensions, this routine is not supported
*/
func (h *Handle) AddTensor(alpha float64, tx TensorD, x Memer, beta float64, tc TensorD, c Memer) error {

	var alphau, betau unsafe.Pointer
	if DataType(tx.data) != DataType(tc.data) {
		return errors.New("The Data Types Don't Match in the TransformTensor")
	}
	switch DataType(tx.data) {

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

func datatypecheck(input interface{}) (DataType, error) {
	switch input.(type) {

	case int32:
		return DataTypeInt32, nil
	case float32:
		return DataTypeFloat, nil

	case float64:
		return DataTypeDouble, nil
	default:
		return DataTypeerror, errors.New("Unsupported DataType")

	}

}
