package gocudnn

/*

#include <cudnn.h>

*/
import "C"
import (
	"errors"
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

func (d DataType) c() C.cudnnDataType_t { return C.cudnnDataType_t(d) }

//MathType are flags to set for cudnnMathType_t
type MathType C.cudnnMathType_t

//Flags for cudnnMathType
const (
	MathTypeDefault MathType = iota
	MathTypeTensorOP
)

func (math MathType) c() C.cudnnMathType_t { return C.cudnnMathType_t(math) }

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

func (p PropagationNAN) c() C.cudnnNanPropagation_t { return C.cudnnNanPropagation_t(p) }

//Determinism is the type for flags that set Determinism
type Determinism C.cudnnDeterminism_t

//Flags for Determinism
const (
	DeterministicNON Determinism = iota
	Deterministic
)

func (d Determinism) c() C.cudnnDeterminism_t { return C.cudnnDeterminism_t(d) }
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
	TensorFormatNCHW      TensorFormat = C.CUDNN_TENSOR_NCHW
	TensorFormatNHWC      TensorFormat = C.CUDNN_TENSOR_NHWC
	TensorFormatNCHWVectC TensorFormat = C.CUDNN_TENSOR_NCHW_VECT_C
)

func (t TensorFormat) c() C.cudnnTensorFormat_t { return C.cudnnTensorFormat_t(t) }

type descflag uint32

const (
	t4d descflag = 1
	tnd descflag = 2
	t2d descflag = 3
)

//TensorD holds the cudnnTensorDescriptor. Which is basically the tensor itself
type TensorD struct {
	descriptor C.cudnnTensorDescriptor_t
	dims       C.int
	flag       descflag
}

//Shape basically takes some arguments and makes a slice out of them. This is made out of convenience to the user when building a tensor. It will not return an error
func Shape(nums ...int32) []int32 {
	return nums
}

//Strides takes some arguements and makes a slice out of them This is made out of convenience to the user when building a tensor. It will not return an error
func Strides(nums ...int32) []int32 {
	return nums
}

//NewTensor4dDescriptor Creates and Sets a Tensor 4d Descriptor.
func NewTensor4dDescriptor(data DataType, format TensorFormat, shape []int32) (*TensorD, error) {
	if len(shape) != 4 {
		return nil, errors.New("Shape array has to be length 4")
	}
	var descriptor C.cudnnTensorDescriptor_t
	err := Status(C.cudnnCreateTensorDescriptor(&descriptor)).error("NewTensor4dDescriptor-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetTensor4dDescriptor(descriptor, C.cudnnTensorFormat_t(format), C.cudnnDataType_t(data), C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3]))).error("NewTensor4dDescriptor-set")
	if err != nil {
		return nil, err
	}
	return &TensorD{descriptor: descriptor, dims: C.int(4), flag: t4d}, nil
}

//NewTensor4dDescriptorEx Creates and Sets A Tensor 4d Descriptor EX
func NewTensor4dDescriptorEx(data DataType, shape, stride []int32) (*TensorD, error) {
	if len(shape) != 4 || len(stride) != 4 {
		return nil, errors.New("len(shape) len(stride) both have to equal 4")
	}

	var descriptor C.cudnnTensorDescriptor_t
	err := Status(C.cudnnCreateTensorDescriptor(&descriptor)).error("NewTensor4dDescriptorEx-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetTensor4dDescriptorEx(descriptor, C.cudnnDataType_t(data), C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3]), C.int(stride[0]), C.int(stride[1]), C.int(stride[2]), C.int(stride[3]))).error("NewTensor4dDescriptorEX-set")
	if err != nil {
		return nil, err
	}
	return &TensorD{descriptor: descriptor, dims: C.int(4), flag: t4d}, nil

}

//NewtTensorNdDescriptor creates and sets an nd descriptor
func NewtTensorNdDescriptor(data DataType, shape, stride []int32) (*TensorD, error) {
	if len(shape) != len(stride) {
		return nil, errors.New("len(shape) must equal len(stride)")
	}
	if len(stride) < 4 || len(stride) > 8 {
		return nil, errors.New("length of arrays need to be >4 or <8")
	}
	var descriptor C.cudnnTensorDescriptor_t
	err := Status(C.cudnnCreateTensorDescriptor(&descriptor)).error("NewTensorNdDescriptor-create")
	if err != nil {
		return nil, err
	}
	dims := C.int(len(shape))
	shapecint := int32Tocint(shape)
	stridecint := int32Tocint(stride)
	err = Status(C.cudnnSetTensorNdDescriptor(descriptor, C.cudnnDataType_t(data), dims, &shapecint[0], &stridecint[0])).error("cudnnSetTensorNdDescriptor")
	if err != nil {
		return nil, err
	}

	return &TensorD{descriptor: descriptor, dims: dims, flag: tnd}, nil
}

//NewTensorNdDescriptorEx creates and sets an ND descriptor ex
func NewTensorNdDescriptorEx(format TensorFormat, data DataType, shape, stride []int32) (*TensorD, error) {
	if len(shape) < 4 {
		return nil, errors.New("Shape array has to be greater than  4")
	}
	var descriptor C.cudnnTensorDescriptor_t
	err := Status(C.cudnnCreateTensorDescriptor(&descriptor)).error("NewTensorNdDescriptorEx-create")
	if err != nil {
		return nil, err
	}
	dims := C.int(len(shape))
	shapecint := int32Tocint(shape)

	err = Status(C.cudnnSetTensorNdDescriptorEx(descriptor, C.cudnnTensorFormat_t(format), C.cudnnDataType_t(data), dims, &shapecint[0])).error("cudnnSetTensorNdDescriptorEx-set")
	if err != nil {
		return nil, err
	}
	return &TensorD{descriptor: descriptor, dims: dims, flag: tnd}, nil
}

//GetDescrptor returns Data Type the Dims for shape and stride and error.  for Descriptors without stride it will still return junk info. so be mindful when you code.
func (t *TensorD) GetDescrptor() (DataType, []int32, []int32, error) {
	shape := make([]C.int, t.dims)
	stride := make([]C.int, t.dims)
	var data C.cudnnDataType_t
	if t.flag == t4d {
		x := C.cudnnGetTensor4dDescriptor(t.descriptor, &data, &shape[0], &shape[1], &shape[2], &shape[3], &stride[0], &stride[1], &stride[2], &stride[3])
		return DataType(data), cintToint32(shape), cintToint32(stride), Status(x).error("SetTensor4dDescriptorEx")

	} else if t.flag == tnd {
		var holder C.int
		x := C.cudnnGetTensorNdDescriptor(t.descriptor, t.dims, &data, &holder, &shape[0], &stride[0])
		return DataType(data), cintToint32(shape), cintToint32(stride), Status(x).error("SetTensor4dDescriptorEx")
	}

	return DataType(data), cintToint32(shape), cintToint32(stride), errors.New("Tensor Not t4d,or tnd")
}

//GetSizeInBytes returns the SizeT in bytes and Status
func (t *TensorD) GetSizeInBytes() (SizeT, error) {
	var sizebytes C.size_t
	x := C.cudnnGetTensorSizeInBytes(t.descriptor, &sizebytes)

	return SizeT(sizebytes), Status(x).error("GetTensorNdDescriptor")
}

//IsDestroyed checks if the tensor is destroyed.  It will return a true if it is destroyed. If it is then this can be used again.

//DestroyTensorD destroys the tensor
func (t *TensorD) DestroyTensorD() error {
	x := Status(C.cudnnDestroyTensorDescriptor(t.descriptor))

	return x.error("GetTensorNdDescriptor")
}

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
