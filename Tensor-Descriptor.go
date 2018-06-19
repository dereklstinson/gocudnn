package gocudnn

/*

#include <cudnn.h>

*/
import "C"
import (
	"errors"
)

func (math MathType) string() string {
	if math == MathType(C.CUDNN_DEFAULT_MATH) {
		return "Math Type Default"
	}
	return "Math Type Tensor OP"
}

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

func tensorDArrayToC(input []*TensorD) []C.cudnnTensorDescriptor_t {
	descs := make([]C.cudnnTensorDescriptor_t, len(input))
	for i := 0; i < len(input); i++ {
		descs[i] = input[i].descriptor
	}
	return descs
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

//DestroyDescriptor destroys the tensor
func (t *TensorD) DestroyDescriptor() error {

	return Status(C.cudnnDestroyTensorDescriptor(t.descriptor)).error("DestroyDescriptor")

}
