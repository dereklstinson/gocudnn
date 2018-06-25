package gocudnn

/*

#include <cudnn.h>

*/
import "C"
import (
	"errors"
)

//Tensor is used for calling Tensor Funcs and also holds the flags
type Tensor struct {
	Flgs  TensorFlags
	Funcs TensorFuncs
}

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

//NewTensor4dDescriptor Creates and Sets a Tensor 4d Descriptor.
func (t Tensor) NewTensor4dDescriptor(data DataType, format TensorFormat, shape []int32) (*TensorD, error) {
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
func (t Tensor) NewTensor4dDescriptorEx(data DataType, shape, stride []int32) (*TensorD, error) {
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
func (t Tensor) NewtTensorNdDescriptor(data DataType, shape, stride []int32) (*TensorD, error) {
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
func (t Tensor) NewTensorNdDescriptorEx(format TensorFormat, data DataType, shape, stride []int32) (*TensorD, error) {
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

//TensorFuncs is used to call functions for tensors
type TensorFuncs struct {
}

//TransformTensor does something like this --> Tensor layout conversion helper (y = alpha * x + beta * y)
//Will have to play around with this layer to figure it out
func (ten TensorFuncs) TransformTensor(h *Handle, data DataType, alpha CScalar, tx *TensorD, x Memer, beta CScalar, ty *TensorD, y Memer) error {
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
func (ten TensorFuncs) AddTensor(h *Handle, data DataType, alpha CScalar, aD *TensorD, A Memer, beta CScalar, cD *TensorD, c Memer) error {

	s := Status(C.cudnnAddTensor(h.x, alpha.CPtr(), aD.descriptor, A.Ptr(), beta.CPtr(), cD.descriptor, c.Ptr()))
	return s.error("AddTensor")
}

//TensorFlags holds all things that tensors use for flags
type TensorFlags struct {
	Data   DataTypeFlag
	Math   MathTypeFlag
	NaN    PropagationNANFlag
	Deter  DeterminismFlag
	Format TensorFormatFlag
}

/*
*
*
*       DataTypeFlag
*
*
 */

//DataTypeFlag is used to pass DataType flags semi-safely though methods
type DataTypeFlag struct {
}

//DataType is used for flags for the tensor layer structs
type DataType C.cudnnDataType_t

//FindScalar finds a CScalar value for the datatype being used by the tensors
func FindScalar(datatype DataType, x float64) CScalar {
	switch datatype {
	case DataType(C.CUDNN_DATA_FLOAT):
		return CFloat(x)
	case DataType(C.CUDNN_DATA_DOUBLE):
		return CDouble(x)
	case DataType(C.CUDNN_DATA_INT8):
		return CInt(x)
	case DataType(C.CUDNN_DATA_INT32):
		return CInt(x)
	default:
		return CInt(x)
	}
}

// Float return DataType(C.CUDNN_DATA_FLOAT)
func (d DataTypeFlag) Float() DataType {
	return DataType(C.CUDNN_DATA_FLOAT)
}

// Double return DataType(C.CUDNN_DATA_DOUBLE)
func (d DataTypeFlag) Double() DataType {
	return DataType(C.CUDNN_DATA_DOUBLE)
}

// Int8 return DataType(C.CUDNN_DATA_INT8)
func (d DataTypeFlag) Int8() DataType {
	return DataType(C.CUDNN_DATA_INT8)
}

// Int32 return DataType(C.CUDNN_DATA_INT32)
func (d DataTypeFlag) Int32() DataType {
	return DataType(C.CUDNN_DATA_INT32)
}

// UInt8 return DataType(C.CUDNN_DATA_INT8)
func (d DataTypeFlag) UInt8() DataType {
	return DataType(C.CUDNN_DATA_UINT8)
}

func (d DataType) c() C.cudnnDataType_t { return C.cudnnDataType_t(d) }

/*
*
*
*       MathTypeFlag
*
*
 */

//MathTypeFlag used to pass MathType Flags semi-safely through methods.
type MathTypeFlag struct {
}

//MathType are flags to set for cudnnMathType_t
type MathType C.cudnnMathType_t

//Default return MathType(C.CUDNN_DEFAULT_MATH)
func (math MathTypeFlag) Default() MathType {
	return MathType(C.CUDNN_DEFAULT_MATH)
}

//TensorOpMath return MathType(C.CUDNN_TENSOR_OP_MATH)
func (math MathTypeFlag) TensorOpMath() MathType {
	return MathType(C.CUDNN_TENSOR_OP_MATH)
}

func (math MathType) c() C.cudnnMathType_t { return C.cudnnMathType_t(math) }

/*
*
*
*       PropagationNANFlag
*
*
 */

//PropagationNANFlag used to return Propagation flags semi-safely through methods
type PropagationNANFlag struct {
}

//PropagationNAN  is type for C.cudnnNanPropagation_t used for flags
type PropagationNAN C.cudnnNanPropagation_t

//NotPropagateNan return PropagationNAN(C.CUDNN_NOT_PROPAGATE_NAN) flag
func (p PropagationNANFlag) NotPropagateNan() PropagationNAN {
	return PropagationNAN(C.CUDNN_NOT_PROPAGATE_NAN)
}

//PropagateNan return PropagationNAN(C.CUDNN_PROPAGATE_NAN) flag
func (p PropagationNANFlag) PropagateNan() PropagationNAN {
	return PropagationNAN(C.CUDNN_PROPAGATE_NAN)
}

func (p PropagationNAN) c() C.cudnnNanPropagation_t { return C.cudnnNanPropagation_t(p) }

/*
*
*
*       DeterminismFlag
*
*
 */

//DeterminismFlag used to pass Determinism flags semi-safely using methods
type DeterminismFlag struct {
}

//Determinism is the type for flags that set Determinism
type Determinism C.cudnnDeterminism_t

//Non returns  Determinism(C.CUDNN_NON_DETERMINISTIC)
func (d DeterminismFlag) Non() Determinism { return Determinism(C.CUDNN_NON_DETERMINISTIC) }

//Deter returns Determinism(C.CUDNN_DETERMINISTIC)
func (d DeterminismFlag) Deter() Determinism { return Determinism(C.CUDNN_DETERMINISTIC) }

func (d Determinism) c() C.cudnnDeterminism_t { return C.cudnnDeterminism_t(d) }

func (d Determinism) string() string {
	if d == Determinism(C.CUDNN_NON_DETERMINISTIC) {
		return "Non Deterministic"
	}
	return "Deterministic "
}

/*
*
*
*       TensorFormatFlag
*
*
 */

//TensorFormatFlag used to pass TensorFormat Flags semi safely
type TensorFormatFlag struct {
}

//TensorFormat is the type used for flags to set tensor format
type TensorFormat C.cudnnTensorFormat_t

//NCHW return TensorFormat(C.CUDNN_TENSOR_NCHW)
func (t TensorFormatFlag) NCHW() TensorFormat {
	return TensorFormat(C.CUDNN_TENSOR_NCHW)
}

//NHWC return TensorFormat(C.CUDNN_TENSOR_NHWC)
func (t TensorFormatFlag) NHWC() TensorFormat {
	return TensorFormat(C.CUDNN_TENSOR_NHWC)
}

//NCHWvectC return TensorFormat(C.CUDNN_TENSOR_NCHW_VECT_C)
func (t TensorFormatFlag) NCHWvectC() TensorFormat {
	return TensorFormat(C.CUDNN_TENSOR_NCHW_VECT_C)
}
func (t TensorFormat) c() C.cudnnTensorFormat_t { return C.cudnnTensorFormat_t(t) }
