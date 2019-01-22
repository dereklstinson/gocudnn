package gocudnn

/*

#include <cudnn.h>

*/
import "C"
import (
	"errors"
	"runtime"
	"strconv"

	"github.com/dereklstinson/half"
)

//Tensor is used for calling Tensor Funcs and also holds the flags
type Tensor struct {
	Flgs TensorFlags
}

func (math MathType) string() string {
	if math == MathType(C.CUDNN_DEFAULT_MATH) {
		return "Math Type Default"
	}
	return "Math Type Tensor OP"
}

type descflag uint32

const (
	t4d   descflag = 1
	tnd   descflag = 2
	t2d   descflag = 3
	t4dex descflag = 4
	tndex descflag = 5
)

//TensorD holds the cudnnTensorDescriptor. Which is basically the tensor itself
type TensorD struct {
	descriptor C.cudnnTensorDescriptor_t
	dims       C.int
	dimsarray  []int32
	dtype      DataType
	stride     []int32
	strided    bool
	frmt       TensorFormat
	flag       descflag
}

func (t *TensorD) keepsalive() {
	runtime.KeepAlive(t)
}
func tensorDArrayToC(input []*TensorD) []C.cudnnTensorDescriptor_t {
	descs := make([]C.cudnnTensorDescriptor_t, len(input))
	for i := 0; i < len(input); i++ {
		descs[i] = input[i].descriptor
	}
	return descs
}

//Shape basically takes some arguments and makes a slice out of them. This is made out of convenience to the user when building a tensor. It will not return an error
func (t Tensor) Shape(nums ...int32) []int32 {
	return nums
}

//NewTensor4dDescriptor Creates and Sets a Tensor 4d Descriptor.
func (t Tensor) NewTensor4dDescriptor(data DataType, format TensorFormat, shape []int32) (*TensorD, error) {
	if len(shape) != 4 {
		return nil, errors.New("Shape array has to be length 4")
	}
	stride := stridecalc(shape)
	var descriptor C.cudnnTensorDescriptor_t
	err := Status(C.cudnnCreateTensorDescriptor(&descriptor)).error("NewTensor4dDescriptor-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetTensor4dDescriptor(descriptor, C.cudnnTensorFormat_t(format), C.cudnnDataType_t(data), C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3]))).error("NewTensor4dDescriptor-set")
	if err != nil {
		return nil, err
	}
	x := &TensorD{descriptor: descriptor, dimsarray: shape, frmt: format, stride: stride, dims: C.int(4), flag: t4d}
	if setfinalizer == true {
		runtime.SetFinalizer(x, destroytensordescriptor)
	}

	return x, nil

}

//NewTensor4dDescriptorEx Creates and Sets A Tensor 4d Descriptor EX
func (t Tensor) NewTensor4dDescriptorEx(data DataType, shape, stride []int32) (*TensorD, error) {
	if len(shape) != 4 || len(stride) != 4 {
		return nil, errors.New("len(shape) = " + strconv.Itoa(len(shape)) + " len(stride) = " + strconv.Itoa(len(stride)) + " .. both have to equal 4")
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
	x := &TensorD{descriptor: descriptor, dimsarray: shape, stride: stride, strided: true, dims: C.int(4), flag: t4dex}
	if setfinalizer == true {
		runtime.SetFinalizer(x, destroytensordescriptor)
	}

	return x, nil

}

//NewTensorNdDescriptor creates and sets an nd descriptor
func (t Tensor) NewTensorNdDescriptor(data DataType, shape, stride []int32) (*TensorD, error) {
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
	x := &TensorD{descriptor: descriptor, dimsarray: shape, strided: true, stride: stride, dims: dims, flag: tnd}
	if setfinalizer == true {
		runtime.SetFinalizer(x, destroytensordescriptor)
	}
	return x, nil
}

//NewTensorNdDescriptorEx creates and sets an ND descriptor ex
func (t Tensor) NewTensorNdDescriptorEx(format TensorFormat, data DataType, shape []int32) (*TensorD, error) {
	if len(shape) < 4 {
		return nil, errors.New("Shape array has to be greater than  4")
	}

	stride := stridecalc(shape)
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
	x := &TensorD{descriptor: descriptor, dimsarray: shape, stride: stride, dims: dims, flag: tndex}
	if setfinalizer == true {
		runtime.SetFinalizer(x, destroytensordescriptor)
	}
	return x, nil

}

//GetFormat returns the format of the tensor error will return if tensor supports slide//
func (t *TensorD) GetFormat() (TensorFormat, error) {

	if t.flag == tndex || t.flag == t4d {
		return t.frmt, nil
	}
	if setkeepalive == true {
		t.keepsalive()
	}
	return 255, errors.New("Tensor Uses slide method")

}

//GetDescrptor returns Data Type the Dims for shape and stride and error.  for Descriptors without stride it will still return junk info. so be mindful when you code.
func (t *TensorD) GetDescrptor() (DataType, []int32, []int32, error) {

	shape := make([]C.int, t.dims)
	stride := make([]C.int, t.dims)
	var data C.cudnnDataType_t
	if t.flag == t4d || t.flag == t4dex {
		x := C.cudnnGetTensor4dDescriptor(t.descriptor, &data, &shape[0], &shape[1], &shape[2], &shape[3], &stride[0], &stride[1], &stride[2], &stride[3])
		if setkeepalive == true {
			t.keepsalive()
		}
		return DataType(data), cintToint32(shape), cintToint32(stride), Status(x).error("GetDescriptor")

	} else if t.flag == tnd || t.flag == tndex {
		var holder C.int
		x := C.cudnnGetTensorNdDescriptor(t.descriptor, t.dims, &data, &holder, &shape[0], &stride[0])
		if setkeepalive == true {
			t.keepsalive()
		}
		return DataType(data), cintToint32(shape), cintToint32(stride), Status(x).error("GetDescriptor")
	} else {
		var holder C.int
		if Status(C.cudnnGetTensorNdDescriptor(t.descriptor, t.dims, &data, &holder, &shape[0], &stride[0])).error("Checking") != nil {
			if Status(C.cudnnGetTensor4dDescriptor(t.descriptor, &data, &shape[0], &shape[1], &shape[2], &shape[3], &stride[0], &stride[1], &stride[2], &stride[3])).error("Checking") != nil {
				return DataType(data), cintToint32(shape), cintToint32(stride), errors.New("Tripplecheckpoint Didn't work I don't know what this tensorD is")
			}
			return DataType(data), cintToint32(shape), cintToint32(stride), nil
		}
		if setkeepalive == true {
			t.keepsalive()
		}
		return DataType(data), cintToint32(shape), cintToint32(stride), nil
	}
}

//Dims returns the dims
func (t *TensorD) Dims() []int32 {
	return t.dimsarray
}

//Strides returns the strides
func (t *TensorD) Strides() []int32 {
	return t.stride
}

//Format returns the format
func (t *TensorD) Format() TensorFormat {
	return t.frmt
}

//DataType holds the datatype
func (t *TensorD) DataType() DataType {
	return t.dtype

}

/*

n, c, h, w := int32(1), int32(3), int32(4), int32(2)
	sharedims := []int32{n, c, h, w}
	//tensor dims a 1,4,4,2... slide is 32,8,2,1
	chw := c * h * w
	hw := h * w
	ostride := []int32{chw, hw, w, 1}
	xDesc, err := tensor.NewTensor4dDescriptorEx(float, sharedims, ostride)
	if err != nil {
		t.Error(err)
	}

	x, y, z := int32(1), int32(4), int32(4)
	xyz := x * y * z
	yz := y * z
	stride := []int32{ostride[0] * xyz, ostride[1] * xyz, ostride[2] * yz, ostride[3] * z}
	outputdims := []int32{(stride[0] * sharedims[0]) / (chw * xyz), (sharedims[1] * stride[1]) / (yz * hw), (sharedims[2] * stride[2]) / (w * z), sharedims[3] * stride[3]}


	//if stride = []int32(w,x,y,z)
	outputdims = []int32{stride[0]}
	//tensor dims a 1,4,4,2...


*/

//GetSizeInBytes returns the SizeT in bytes and Status
func (t *TensorD) GetSizeInBytes() (SizeT, error) {

	var sizebytes C.size_t
	x := C.cudnnGetTensorSizeInBytes(t.descriptor, &sizebytes)
	if setkeepalive == true {
		t.keepsalive()
	}
	return SizeT(sizebytes), Status(x).error("GetTensorNdDescriptor")
}

//IsDestroyed checks if the tensor is destroyed.  It will return a true if it is destroyed. If it is then this can be used again.

//DestroyDescriptor destroys the tensor
func (t *TensorD) DestroyDescriptor() error {

	return destroytensordescriptor(t)
}
func destroytensordescriptor(t *TensorD) error {
	return Status(C.cudnnDestroyTensorDescriptor(t.descriptor)).error("DestroyDescriptor")

}

//TensorFuncs is used to call functions for tensors usually the are functions that pass the Handle Type

//TransformTensor see below
/*
From the SDK Documentation:
This function copies the scaled data from one tensor to another tensor with a different layout.
Those descriptors need to have the same dimensions but not necessarily the same strides.
The input and output tensors must not overlap in any way (i.e., tensors cannot be transformed in place).
This function can be used to convert a tensor with an unsupported format to a supported one.

cudnnStatus_t cudnnTransformTensor(
    cudnnHandle_t                  handle,
    const void                    *alpha,
    const cudnnTensorDescriptor_t  xDesc,
    const void                    *x,
    const void                    *beta,
    const cudnnTensorDescriptor_t  yDesc,
	void                          *y)

y = Transfomr((alpha *x),(beta * y))
This will change the layout of a tensor stride wise
*/
func (t Tensor) TransformTensor(h *Handle, alpha CScalar, tx *TensorD, x *Malloced, beta CScalar, ty *TensorD, y *Malloced) error {

	var s Status

	s = Status(C.cudnnTransformTensor(h.x, alpha.CPtr(), tx.descriptor, x.Ptr(), beta.CPtr(), ty.descriptor, y.Ptr()))
	if setkeepalive == true {
		keepsalivebuffer(h, tx, x, ty, y)
	}

	return s.error("TransformTensor")
}

//AddTensor Tensor Bias addition : C = alpha * A + beta * C // c is both the input and output
/*From Documentation
This function adds the scaled values of a bias tensor to another tensor.
Each dimension of the bias tensor A must match the corresponding dimension of the destination tensor C or must be equal to 1.
In the latter case, the same value from the bias tensor for those dimensions will be used to blend into the C tensor.

**Note: Up to dimension 5, all tensor formats are supported. Beyond those dimensions, this routine is not supported
*/
func (t Tensor) AddTensor(h *Handle, alpha CScalar, aD *TensorD, A *Malloced, beta CScalar, cD *TensorD, c *Malloced) error {

	s := Status(C.cudnnAddTensor(h.x, alpha.CPtr(), aD.descriptor, A.Ptr(), beta.CPtr(), cD.descriptor, c.Ptr()))
	if setkeepalive == true {
		keepsalivebuffer(h, aD, A, cD, c)
	}

	return s.error("AddTensor")
}

//ScaleTensor - Scale all values of a tensor by a given factor : y[i] = alpha * y[i]
func (t Tensor) ScaleTensor(h *Handle, yD *TensorD, y *Malloced, alpha CScalar) error {
	keepsalivebuffer(h, yD, y)
	return Status(C.cudnnScaleTensor(h.x, yD.descriptor, y.Ptr(), alpha.CPtr())).error("ScaleTensor")
}

//SetTensor -  Set all values of a tensor to a given value : y[i] = value[0]
func (t Tensor) SetTensor(h *Handle, yD *TensorD, y *Malloced, v CScalar) error {

	x := C.cudnnSetTensor(h.x, yD.descriptor, y.Ptr(), v.CPtr())
	if setkeepalive == true {
		keepsalivebuffer(h, yD, y)
	}
	return Status(x).error("SetTensor")
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
		return CInt8(x)
	case DataType(C.CUDNN_DATA_INT32):
		return CInt(x)
	case DataType(C.CUDNN_DATA_HALF):
		return CHalf(half.NewFloat16(float32(x)))
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

// Half returns DataType(C.CUDNN_DATA_HALF)
func (d DataTypeFlag) Half() DataType {
	return DataType(C.CUDNN_DATA_HALF)
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

/*
//SlideMethod will be used pretty much only as a backend.
func (t TensorFormatFlag)SlideMethod() TensorFormat{
	return TensorFormat(C.cudnnTensorFormat_t(255))
}
*/
func (t TensorFormat) c() C.cudnnTensorFormat_t { return C.cudnnTensorFormat_t(t) }
