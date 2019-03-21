package gocudnn

/*

#include <cudnn.h>

*/
import "C"
import (
	"errors"
	"runtime"
	"strconv"

	"github.com/dereklstinson/GoCudnn/gocu"
)

func (m MathType) string() string {
	if m == MathType(C.CUDNN_DEFAULT_MATH) {
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
	setbyother bool
	frmt       TensorFormat
	flag       descflag
	gogc       bool
}

func tensorDArrayToC(input []*TensorD) []C.cudnnTensorDescriptor_t {
	descs := make([]C.cudnnTensorDescriptor_t, len(input))
	for i := 0; i < len(input); i++ {
		descs[i] = input[i].descriptor
	}
	return descs
}

func createtensordescriptor(setbyotheroperations, gogc bool) (*TensorD, error) {
	d := new(TensorD)
	err := Status(C.cudnnCreateTensorDescriptor(&d.descriptor)).error("NewTensor4dDescriptor-create")
	if setfinalizer || gogc {
		d.gogc = true
		runtime.SetFinalizer(d, destroytensordescriptor)
	}
	if err != nil {
		return nil, err
	}
	if setbyotheroperations {
		d.frmt.Unknown()
	}

	return d, nil
}

//CreateTensorDescriptor creates an empty tensor descriptor
func CreateTensorDescriptor() (*TensorD, error) {
	if setfinalizer {
		return createtensordescriptor(false, true)
	}
	return createtensordescriptor(false, false)

}

//NewTensor4dDescriptor Creates and Sets a Tensor 4d Descriptor.
func NewTensor4dDescriptor(data DataType, format TensorFormat, shape []int32) (*TensorD, error) {

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
	if setfinalizer {
		runtime.SetFinalizer(x, destroytensordescriptor)
	}

	return x, nil

}

//NewTensor4dDescriptorEx Creates and Sets A Tensor 4d Descriptor EX
func NewTensor4dDescriptorEx(data DataType, shape, stride []int32) (*TensorD, error) {
	if len(shape) != 4 || len(stride) != 4 {
		return nil, errors.New("len(shape) = " + strconv.Itoa(len(shape)) + " len(stride) = " + strconv.Itoa(len(stride)) + " .. both have to equal 4")
	}
	var tflg TensorFormat
	var descriptor C.cudnnTensorDescriptor_t
	err := Status(C.cudnnCreateTensorDescriptor(&descriptor)).error("NewTensor4dDescriptorEx-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetTensor4dDescriptorEx(descriptor, C.cudnnDataType_t(data), C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3]), C.int(stride[0]), C.int(stride[1]), C.int(stride[2]), C.int(stride[3]))).error("NewTensor4dDescriptorEX-set")
	if err != nil {
		return nil, err
	}
	x := &TensorD{descriptor: descriptor, dimsarray: shape, stride: stride, frmt: tflg.Strided(), dims: C.int(4), flag: t4dex}
	if setfinalizer == true {
		runtime.SetFinalizer(x, destroytensordescriptor)
	}

	return x, nil

}

//NewTensorNdDescriptor creates and sets an nd descriptor
func NewTensorNdDescriptor(data DataType, shape, stride []int32) (*TensorD, error) {
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
	var tflg TensorFormat
	x := &TensorD{descriptor: descriptor, dimsarray: shape, frmt: tflg.Strided(), stride: stride, dims: dims, flag: tnd}
	if setfinalizer == true {
		runtime.SetFinalizer(x, destroytensordescriptor)
	}
	return x, nil
}

//NewTensorNdDescriptorEx creates and sets an ND descriptor ex
func NewTensorNdDescriptorEx(format TensorFormat, data DataType, shape []int32) (*TensorD, error) {
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
	x := &TensorD{descriptor: descriptor, dimsarray: shape, frmt: format, stride: stride, dims: dims, flag: tndex}
	if setfinalizer == true {
		runtime.SetFinalizer(x, destroytensordescriptor)
	}
	return x, nil

}

//GetFormat returns the format of the tensor error will return if tensor supports slide//
func (t *TensorD) GetFormat() TensorFormat {
	if t.descriptor != nil {
		return t.frmt
	}
	return t.frmt.Unknown()

}

//GetDescrptor returns Data Type the Dims for shape and stride and error.  for Descriptors without stride it will still return junk info. so be mindful when you code.
func (t *TensorD) GetDescrptor() (frmt TensorFormat, dtype DataType, tshape []int32, tstride []int32, err error) {

	shape := make([]C.int, t.dims)
	stride := make([]C.int, t.dims)
	var data C.cudnnDataType_t
	if t.flag == t4d || t.flag == t4dex {
		x := C.cudnnGetTensor4dDescriptor(t.descriptor, &data, &shape[0], &shape[1], &shape[2], &shape[3], &stride[0], &stride[1], &stride[2], &stride[3])

		return t.frmt, DataType(data), cintToint32(shape), cintToint32(stride), Status(x).error("GetDescriptor")

	} else if t.flag == tnd || t.flag == tndex {
		var holder C.int
		x := C.cudnnGetTensorNdDescriptor(t.descriptor, t.dims, &data, &holder, &shape[0], &stride[0])

		return t.frmt, DataType(data), cintToint32(shape), cintToint32(stride), Status(x).error("GetDescriptor")
	} else {
		var holder C.int
		if Status(C.cudnnGetTensorNdDescriptor(t.descriptor, t.dims, &data, &holder, &shape[0], &stride[0])).error("Checking") != nil {
			if Status(C.cudnnGetTensor4dDescriptor(t.descriptor, &data, &shape[0], &shape[1], &shape[2], &shape[3], &stride[0], &stride[1], &stride[2], &stride[3])).error("Checking") != nil {
				return t.frmt, DataType(data), cintToint32(shape), cintToint32(stride), errors.New("Tripplecheckpoint Didn't work I don't know what this tensorD is")
			}
			return t.frmt, DataType(data), cintToint32(shape), cintToint32(stride), nil
		}

		return t.frmt, DataType(data), cintToint32(shape), cintToint32(stride), nil
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
func (t *TensorD) GetSizeInBytes() (uint, error) {

	var sizebytes C.size_t
	x := C.cudnnGetTensorSizeInBytes(t.descriptor, &sizebytes)

	return uint(sizebytes), Status(x).error("GetSizeInBytes")
}

//IsDestroyed checks if the tensor is destroyed.  It will return a true if it is destroyed. If it is then this can be used again.

//Destroy destroys the tensor.
//In future I am going to add a GC setting that will enable or disable the GC.
//When the GC is disabled It will allow the user more control over memory.
//right now it does nothing and returns nil
func (t *TensorD) Destroy() error {
	if t.gogc || setfinalizer {
		return nil
	}
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
func TransformTensor(h *Handle, alpha float64, tx *TensorD, x gocu.Mem, beta float64, ty *TensorD, y gocu.Mem) error {

	var s Status
	a := cscalarbydatatype(ty.dtype, alpha)
	b := cscalarbydatatype(ty.dtype, beta)
	s = Status(C.cudnnTransformTensor(h.x, a.CPtr(), tx.descriptor, x.Ptr(), b.CPtr(), ty.descriptor, y.Ptr()))

	return s.error("TransformTensor")
}

//AddTensor Tensor Bias addition : C = alpha * A + beta * C // c is both the input and output
/*From Documentation
This function adds the scaled values of a bias tensor to another tensor.
Each dimension of the bias tensor A must match the corresponding dimension of the destination tensor C or must be equal to 1.
In the latter case, the same value from the bias tensor for those dimensions will be used to blend into the C tensor.

**Note: Up to dimension 5, all tensor formats are supported. Beyond those dimensions, this routine is not supported
*/
func AddTensor(h *Handle, alpha float64, aD *TensorD, A gocu.Mem, beta float64, cD *TensorD, c gocu.Mem) error {
	a := cscalarbydatatype(aD.dtype, alpha)
	b := cscalarbydatatype(aD.dtype, beta)
	s := Status(C.cudnnAddTensor(h.x, a.CPtr(), aD.descriptor, A.Ptr(), b.CPtr(), cD.descriptor, c.Ptr()))

	return s.error("AddTensor")
}

//ScaleTensor - Scale all values of a tensor by a given factor : y[i] = alpha * y[i]
//
func ScaleTensor(h *Handle, yD *TensorD, y gocu.Mem, alpha float64) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	return Status(C.cudnnScaleTensor(h.x, yD.descriptor, y.Ptr(), a.CPtr())).error("ScaleTensor")
}

//SetTensor -  Set all values of a tensor to a given value : y[i] = value[0]
//v is type casted to the correct type within function
func SetTensor(h *Handle, yD *TensorD, y gocu.Mem, v float64) error {

	vc := cscalarbydatatypeforsettensor(yD.dtype, v)
	x := C.cudnnSetTensor(h.x, yD.descriptor, y.Ptr(), vc.CPtr())

	return Status(x).error("SetTensor")
}

/*
*
*
*       DataType
*
*
 */

//DataType is used for flags for the tensor layer structs
type DataType C.cudnnDataType_t

// Float sets d to DataType(C.CUDNN_DATA_FLOAT) and returns the changed value
func (d *DataType) Float() DataType { *d = DataType(C.CUDNN_DATA_FLOAT); return *d }

// Double sets d to DataType(C.CUDNN_DATA_DOUBLE) and returns the changed value
func (d *DataType) Double() DataType { *d = DataType(C.CUDNN_DATA_DOUBLE); return *d }

// Int8 sets d to DataType(C.CUDNN_DATA_INT8) and returns the changed value
func (d *DataType) Int8() DataType { *d = DataType(C.CUDNN_DATA_INT8); return *d }

// Int32 sets d to DataType(C.CUDNN_DATA_INT32) and returns the changed value
func (d *DataType) Int32() DataType { *d = DataType(C.CUDNN_DATA_INT32); return *d }

// Half sets d to DataType(C.CUDNN_DATA_HALF) and returns the changed value
func (d *DataType) Half() DataType { *d = DataType(C.CUDNN_DATA_HALF); return *d }

// UInt8 sets d to DataType(C.CUDNN_DATA_INT8) and returns the changed value
func (d *DataType) UInt8() DataType { *d = DataType(C.CUDNN_DATA_INT8); return *d }

//Int8x32 sets d to  DataType(C.CUDNN_DATA_INT8x32) and returns the changed value -- only supported by sm_72.
func (d *DataType) Int8x32() DataType { *d = DataType(C.CUDNN_DATA_INT8x32); return *d }

func (d DataType) c() C.cudnnDataType_t      { return C.cudnnDataType_t(d) }
func (d *DataType) cptr() *C.cudnnDataType_t { return (*C.cudnnDataType_t)(d) }

/*
*
*
*       MathTypeFlag
*
*
 */

//MathType are flags to set for cudnnMathType_t and can be called by types methods
type MathType C.cudnnMathType_t

//Default sets m to MathType(C.CUDNN_DEFAULT_MATH) and returns changed value
func (m *MathType) Default() MathType { *m = MathType(C.CUDNN_DEFAULT_MATH); return *m }

//TensorOpMath return MathType(C.CUDNN_TENSOR_OP_MATH)
func (m *MathType) TensorOpMath() MathType { *m = MathType(C.CUDNN_TENSOR_OP_MATH); return *m }

func (m MathType) c() C.cudnnMathType_t      { return (C.cudnnMathType_t)(m) }
func (m *MathType) cptr() *C.cudnnMathType_t { return (*C.cudnnMathType_t)(m) }

/*
*
*
*       PropagationNANFlag
*
*
 */

//NANProp is type for C.cudnnNanPropagation_t used for flags and are called and changed through type's methods
type NANProp C.cudnnNanPropagation_t

//NotPropigate sets p to PropagationNAN(C.CUDNN_NOT_PROPAGATE_NAN) and returns that value
func (p *NANProp) NotPropigate() NANProp { *p = NANProp(C.CUDNN_NOT_PROPAGATE_NAN); return *p }

//Propigate sets p to PropagationNAN(C.CUDNN_PROPAGATE_NAN) and returns that value
func (p *NANProp) Propigate() NANProp { *p = NANProp(C.CUDNN_PROPAGATE_NAN); return *p }

func (p NANProp) c() C.cudnnNanPropagation_t      { return C.cudnnNanPropagation_t(p) }
func (p *NANProp) cptr() *C.cudnnNanPropagation_t { return (*C.cudnnNanPropagation_t)(p) }

/*
*
*
*       Determinism
*
*
 */

//Determinism is the type for flags that set Determinism and are called and changed through type's methods
type Determinism C.cudnnDeterminism_t

func (d *Determinism) cptr() *C.cudnnDeterminism_t { return (*C.cudnnDeterminism_t)(d) }
func (d Determinism) c() C.cudnnDeterminism_t      { return C.cudnnDeterminism_t(d) }

//Non returns sets d to Determinism(C.CUDNN_NON_DETERMINISTIC) and returns the value
func (d *Determinism) Non() Determinism { *d = Determinism(C.CUDNN_NON_DETERMINISTIC); return *d }

//Deterministic sets d to Determinism(C.CUDNN_DETERMINISTIC) and returns the value
func (d *Determinism) Deterministic() Determinism { *d = Determinism(C.CUDNN_DETERMINISTIC); return *d }

func (d Determinism) string() string {
	if d == Determinism(C.CUDNN_NON_DETERMINISTIC) {
		return "Non Deterministic"
	}
	return "Deterministic "
}

//TensorFormat is the type used for flags to set tensor format.
//Type contains methods that change the value of the type.
//Caution: Methods will also change the value of variable that calls the method.
//		   If you need to make a case switch make another variable and call it flag and use that.  Look at ToString.
//
//Semi-Custom gocudnn flag.  NCHW,NHWC,NCHWvectC come from cudnn. GoCudnn adds Strided, and Unknown
//Reasonings --
//Strided - When the tensor is set with strides there is no TensorFormat flag passed.
//Also cudnnGetTensor4dDescriptor,and cudnnGetTensorNdDescriptor doesn't return the tensor format.
//Which is really annoying.  GoCudnn will hide this flag in TensorD so that it can be returned with the tensor.
//Unknown--Was was made because with at least with the new AttentionD in cudnn V7.5 it will make a descriptor for you.
//IDK what the tensor format will be. So lets not make an (ASSUME) and mark it with this.
type TensorFormat C.cudnnTensorFormat_t

//NCHW return TensorFormat(C.CUDNN_TENSOR_NCHW)
//Method sets type and returns new value.
func (t *TensorFormat) NCHW() TensorFormat {
	return TensorFormat(C.CUDNN_TENSOR_NCHW)
}

//NHWC return TensorFormat(C.CUDNN_TENSOR_NHWC)
//Method sets type and returns new value.
func (t *TensorFormat) NHWC() TensorFormat {
	return TensorFormat(C.CUDNN_TENSOR_NHWC)
}

//NCHWvectC return TensorFormat(C.CUDNN_TENSOR_NCHW_VECT_C)
//Method sets type and returns new value.
func (t *TensorFormat) NCHWvectC() TensorFormat {
	return TensorFormat(C.CUDNN_TENSOR_NCHW_VECT_C)
}

//Strided returns  TensorFormat(127) This is custom GoCudnn flag. Read TensorFormat notes for explanation.
//Method sets type and returns new value.
func (t *TensorFormat) Strided() TensorFormat {

	return TensorFormat(127)
}

//Unknown returns TensorFormat(128). This is custom GoCudnn flag.  Read TensorFormat notes for explanation.
//Method sets type and returns new value.
func (t *TensorFormat) Unknown() TensorFormat {
	return TensorFormat(128)
}

func (t TensorFormat) c() C.cudnnTensorFormat_t { return C.cudnnTensorFormat_t(t) }
func (t *TensorFormat) cptr() *C.cudnnTensorFormat_t {
	return (*C.cudnnTensorFormat_t)(t)
}

//ToString will return a human readable string that can be printed for debugging.
func (t TensorFormat) ToString() string {
	var flg TensorFormat
	switch t {
	case flg.NCHW():
		return "NCHW"
	case flg.NHWC():
		return "NHWC"
	case flg.NCHWvectC():
		return "NCHWvectC"
	case flg.Strided():
		return "Strided"
	case flg.Unknown():
		return "Unknown"

	}
	return "ERROR no such flag"
}
