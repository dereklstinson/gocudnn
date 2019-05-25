package gocudnn

/*

#include <cudnn.h>

*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
)

func (m MathType) string() string {
	if m == MathType(C.CUDNN_DEFAULT_MATH) {
		return "Math Type Default"
	}
	return "Math Type Tensor OP"
}

//TensorD holds the cudnnTensorDescriptor. Which is basically the tensor itself
type TensorD struct {
	descriptor C.cudnnTensorDescriptor_t
	dims       C.int
	shape      []int32
	stride     []int32
	dtype      DataType
	setbyother bool
	frmt       TensorFormat
	fflag      TensorFormat
	gogc       bool
}

//Dims returns the shape of the tensor
func (t *TensorD) Dims() []int32 {
	return t.shape
}

//DataType returns the datatype of the tensor
func (t *TensorD) DataType() DataType {
	return t.dtype
}

//Format returns the tensor format
func (t *TensorD) Format() TensorFormat {
	return t.frmt
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
	if setbyotheroperations {
		d.frmt.Unknown()
	}
	err := Status(C.cudnnCreateTensorDescriptor(&d.descriptor)).error("NewTensor4dDescriptor-create")
	if err != nil {
		return nil, err
	}

	if setfinalizer || gogc {
		d.gogc = true
		runtime.SetFinalizer(d, destroytensordescriptor)
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

//Set sets the tensor accourding to the values passed.
//	Basic 4D formats:
//
//	NCHW:
//
//		  shape[0] = # of batches
//		  shape[1] = # of channels
//		  shape[2] = height
//		  shape[3] = width
//
//	NHWC:
//
//		  shape[0] = # of batches
//		  shape[1] = height
//		  shape[2] = width
//		  shape[3] = # of channels
//
//	Strided:
//
//	Strided is kind of hard to explain.  So here is an example of how values would be placed.
//	n, c, h, w := 3,3,256,256 //Here is a batch of 3 images using rgb the size of 256x256
//	dims := []int{n, c, h, w}  // Here we have the dims set.
//	chw := c * h * w
//	hw := h * w
//	stride := []int{chw, hw, w, 1}  //This is how stride is usually set.
//  //If you wanted to get or place a value at a certain location.
//	//Such as:
//	//func GetValue(tensor []float32, location [4]int, stride [4]int){
//	//l,s:=location,stride
//	//return tensor[(l[0]*s[0])+(l[1]*s[1])+(l[2]*s[2])+(l[3]*s[3])] //As you can see the stride changes where you look in the tensor.
//	//}
//
//	Notes:
//
//	1) The total size of a tensor including the potential padding between dimensions is limited to 2 Giga-elements of type datatype.
//	   Tensors are restricted to having at least 4 dimensions, and at most DimMax (a const with val of 8 at the time of writing this) dimensions.
//     When working with lower dimensional data, it is recommended that the user create a 4D tensor, and set the size along unused dimensions to 1.
//	2) Stride is ignored if frmt is set to frmt.Strided(). So it can be set to nil.
func (t *TensorD) Set(frmt TensorFormat, data DataType, shape, stride []int32) error {
	t.frmt = frmt
	t.shape = shape

	t.dims = (C.int)(len(shape))
	t.dtype = data
	switch t.frmt {
	case t.fflag.Strided():
		t.stride = stride
		shapecint := int32Tocint(shape)
		stridecint := int32Tocint(stride)
		return Status(C.cudnnSetTensorNdDescriptor(t.descriptor, C.cudnnDataType_t(data), t.dims, &shapecint[0], &stridecint[0])).error("cudnnSetTensorNdDescriptor")

	default:
		t.stride = stridecalc(shape)
		shapecint := int32Tocint(shape)
		if len(shape) == 0 {
			panic("format is: " + t.frmt.ToString() + " and len is zero")
		}
		return Status(C.cudnnSetTensorNdDescriptorEx(t.descriptor, t.frmt.c(), data.c(), t.dims, &shapecint[0])).error("cudnnSetTensorNdDescriptorEx-set")

	}

}

//Get returns Data Type the Dims for shape and stride and error.  for Descriptors without stride it will still return junk info. so be mindful when you code.
func (t *TensorD) Get() (frmt TensorFormat, dtype DataType, shape []int32, stride []int32, err error) {
	if t.dims == 0 {
		t.dims = C.CUDNN_DIM_MAX
		shapec := make([]C.int, t.dims)
		stridec := make([]C.int, t.dims)
		frmt = t.frmt
		var actual C.int
		err = Status(C.cudnnGetTensorNdDescriptor(t.descriptor, t.dims, dtype.cptr(), &actual, &shapec[0], &stridec[0])).error("cudnnSetTensorNdDescriptor")
		t.dims = actual
		shape = cintToint32(shapec[:t.dims])
		stride = cintToint32(stridec[:t.dims])
		return frmt, dtype, shape, stride, err
	}

	shapec := make([]C.int, t.dims)
	stridec := make([]C.int, t.dims)
	frmt = t.frmt
	var actual C.int
	err = Status(C.cudnnGetTensorNdDescriptor(t.descriptor, t.dims, dtype.cptr(), &actual, &shapec[0], &stridec[0])).error("cudnnSetTensorNdDescriptor")
	if t.dims != actual {
		panic("t.dims should be actual")
	}
	shape = cintToint32(shapec)
	stride = cintToint32(stridec)
	return frmt, dtype, shape, stride, err
}

/*

	n, c, h, w := int32(1), int32(3), int32(4), int32(2)
	sharedims := []int32{n, c, h, w}
	//tensor dims a 1,4,4,2... stride is 32,8,2,1
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
func TransformTensor(h *Handle,
	alpha float64,
	xD *TensorD, x gocu.Mem,
	beta float64,
	yD *TensorD, y gocu.Mem) error {

	var s Status
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	s = Status(C.cudnnTransformTensor(h.x, a.CPtr(), xD.descriptor, x.Ptr(), b.CPtr(), yD.descriptor, y.Ptr()))

	return s.error("TransformTensor")
}

//TransformTensorUS is like TransformTensor but it uses unsafe.Pointer instead of gocu.Mem
func TransformTensorUS(h *Handle, alpha float64, xD *TensorD, x unsafe.Pointer, beta float64, yD *TensorD, y unsafe.Pointer) error {

	var s Status
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	s = Status(C.cudnnTransformTensor(h.x, a.CPtr(), xD.descriptor, x, b.CPtr(), yD.descriptor, y))

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

//AddTensorUS is like AddTensor but uses unsafe.Pointer instead of gocu.Mem
func AddTensorUS(h *Handle, alpha float64, aD *TensorD, A unsafe.Pointer, beta float64, cD *TensorD, c unsafe.Pointer) error {
	a := cscalarbydatatype(aD.dtype, alpha)
	b := cscalarbydatatype(aD.dtype, beta)
	s := Status(C.cudnnAddTensor(h.x, a.CPtr(), aD.descriptor, A, b.CPtr(), cD.descriptor, c))

	return s.error("AddTensor")
}

//ScaleTensor - Scale all values of a tensor by a given factor : y[i] = alpha * y[i]
func ScaleTensor(h *Handle, yD *TensorD, y gocu.Mem, alpha float64) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	return Status(C.cudnnScaleTensor(h.x, yD.descriptor, y.Ptr(), a.CPtr())).error("ScaleTensor")
}

//ScaleTensorUS is like ScaleTensor but it uses unsafe.Pointer instead of gocu.Mem
func ScaleTensorUS(h *Handle, yD *TensorD, y unsafe.Pointer, alpha float64) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	return Status(C.cudnnScaleTensor(h.x, yD.descriptor, y, a.CPtr())).error("ScaleTensor")
}

//SetTensor -  Set all values of a tensor to a given value : y[i] = value[0]
func SetTensor(h *Handle, yD *TensorD, y gocu.Mem, v float64) error {

	vc := cscalarbydatatypeforsettensor(yD.dtype, v)
	x := C.cudnnSetTensor(h.x, yD.descriptor, y.Ptr(), vc.CPtr())

	return Status(x).error("SetTensor")
}

//SetTensorUS is like SetTensor but it uses unsafe.Pointer instead of gocu.Mem
func SetTensorUS(h *Handle, yD *TensorD, y unsafe.Pointer, v float64) error {

	vc := cscalarbydatatypeforsettensor(yD.dtype, v)
	x := C.cudnnSetTensor(h.x, yD.descriptor, y, vc.CPtr())

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

//Int8x32 sets d to  DataType(C.CUDNN_DATA_INT8x32) and returns the changed value -- only supported by sm_72?.
func (d *DataType) Int8x32() DataType { *d = DataType(C.CUDNN_DATA_INT8x32); return *d }

//Int8x4 sets d to  DataType(C.CUDNN_DATA_INT8x4) and returns the changed value -- only supported by sm_72?.
func (d *DataType) Int8x4() DataType { *d = DataType(C.CUDNN_DATA_INT8x4); return *d }

//UInt8x4 sets d to  DataType(C.CUDNN_DATA_UINT8x4) and returns the changed value -- only supported by sm_72?.
func (d *DataType) UInt8x4() DataType { *d = DataType(C.CUDNN_DATA_UINT8x4); return *d }

func (d DataType) c() C.cudnnDataType_t      { return C.cudnnDataType_t(d) }
func (d *DataType) cptr() *C.cudnnDataType_t { return (*C.cudnnDataType_t)(d) }

//ToString will return a human readable string that can be printed for debugging.
func (d DataType) ToString() string {
	var flg DataType
	switch d {
	case flg.Float():
		return "Float"
	case flg.Double():
		return "Double"
	case flg.Int8():
		return "Int8"
	case flg.Int32():
		return "Int32"
	case flg.Half():
		return "Half"
	case flg.Int8x32():
		return "Int8x32"
	case flg.UInt8():
		return "UInt8"
	case flg.Int8x4():
		return "Int8x4"
	case flg.UInt8x4():
		return "UInt8x4"

	}
	return "ERROR no such flag"
}

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

//ToString outputs a string of the type
func (d Determinism) ToString() string {
	if d == Determinism(C.CUDNN_NON_DETERMINISTIC) {
		return "Non-Deterministic"
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

/*
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
*/

/*
//GetFormat returns the format of the tensor error will return if tensor supports slide//
func (t *TensorD) GetFormat() TensorFormat {
	if t.descriptor != nil {
		return t.frmt
	}
	return t.frmt.Unknown()

}
*/
