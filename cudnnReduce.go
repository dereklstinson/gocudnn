package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//ReduceTensorD is the struct that is used for reduce tensor ops
type ReduceTensorD struct {
	tensorDesc C.cudnnReduceTensorDescriptor_t
	//	tensorOp          C.cudnnReduceTensorOp_t
	//	tensorCompType    C.cudnnDataType_t
	//	tensorNanOpt      C.cudnnNanPropagation_t
	//	tensorIndices     C.cudnnReduceTensorIndices_t
	//	tensorIndicesType C.cudnnIndicesType_t
	gogc bool
}

//CreateReduceTensorDescriptor creates an empry Reduce Tensor Descriptor
func CreateReduceTensorDescriptor() (*ReduceTensorD, error) {
	rt := new(ReduceTensorD)
	err := Status(C.cudnnCreateReduceTensorDescriptor(&rt.tensorDesc)).error("CreateReduceTensorDescriptor-create")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		rt.gogc = true
		runtime.SetFinalizer(rt, cudnnDestroyReduceTensorDescriptor)
	}
	return rt, nil
}

//Set sets r with the values passed
func (r *ReduceTensorD) Set(reduceop ReduceTensorOp,
	datatype DataType,
	nanprop NANProp,
	reducetensorinds ReduceTensorIndices,
	indicietype IndiciesType) error {

	//r.tensorOp = reduceop.c()
	//	r.tensorCompType = datatype.c()
	//	r.tensorNanOpt = nanprop.c()
	//	r.tensorIndices = reducetensorinds.c()
	//	r.tensorIndicesType = indicietype.c()
	return Status(C.cudnnSetReduceTensorDescriptor(r.tensorDesc, reduceop.c(), datatype.c(), nanprop.c(), reducetensorinds.c(), indicietype.c())).error("SetReduceTensorDescriptor")
}

//Get values that were set for r in set
func (r *ReduceTensorD) Get() (reduceop ReduceTensorOp,
	datatype DataType,
	nanprop NANProp,
	reducetensorinds ReduceTensorIndices,
	indicietype IndiciesType, err error) {
	err = Status(C.cudnnGetReduceTensorDescriptor(r.tensorDesc, reduceop.cptr(), datatype.cptr(), nanprop.cptr(), reducetensorinds.cptr(), indicietype.cptr())).error("SetReduceTensorDescriptor")
	return reduceop, datatype, nanprop, reducetensorinds, indicietype, err
}

//String satisfies stringer interface
func (r *ReduceTensorD) String() string {
	op, dtype, nanprop, indc, indctype, err := r.Get()
	if err != nil {
		return fmt.Sprintf("ReduceTensorD{\nError:%v,\n}\n", err)
	}
	return fmt.Sprintf("ReduceTensorD{\n%v,\n%v,\n%v,\n%v,\n%v,\n}\n", op, dtype, nanprop, indc, indctype)
}

//Destroy destroys the reducetensordescriptor
func (r *ReduceTensorD) Destroy() error {
	if setfinalizer || r.gogc {
		return nil
	}
	return cudnnDestroyReduceTensorDescriptor(r)
}
func cudnnDestroyReduceTensorDescriptor(reduce *ReduceTensorD) error {
	x := C.cudnnDestroyReduceTensorDescriptor(reduce.tensorDesc)
	err := Status(x).error("DestroyTensorDescriptor")

	return err
}

/*GetIndiciesSize Helper function to return the minimum size in bytes of the index space to be passed to the reduction given the input and output tensors */
func (r *ReduceTensorD) GetIndiciesSize(
	handle *Handle,
	aDesc, cDesc *TensorD) (uint, error) {
	var sizeinbytes C.size_t
	x := C.cudnnGetReductionIndicesSize(handle.x, r.tensorDesc, aDesc.descriptor, cDesc.descriptor, &sizeinbytes)

	return uint(sizeinbytes), Status(x).error("GetReductionIndicesSize")

}

//GetWorkSpaceSize  Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors
func (r *ReduceTensorD) GetWorkSpaceSize(
	handle *Handle,
	aDesc, cDesc *TensorD) (uint, error) {
	var sizeinbytes C.size_t
	x := C.cudnnGetReductionWorkspaceSize(handle.x, r.tensorDesc, aDesc.descriptor, cDesc.descriptor, &sizeinbytes)

	return uint(sizeinbytes), Status(x).error("GetReductionWorkspaceSize")

}

//ReduceTensorOp Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
func (r *ReduceTensorD) ReduceTensorOp(
	handle *Handle,
	indices cutil.Mem,
	indiciessize uint,
	wspace cutil.Mem,
	wspacesize uint,
	alpha float64,
	aDesc *TensorD,
	A cutil.Mem,
	beta float64,
	cDesc *TensorD,
	Ce cutil.Mem) error {
	a := cscalarbydatatype(aDesc.dtype, alpha)
	b := cscalarbydatatype(cDesc.dtype, beta)
	var x C.cudnnStatus_t
	if indices == nil && wspace != nil {
		x = C.cudnnReduceTensor(handle.x, r.tensorDesc, nil,
			C.size_t(0), wspace.Ptr(), C.size_t(wspacesize),
			a.CPtr(), aDesc.descriptor, A.Ptr(), b.CPtr(), cDesc.descriptor, Ce.Ptr())
	} else if indices != nil && wspace == nil {
		x = C.cudnnReduceTensor(handle.x, r.tensorDesc, indices.Ptr(),
			C.size_t(indiciessize), nil, C.size_t(0),
			a.CPtr(), aDesc.descriptor, A.Ptr(), b.CPtr(), cDesc.descriptor, Ce.Ptr())

	} else if indices == nil && wspace == nil {
		x = C.cudnnReduceTensor(handle.x, r.tensorDesc, nil,
			C.size_t(0), nil, C.size_t(0),
			a.CPtr(), aDesc.descriptor, A.Ptr(), b.CPtr(), cDesc.descriptor, Ce.Ptr())

	} else {
		x = C.cudnnReduceTensor(handle.x, r.tensorDesc, indices.Ptr(),
			C.size_t(indiciessize), wspace.Ptr(), C.size_t(wspacesize),
			a.CPtr(), aDesc.descriptor, A.Ptr(), b.CPtr(), cDesc.descriptor, Ce.Ptr())

	}

	return Status(x).error("ReduceTensor")
}

//ReduceTensorOpUS is like ReduceTensorOp but uses unsafe.Pointer instead of cutil.Mem
func (r *ReduceTensorD) ReduceTensorOpUS(
	handle *Handle,
	indices unsafe.Pointer, indiciessize uint,
	wspace unsafe.Pointer, wspacesize uint,
	alpha float64,
	aDesc *TensorD, A unsafe.Pointer,
	beta float64,
	cDesc *TensorD, Ce unsafe.Pointer) error {
	a := cscalarbydatatype(aDesc.dtype, alpha)
	b := cscalarbydatatype(cDesc.dtype, beta)
	var x C.cudnnStatus_t

	x = C.cudnnReduceTensor(handle.x, r.tensorDesc,
		indices, C.size_t(indiciessize),
		wspace, C.size_t(wspacesize),
		a.CPtr(),
		aDesc.descriptor, A,
		b.CPtr(),
		cDesc.descriptor, Ce)

	return Status(x).error("ReduceTensor")
}

//ReduceTensorOp used for flags for reduce tensor functions
type ReduceTensorOp C.cudnnReduceTensorOp_t

func (r ReduceTensorOp) c() C.cudnnReduceTensorOp_t {
	return C.cudnnReduceTensorOp_t(r)
}
func (r *ReduceTensorOp) cptr() *C.cudnnReduceTensorOp_t {
	return (*C.cudnnReduceTensorOp_t)(r)
}

//Add sets r to and returns reduceTensorAdd flag
func (r *ReduceTensorOp) Add() ReduceTensorOp {
	*r = ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_ADD)
	return *r
}

//Mul sets r to and returns reduceTensorMul flag
func (r *ReduceTensorOp) Mul() ReduceTensorOp {
	*r = ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_MUL)
	return *r
}

//Min sets r to and returns reduceTensorMin flag
func (r *ReduceTensorOp) Min() ReduceTensorOp {
	*r = ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_MIN)
	return *r
}

//Max sets r to and returns reduceTensorMax flag
func (r *ReduceTensorOp) Max() ReduceTensorOp {
	*r = ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_MAX)
	return *r
}

//Amax sets r to and returns reduceTensorAmax flag
func (r *ReduceTensorOp) Amax() ReduceTensorOp {
	*r = ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_AMAX)
	return *r
}

//Avg sets r to and returns reduceTensorAvg flag
func (r *ReduceTensorOp) Avg() ReduceTensorOp {
	*r = ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_AVG)
	return *r
}

//Norm1 sets r to and returns reduceTensorNorm1 flag
func (r *ReduceTensorOp) Norm1() ReduceTensorOp {
	*r = ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_NORM1)
	return *r
}

//Norm2 sets r to and returns reduceTensorNorm2 flag
func (r *ReduceTensorOp) Norm2() ReduceTensorOp {
	*r = ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_NORM2)
	return *r
}

//MulNoZeros sets r to and returns reduceTensorMulNoZeros flag
func (r *ReduceTensorOp) MulNoZeros() ReduceTensorOp {
	*r = ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS)
	return *r
}

//String satisfies stringer interface
func (r ReduceTensorOp) String() string {
	var x string
	f := r
	switch r {
	case f.Add():
		x = "Add"
	case f.Amax():
		x = "Amax"
	case f.Avg():
		x = "Avg"
	case f.Max():
		x = "Max"
	case f.Min():
		x = "Min"
	case f.Mul():
		x = "Mul"
	case f.MulNoZeros():
		x = "MulNoZeros"
	case f.Norm1():
		x = "Norm1"
	case f.Norm2():
		x = "Norm2"
	default:
		x = "Unsupported Flag"
	}
	return "ReduceTensorOp: " + x
}

//ReduceTensorIndices are used for flags exposed by type's methods
type ReduceTensorIndices C.cudnnReduceTensorIndices_t

//NoIndices sets r to and returns  ReduceTensorIndices(C.CUDNN_REDUCE_TENSOR_NO_INDICES)
func (r *ReduceTensorIndices) NoIndices() ReduceTensorIndices {
	*r = ReduceTensorIndices(C.CUDNN_REDUCE_TENSOR_NO_INDICES)
	return *r
}

//FlattenedIndicies sets r to and returns  ReduceTensorIndices(C.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES)
func (r *ReduceTensorIndices) FlattenedIndicies() ReduceTensorIndices {
	*r = ReduceTensorIndices(C.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES)
	return *r
}

func (r ReduceTensorIndices) c() C.cudnnReduceTensorIndices_t {
	return C.cudnnReduceTensorIndices_t(r)
}
func (r *ReduceTensorIndices) cptr() *C.cudnnReduceTensorIndices_t {
	return (*C.cudnnReduceTensorIndices_t)(r)
}

//String satisfies stringer interface
func (r ReduceTensorIndices) String() string {
	var x string
	f := r
	switch r {
	case f.FlattenedIndicies():
		x = "FlattenedIndicies"
	case f.NoIndices():
		x = "NoIndices"
	default:
		x = "Unsupported Flag"
	}
	return "ReduceTensorIndices: " + x
}

//IndiciesType are flags
type IndiciesType C.cudnnIndicesType_t

//Type32Bit sets i to and returns IndiciesType( C.CUDNN_32BIT_INDICES) flag
func (i *IndiciesType) Type32Bit() IndiciesType { *i = IndiciesType(C.CUDNN_32BIT_INDICES); return *i }

//Type64Bit sets i to and returns  IndiciesType( C.CUDNN_64BIT_INDICES) flag
func (i *IndiciesType) Type64Bit() IndiciesType { *i = IndiciesType(C.CUDNN_64BIT_INDICES); return *i }

//Type16Bit sets i to and returns IndiciesType( C.CUDNN_16BIT_INDICES) flag
func (i *IndiciesType) Type16Bit() IndiciesType { *i = IndiciesType(C.CUDNN_16BIT_INDICES); return *i }

//Type8Bit sets i to and returns  IndiciesType( C.CUDNN_8BIT_INDICES) flag
func (i *IndiciesType) Type8Bit() IndiciesType      { *i = IndiciesType(C.CUDNN_8BIT_INDICES); return *i }
func (i IndiciesType) c() C.cudnnIndicesType_t      { return C.cudnnIndicesType_t(i) }
func (i *IndiciesType) cptr() *C.cudnnIndicesType_t { return (*C.cudnnIndicesType_t)(i) }

//String satisfies stringer interface
func (i IndiciesType) String() string {
	var x string
	f := i
	switch i {
	case f.Type16Bit():
		x = "Type16Bit"
	case f.Type32Bit():
		x = "Type32Bit"
	case f.Type64Bit():
		x = "Type64Bit"
	case f.Type8Bit():
		x = "Type8Bit"
	default:
		x = "Unsupported Flag"
	}
	return "IndiciesType: " + x

}

/*
//TensorOP returns the tensorop value for the ReduceTensor
func (r *ReduceTensorD) TensorOP() ReduceTensorOp { return ReduceTensorOp(r.tensorOp) }

//CompType returns the Datatype of the reducetensor
func (rrndArrayToCarray *ReduceTensorD) CompType() DataType { return DataType(r.tensorCompType) }

//NanOpt returns the Nan operation flag for the reduce tensor
func (r *ReduceTensorD) NanOpt() NANProp { return NANProp(r.tensorNanOpt) }

//Indices returns the indicies for the Reudce tensor
func (r *ReduceTensorD) Indices() ReduceTensorIndices {
	return ReduceTensorIndices(r.tensorIndices)
}

//IndicType returns the IndicieType flag
func (r *ReduceTensorD) IndicType() IndiciesType { return IndiciesType(r.tensorIndicesType) }
*/
