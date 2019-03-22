package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//ReduceTensorD is the struct that is used for reduce tensor ops
type ReduceTensorD struct {
	tensorDesc        C.cudnnReduceTensorDescriptor_t
	tensorOp          C.cudnnReduceTensorOp_t
	tensorCompType    C.cudnnDataType_t
	tensorNanOpt      C.cudnnNanPropagation_t
	tensorIndices     C.cudnnReduceTensorIndices_t
	tensorIndicesType C.cudnnIndicesType_t
	gogc              bool
}

//TensorOP returns the tensorop value for the ReduceTensor
func (reduce *ReduceTensorD) TensorOP() ReduceTensorOp { return ReduceTensorOp(reduce.tensorOp) }

//CompType returns the Datatype of the reducetensor
func (reduce *ReduceTensorD) CompType() DataType { return DataType(reduce.tensorCompType) }

//NanOpt returns the Nan operation flag for the reduce tensor
func (reduce *ReduceTensorD) NanOpt() NANProp { return NANProp(reduce.tensorNanOpt) }

//Indices returns the indicies for the Reudce tensor
func (reduce *ReduceTensorD) Indices() ReduceTensorIndices {
	return ReduceTensorIndices(reduce.tensorIndices)
}

//IndicType returns the IndicieType flag
func (reduce *ReduceTensorD) IndicType() IndiciesType { return IndiciesType(reduce.tensorIndicesType) }

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
func (r *ReduceTensorD) Set(reduceop ReduceTensorOp,
	datatype DataType,
	nanprop NANProp,
	reducetensorinds ReduceTensorIndices,
	indicietype IndiciesType) error {
	return Status(C.cudnnSetReduceTensorDescriptor(r.tensorDesc, reduceop.c(), datatype.c(), nanprop.c(), reducetensorinds.c(), indicietype.c())).error("SetReduceTensorDescriptor")
}

/*
//NewReduceTensorDescriptor creates and sets a reduce tensor Descriptor
func NewReduceTensorDescriptor(
	reduceop ReduceTensorOp,
	datatype DataType,
	nanprop NANProp,
	reducetensorinds ReduceTensorIndices,
	indicietype IndiciesType) (descriptor *ReduceTensorD, err error) {
	//	var reduce ReduceTensorD
	var rtensdesc C.cudnnReduceTensorDescriptor_t
	err = Status(C.cudnnCreateReduceTensorDescriptor(&rtensdesc)).error("CreateReduceTensorDescriptor-create")
	if err != nil {
		return nil, err
	}
	descriptor = &ReduceTensorD{
		tensorDesc:        rtensdesc,
		tensorOp:          reduceop.c(),
		tensorCompType:    datatype.c(),
		tensorNanOpt:      nanprop.c(),
		tensorIndices:     reducetensorinds.c(),
		tensorIndicesType: indicietype.c(),
	}
	err = Status(C.cudnnSetReduceTensorDescriptor(rtensdesc, reduceop.c(), datatype.c(), nanprop.c(), reducetensorinds.c(), indicietype.c())).error("SetReduceTensorDescriptor")
	if setfinalizer == true {
		runtime.SetFinalizer(descriptor, cudnnDestroyReduceTensorDescriptor)
	}
	return descriptor, err
}
*/
//SetReduceTensorDescriptor Sets the reduce tensor Descriptor
func (reduce *ReduceTensorD) setReduceTensorDescriptor() error {

	x := C.cudnnSetReduceTensorDescriptor(reduce.tensorDesc, reduce.tensorOp, reduce.tensorCompType, reduce.tensorNanOpt, reduce.tensorIndices, reduce.tensorIndicesType)
	return Status(x).error("SetReduceTensorDescriptor")
}

/*
//GetReduceTensorDescriptor Gets a copy of reduce tensor descriptor
func (reduce *ReduceTensorD) GetReduceTensorDescriptor() (reduceop ReduceTensorOp,
	datatype DataType,
	nanprop NANProp,
	reducetensorinds ReduceTensorIndices,
	indicietype IndiciesType) {

	reducex.tensorDesc = reduce.tensorDesc
	x := C.cudnnGetReduceTensorDescriptor(reduce.tensorDesc, &reducex.tensorOp, &reducex.tensorCompType, &reducex.tensorNanOpt, &reducex.tensorIndices, &reducex.tensorIndicesType)
	return reducex, Status(x).error("GetReduceTensorDescriptor")
}
*/

//Destroy destroys the reducetensordescriptor
func (reduce *ReduceTensorD) Destroy() error {
	if setfinalizer || reduce.gogc {
		return nil
	}
	return cudnnDestroyReduceTensorDescriptor(reduce)
}
func cudnnDestroyReduceTensorDescriptor(reduce *ReduceTensorD) error {
	x := C.cudnnDestroyReduceTensorDescriptor(reduce.tensorDesc)
	err := Status(x).error("DestroyTensorDescriptor")

	return err
}

/*IndiciesSize Helper function to return the minimum size in bytes of the index space to be passed to the reduction given the input and output tensors */
/*
func (reduce *ReduceTensorD) IndiciesSize(
	handle *Handle,
	aDesc, cDesc *TensorD) (uint, error) {
	var sizeinbytes C.size_t
	x := C.cudnnGetReductionIndicesSize(handle.x, reduce.tensorDesc, aDesc.descriptor, cDesc.descriptor, &sizeinbytes)
	if setkeepalive == true {
		keepsalivebuffer(reduce, handle, aDesc, cDesc)
	}
	return uint(sizeinbytes), Status(x).error("GetReductionIndicesSize")

}
*/

//GetWorkSpaceSize  Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors
func (reduce *ReduceTensorD) GetWorkSpaceSize(
	handle *Handle,
	aDesc, cDesc *TensorD) (uint, error) {
	var sizeinbytes C.size_t
	x := C.cudnnGetReductionWorkspaceSize(handle.x, reduce.tensorDesc, aDesc.descriptor, cDesc.descriptor, &sizeinbytes)
	if setkeepalive == true {
		keepsalivebuffer(reduce, handle, aDesc, cDesc)
	}
	return uint(sizeinbytes), Status(x).error("GetReductionWorkspaceSize")

}

//ReduceTensorOp Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
func (reduce *ReduceTensorD) ReduceTensorOp(
	handle *Handle,
	indices gocu.Mem,
	indiciessize uint,
	wspace gocu.Mem,
	wspacesize uint,
	alpha float64,
	aDesc *TensorD,
	A gocu.Mem,
	beta float64,
	cDesc *TensorD,
	Ce gocu.Mem) error {
	a := cscalarbydatatype(aDesc.dtype, alpha)
	b := cscalarbydatatype(cDesc.dtype, beta)
	var x C.cudnnStatus_t
	if indices == nil && wspace != nil {
		x = C.cudnnReduceTensor(handle.x, reduce.tensorDesc, nil,
			C.size_t(0), wspace.Ptr(), C.size_t(wspacesize),
			a.CPtr(), aDesc.descriptor, A.Ptr(), b.CPtr(), cDesc.descriptor, Ce.Ptr())
	} else if indices != nil && wspace == nil {
		x = C.cudnnReduceTensor(handle.x, reduce.tensorDesc, indices.Ptr(),
			C.size_t(indiciessize), nil, C.size_t(0),
			a.CPtr(), aDesc.descriptor, A.Ptr(), b.CPtr(), cDesc.descriptor, Ce.Ptr())

	} else if indices == nil && wspace == nil {
		x = C.cudnnReduceTensor(handle.x, reduce.tensorDesc, nil,
			C.size_t(0), nil, C.size_t(0),
			a.CPtr(), aDesc.descriptor, A.Ptr(), b.CPtr(), cDesc.descriptor, Ce.Ptr())

	} else {
		x = C.cudnnReduceTensor(handle.x, reduce.tensorDesc, indices.Ptr(),
			C.size_t(indiciessize), wspace.Ptr(), C.size_t(wspacesize),
			a.CPtr(), aDesc.descriptor, A.Ptr(), b.CPtr(), cDesc.descriptor, Ce.Ptr())

	}

	return Status(x).error("ReduceTensor")
}

//ReduceTensorOp used for flags for reduce tensor functions
type ReduceTensorOp C.cudnnReduceTensorOp_t

func (r ReduceTensorOp) c() C.cudnnReduceTensorOp_t {
	return C.cudnnReduceTensorOp_t(r)
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

//IndiciesType are flags
type IndiciesType C.cudnnIndicesType_t

//Type32Bit sets i to and returns IndiciesType( C.CUDNN_32BIT_INDICES) flag
func (i *IndiciesType) Type32Bit() IndiciesType { *i = IndiciesType(C.CUDNN_32BIT_INDICES); return *i }

//Type64Bit sets i to and returns  IndiciesType( C.CUDNN_64BIT_INDICES) flag
func (i *IndiciesType) Type64Bit() IndiciesType { *i = IndiciesType(C.CUDNN_64BIT_INDICES); return *i }

//Type16Bit sets i to and returns IndiciesType( C.CUDNN_16BIT_INDICES) flag
func (i *IndiciesType) Type16Bit() IndiciesType { *i = IndiciesType(C.CUDNN_16BIT_INDICES); return *i }

//Type8Bit sets i to and returns  IndiciesType( C.CUDNN_8BIT_INDICES) flag
func (i *IndiciesType) Type8Bit() IndiciesType { *i = IndiciesType(C.CUDNN_8BIT_INDICES); return *i }
func (i IndiciesType) c() C.cudnnIndicesType_t { return C.cudnnIndicesType_t(i) }
