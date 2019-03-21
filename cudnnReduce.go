package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//Reduce holds Reduce flags and funcs also used to access create reduce tensor function
type Reduce struct {
	Flgs ReduceFlags
}

//ReduceTensorD is the struct that is used for reduce tensor ops
type ReduceTensorD struct {
	tensorDesc        C.cudnnReduceTensorDescriptor_t
	tensorOp          C.cudnnReduceTensorOp_t
	tensorCompType    C.cudnnDataType_t
	tensorNanOpt      C.cudnnNanPropagation_t
	tensorIndices     C.cudnnReduceTensorIndices_t
	tensorIndicesType C.cudnnIndicesType_t
}

func (reduce *ReduceTensorD) keepsalive() {
	runtime.KeepAlive(reduce)
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

//NewReduceTensorDescriptor creates and sets a reduce tensor Descriptor
func (red Reduce) NewReduceTensorDescriptor(
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
		runtime.SetFinalizer(descriptor, destroyreducetensordescriptor)
	}
	return descriptor, err
}

//SetReduceTensorDescriptor Sets the reduce tensor Descriptor
func (reduce *ReduceTensorD) setReduceTensorDescriptor() error {

	x := C.cudnnSetReduceTensorDescriptor(reduce.tensorDesc, reduce.tensorOp, reduce.tensorCompType, reduce.tensorNanOpt, reduce.tensorIndices, reduce.tensorIndicesType)
	return Status(x).error("SetReduceTensorDescriptor")
}

/*
//GetReduceTensorDescriptor Gets a copy of reduce tensor descriptor
func (reduce *ReduceTensor) GetReduceTensorDescriptor() (ReduceTensor, error) {
	var reducex ReduceTensor
	reducex.tensorDesc = reduce.tensorDesc
	x := C.cudnnGetReduceTensorDescriptor(reducex.tensorDesc, &reducex.tensorOp, &reducex.tensorCompType, &reducex.tensorNanOpt, &reducex.tensorIndices, &reducex.tensorIndicesType)
	return reducex, Status(x).error("GetReduceTensorDescriptor")
}
*/

//Destroy destroys the reducetensordescriptor
func (reduce *ReduceTensorD) Destroy() error {
	return destroyreducetensordescriptor(reduce)
}
func destroyreducetensordescriptor(reduce *ReduceTensorD) error {
	x := C.cudnnDestroyReduceTensorDescriptor(reduce.tensorDesc)
	err := Status(x).error("DestroyTensorDescriptor")

	return err
}

/*IndiciesSize Helper function to return the minimum size in bytes of the index space to be passed to the reduction given the input and output tensors */
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

//ReduceFlags holds the flag holders that are used for reduce flags
type ReduceFlags struct {
	RedTenOp   ReduceTensorOpFlag
	RedTenIndc ReduceTensorIndicesFlag
	IndcType   IndiciesTypeFlag
}

//ReduceTensorOp used for flags for reduce tensor functions
type ReduceTensorOp C.cudnnReduceTensorOp_t

func (r ReduceTensorOp) c() C.cudnnReduceTensorOp_t {
	return C.cudnnReduceTensorOp_t(r)
}

//ReduceTensorOpFlag is used to pass ReduceTensorOp flags semi safely for users using methods
type ReduceTensorOpFlag struct {
}

//Add returns reduceTensorAdd flag
func (r ReduceTensorOpFlag) Add() ReduceTensorOp {
	return ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_ADD)
}

//Mul returns reduceTensorMul flag
func (r ReduceTensorOpFlag) Mul() ReduceTensorOp {
	return ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_MUL)
}

//Min returns reduceTensorMin flag
func (r ReduceTensorOpFlag) Min() ReduceTensorOp {
	return ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_MIN)
}

//Max returns reduceTensorMax flag
func (r ReduceTensorOpFlag) Max() ReduceTensorOp {
	return ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_MAX)
}

//Amax returns reduceTensorAmax flag
func (r ReduceTensorOpFlag) Amax() ReduceTensorOp {
	return ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_AMAX)
}

//Avg returns reduceTensorAvg flag
func (r ReduceTensorOpFlag) Avg() ReduceTensorOp {
	return ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_AVG)
}

//Norm1 returns reduceTensorNorm1 flag
func (r ReduceTensorOpFlag) Norm1() ReduceTensorOp {
	return ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_NORM1)
}

//Norm2 returns reduceTensorNorm2 flag
func (r ReduceTensorOpFlag) Norm2() ReduceTensorOp {
	return ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_NORM2)
}

//MulNoZeros returns reduceTensorMulNoZeros flag
func (r ReduceTensorOpFlag) MulNoZeros() ReduceTensorOp {
	return ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS)
}

//ReduceTensorIndices are used for flags
type ReduceTensorIndices C.cudnnReduceTensorIndices_t

//ReduceTensorIndicesFlag used to pass reduce tensor indices through methods
type ReduceTensorIndicesFlag struct {
}

//NoIndices returns reduceTensorNoIndices flag
func (r ReduceTensorIndicesFlag) NoIndices() ReduceTensorIndices {

	return ReduceTensorIndices(C.CUDNN_REDUCE_TENSOR_NO_INDICES)
}

//FlattenedIndicies returns reduceTensorFlattenedIndicies flag
func (r ReduceTensorIndicesFlag) FlattenedIndicies() ReduceTensorIndices {

	return ReduceTensorIndices(C.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES)
}

func (r ReduceTensorIndices) c() C.cudnnReduceTensorIndices_t {
	return C.cudnnReduceTensorIndices_t(r)
}

//IndiciesType are flags
type IndiciesType C.cudnnIndicesType_t

//IndiciesTypeFlag is used to pass IndiciesType flags through method
type IndiciesTypeFlag struct {
}

//Type32Bit returns  IndiciesType( C.CUDNN_32BIT_INDICES) flag
func (i IndiciesTypeFlag) Type32Bit() IndiciesType {
	return IndiciesType(C.CUDNN_32BIT_INDICES)
}

//Type64Bit returns  IndiciesType( C.CUDNN_64BIT_INDICES) flag
func (i IndiciesTypeFlag) Type64Bit() IndiciesType {
	return IndiciesType(C.CUDNN_64BIT_INDICES)
}

//Type16Bit returns IndiciesType( C.CUDNN_16BIT_INDICES) flag
func (i IndiciesTypeFlag) Type16Bit() IndiciesType {
	return IndiciesType(C.CUDNN_16BIT_INDICES)
}

//Type8Bit returns  IndiciesType( C.CUDNN_8BIT_INDICES) flag
func (i IndiciesTypeFlag) Type8Bit() IndiciesType {
	return IndiciesType(C.CUDNN_8BIT_INDICES)
}
func (i IndiciesType) c() C.cudnnIndicesType_t { return C.cudnnIndicesType_t(i) }
