package cudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

//ReduceTensorOp used for flags for reduce tensor functions
type ReduceTensorOp C.cudnnReduceTensorOp_t

//Flags for ReduceTensorOp
const (
	ReduceTensorAdd        ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_ADD
	ReduceTensorMul        ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MUL
	ReduceTensorMin        ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MIN
	ReduceTensorMax        ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MAX
	ReduceTensorAmax       ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_AMAX
	ReduceTensorAvg        ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_AVG
	ReduceTensorNorm1      ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_NORM1
	ReduceTensorNorm2      ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_NORM2
	ReduceTensorMulNoZeros ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS
)

//ReduceTensorIndices are used for flags
type ReduceTensorIndices C.cudnnReduceTensorIndices_t

//flags for Reduce Indicies
const (
	ReduceTensorNoIndices         ReduceTensorIndices = C.CUDNN_REDUCE_TENSOR_NO_INDICES
	ReduceTensorFlattenedIndicies ReduceTensorIndices = C.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES
)

//IndiciesType are flags
type IndiciesType C.cudnnIndicesType_t

//Flags for Indicies Type
const (
	IndiciesType32Bit IndiciesType = C.CUDNN_32BIT_INDICES
	IndiciesType64Bit IndiciesType = C.CUDNN_64BIT_INDICES
	IndiciesType16Bit IndiciesType = C.CUDNN_16BIT_INDICES
	IndiciesType8Bit  IndiciesType = C.CUDNN_8BIT_INDICES
)

//ReduceTensor is the struct that is used for reduce tensor ops
type ReduceTensor struct {
	tensorDesc        C.cudnnReduceTensorDescriptor_t
	tensorOp          C.cudnnReduceTensorOp_t
	tensorCompType    C.cudnnDataType_t
	tensorNanOpt      C.cudnnNanPropagation_t
	tensorIndices     C.cudnnReduceTensorIndices_t
	tensorIndicesType C.cudnnIndicesType_t
}

//TensorOP returns the tensorop value for the ReduceTensor
func (reduce *ReduceTensor) TensorOP() ReduceTensorOp { return ReduceTensorOp(reduce.tensorOp) }

//CompType returns the Datatype of the reducetensor
func (reduce *ReduceTensor) CompType() DataType { return DataType(reduce.tensorCompType) }

//NanOpt returns the Nan operation flag for the reduce tensor
func (reduce *ReduceTensor) NanOpt() PropagationNAN { return PropagationNAN(reduce.tensorNanOpt) }

//Indices returns the indicies for the Reudce tensor
func (reduce *ReduceTensor) Indices() ReduceTensorIndices {
	return ReduceTensorIndices(reduce.tensorIndices)
}

//IndicType returns the IndicieType flag
func (reduce *ReduceTensor) IndicType() IndiciesType { return IndiciesType(reduce.tensorIndicesType) }

//CreateReduceTensorDescriptor creates the tensor discritper struct
func CreateReduceTensorDescriptor(reduceop ReduceTensorOp, datatype DataType, nanprop PropagationNAN, reducetensorinds ReduceTensorIndices, indicietype IndiciesType) (ReduceTensor, error) {
	var reduce ReduceTensor
	x := Status(C.cudnnCreateReduceTensorDescriptor(&reduce.tensorDesc)).error("CreateReduceTensorDescriptor-create")
	if x != nil {
		return reduce, x
	}
	reduce.tensorOp = C.cudnnReduceTensorOp_t(reduceop)
	reduce.tensorCompType = C.cudnnDataType_t(datatype)
	reduce.tensorNanOpt = C.cudnnNanPropagation_t(nanprop)
	reduce.tensorIndices = C.cudnnReduceTensorIndices_t(reducetensorinds)
	reduce.tensorIndicesType = C.cudnnIndicesType_t(indicietype)
	x = reduce.setReduceTensorDescriptor()
	return reduce, x
}

//SetReduceTensorDescriptor Sets the reduce tensor Descriptor
func (reduce *ReduceTensor) setReduceTensorDescriptor() error {

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

//DestroyReduceTensorDescriptor destroys the reducetensordescriptor
func (reduce *ReduceTensor) DestroyReduceTensorDescriptor() error {
	x := C.cudnnDestroyReduceTensorDescriptor(reduce.tensorDesc)
	err := Status(x).error("DestroyTensorDescriptor")

	return err
}

//Handler stuff

/*GetReductionIndicesSize Helper function to return the minimum size in bytes of the index space to be passed to the reduction given the input and output tensors */
func (handle *Handle) GetReductionIndicesSize(reducer *ReduceTensor, aDesc, cDesc *TensorD) (SizeT, error) {
	var sizeinbytes C.size_t
	x := C.cudnnGetReductionIndicesSize(handle.x, reducer.tensorDesc, aDesc.descriptor, cDesc.descriptor, &sizeinbytes)
	return SizeT(sizeinbytes), Status(x).error("GetReductionIndicesSize")

}

//GetReductionWorkspaceSize  Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors
func (handle *Handle) GetReductionWorkspaceSize(reducer *ReduceTensor, aDesc, cDesc *TensorD) (SizeT, error) {
	var sizeinbytes C.size_t
	x := C.cudnnGetReductionWorkspaceSize(handle.x, reducer.tensorDesc, aDesc.descriptor, cDesc.descriptor, &sizeinbytes)
	return SizeT(sizeinbytes), Status(x).error("GetReductionWorkspaceSize")

}

//ReduceTensorOp Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
func (handle *Handle) ReduceTensorOp(reducer *ReduceTensor, indices, workspace Memer,
	alpha float64, aDesc *TensorD, A Memer, beta float64, cDesc *TensorD, Ce Memer) error {
	var alphau, betau unsafe.Pointer
	if DataType(aDesc.data) != DataType(cDesc.data) {
		return errors.New("The Data Types Don't Match in the TransformTensor")
	}
	switch DataType(aDesc.data) {

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
	x := C.cudnnReduceTensor(handle.x, reducer.tensorDesc, indices.Ptr(),
		C.size_t(indices.ByteSize()), workspace.Ptr(), C.size_t(workspace.ByteSize()),
		alphau, aDesc.descriptor, A.Ptr(), betau, cDesc.descriptor, Ce.Ptr())
	return Status(x).error("ReduceTensor")
}

//SetTensor -  Set all values of a tensor to a given value : y[i] = value[0]
func (handle *Handle) SetTensor(yDesc TensorD, y Memer, v float64) error {
	var vu unsafe.Pointer

	switch DataType(yDesc.data) {

	case DataTypeInt32:
		b := C.int(v)
		vu = unsafe.Pointer(&b)
	case DataTypeFloat:
		b := C.float(v)
		vu = unsafe.Pointer(&b)
	case DataTypeDouble:
		b := C.double(v)
		vu = unsafe.Pointer(&b)
	default:
		return errors.New("Should have never reached this place we are in trouble")
	}
	x := C.cudnnSetTensor(handle.x, yDesc.descriptor, y.Ptr(), vu)
	return Status(x).error("SetTensor")
}

//ScaleTensor - Scale all values of a tensor by a given factor : y[i] = alpha * y[i]
func (handle *Handle) ScaleTensor(yDesc TensorD, y Memer, alpha float64) error {
	var vu unsafe.Pointer

	switch DataType(yDesc.data) {

	case DataTypeInt32:
		b := C.int(alpha)
		vu = unsafe.Pointer(&b)
	case DataTypeFloat:
		b := C.float(alpha)
		vu = unsafe.Pointer(&b)
	case DataTypeDouble:
		b := C.double(alpha)
		vu = unsafe.Pointer(&b)
	default:
		return errors.New("Should have never reached this place we are in trouble")
	}
	x := C.cudnnScaleTensor(handle.x, yDesc.descriptor, y.Ptr(), vu)
	return Status(x).error("ScaleTensor")
}
