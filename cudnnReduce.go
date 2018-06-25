package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//Reduce holds Reduce flags and funcs also used to access create reduce tensor function
type Reduce struct {
	Funcs ReduceFuncs
	Flgs  ReduceFlags
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

//TensorOP returns the tensorop value for the ReduceTensor
func (reduce *ReduceTensorD) TensorOP() ReduceTensorOp { return ReduceTensorOp(reduce.tensorOp) }

//CompType returns the Datatype of the reducetensor
func (reduce *ReduceTensorD) CompType() DataType { return DataType(reduce.tensorCompType) }

//NanOpt returns the Nan operation flag for the reduce tensor
func (reduce *ReduceTensorD) NanOpt() PropagationNAN { return PropagationNAN(reduce.tensorNanOpt) }

//Indices returns the indicies for the Reudce tensor
func (reduce *ReduceTensorD) Indices() ReduceTensorIndices {
	return ReduceTensorIndices(reduce.tensorIndices)
}

//IndicType returns the IndicieType flag
func (reduce *ReduceTensorD) IndicType() IndiciesType { return IndiciesType(reduce.tensorIndicesType) }

//CreateReduceTensorDescriptor creates the tensor discritper struct
func (red Reduce) CreateReduceTensorDescriptor(reduceop ReduceTensorOp, datatype DataType, nanprop PropagationNAN, reducetensorinds ReduceTensorIndices, indicietype IndiciesType) (ReduceTensorD, error) {
	var reduce ReduceTensorD
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

//DestroyReduceTensorDescriptor destroys the reducetensordescriptor
func (reduce *ReduceTensorD) DestroyReduceTensorDescriptor() error {
	x := C.cudnnDestroyReduceTensorDescriptor(reduce.tensorDesc)
	err := Status(x).error("DestroyTensorDescriptor")

	return err
}

//ReduceFuncs is a nil struct used to call Reduce functions
type ReduceFuncs struct {
}

/*GetReductionIndicesSize Helper function to return the minimum size in bytes of the index space to be passed to the reduction given the input and output tensors */
func (r ReduceFuncs) GetReductionIndicesSize(
	handle *Handle,
	reducer *ReduceTensorD,
	aDesc, cDesc *TensorD) (SizeT, error) {
	var sizeinbytes C.size_t
	x := C.cudnnGetReductionIndicesSize(handle.x, reducer.tensorDesc, aDesc.descriptor, cDesc.descriptor, &sizeinbytes)
	return SizeT(sizeinbytes), Status(x).error("GetReductionIndicesSize")

}

//GetReductionWorkspaceSize  Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors
func (r ReduceFuncs) GetReductionWorkspaceSize(
	handle *Handle,
	reducer *ReduceTensorD,
	aDesc, cDesc *TensorD) (SizeT, error) {
	var sizeinbytes C.size_t
	x := C.cudnnGetReductionWorkspaceSize(handle.x, reducer.tensorDesc, aDesc.descriptor, cDesc.descriptor, &sizeinbytes)
	return SizeT(sizeinbytes), Status(x).error("GetReductionWorkspaceSize")

}

//ReduceTensorOp Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
func (r ReduceFuncs) ReduceTensorOp(
	handle *Handle,
	data DataType,
	reducer *ReduceTensorD,
	indices,
	workspace Memer,
	alpha CScalar,
	aDesc *TensorD,
	A Memer,
	beta CScalar,
	cDesc *TensorD,
	Ce Memer) error {

	x := C.cudnnReduceTensor(handle.x, reducer.tensorDesc, indices.Ptr(),
		C.size_t(indices.ByteSize()), workspace.Ptr(), C.size_t(workspace.ByteSize()),
		alpha.CPtr(), aDesc.descriptor, A.Ptr(), beta.CPtr(), cDesc.descriptor, Ce.Ptr())
	return Status(x).error("ReduceTensor")
}

//SetTensor -  Set all values of a tensor to a given value : y[i] = value[0]
func (r ReduceFuncs) SetTensor(handle *Handle, data DataType, yDesc *TensorD, y Memer, v CScalar) error {

	x := C.cudnnSetTensor(handle.x, yDesc.descriptor, y.Ptr(), v.CPtr())
	return Status(x).error("SetTensor")
}

//ScaleTensor - Scale all values of a tensor by a given factor : y[i] = alpha * y[i]
func (r ReduceFuncs) ScaleTensor(handle *Handle, data DataType, yDesc *TensorD, y Memer, alpha CScalar) error {

	x := C.cudnnScaleTensor(handle.x, yDesc.descriptor, y.Ptr(), alpha.CPtr())
	return Status(x).error("ScaleTensor")
}

//ReduceFlags holds the flag holders that are used for reduce flags
type ReduceFlags struct {
	RedTenOp   ReduceTensorOpFlag
	RedTenIndc ReduceTensorIndicesFlag
	IndcType   IndiciesTypeFlag
}

//ReduceTensorOp used for flags for reduce tensor functions
type ReduceTensorOp C.cudnnReduceTensorOp_t

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
