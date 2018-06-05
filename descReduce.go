package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//ReduceTensorOp used for flags for reduce tensor functions
type ReduceTensorOp C.cudnnReduceTensorOp_t

//ReduceTensorOpFlag func for ReduceTensorOp flags it defaults with ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_ADD) can be changed with methods
func ReduceTensorOpFlag() ReduceTensorOp {
	return ReduceTensorOp(C.CUDNN_REDUCE_TENSOR_ADD)
}

// This will go away...eventually ... maybe.  I will pass the C.<flag> through methods (like above) instead of how it is done below
const (
	reduceTensorAdd        ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_ADD
	reduceTensorMul        ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MUL
	reduceTensorMin        ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MIN
	reduceTensorMax        ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MAX
	reduceTensorAmax       ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_AMAX
	reduceTensorAvg        ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_AVG
	reduceTensorNorm1      ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_NORM1
	reduceTensorNorm2      ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_NORM2
	reduceTensorMulNoZeros ReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS
)

//Add returns reduceTensorAdd flag
func (r ReduceTensorOp) Add() ReduceTensorOp {
	return reduceTensorAdd
}

//Mul returns reduceTensorMul flag
func (r ReduceTensorOp) Mul() ReduceTensorOp {
	return reduceTensorMul
}

//Min returns reduceTensorMin flag
func (r ReduceTensorOp) Min() ReduceTensorOp {
	return reduceTensorMin
}

//Max returns reduceTensorMax flag
func (r ReduceTensorOp) Max() ReduceTensorOp {
	return reduceTensorMax
}

//Amax returns reduceTensorAmax flag
func (r ReduceTensorOp) Amax() ReduceTensorOp {
	return reduceTensorAmax
}

//Avg returns reduceTensorAvg flag
func (r ReduceTensorOp) Avg() ReduceTensorOp {
	return reduceTensorAvg
}

//Norm1 returns reduceTensorNorm1 flag
func (r ReduceTensorOp) Norm1() ReduceTensorOp {
	return reduceTensorNorm1
}

//Norm2 returns reduceTensorNorm2 flag
func (r ReduceTensorOp) Norm2() ReduceTensorOp {
	return reduceTensorNorm2
}

//MulNoZeros returns reduceTensorMulNoZeros flag
func (r ReduceTensorOp) MulNoZeros() ReduceTensorOp {
	return reduceTensorMulNoZeros
}

//ReduceTensorIndices are used for flags
type ReduceTensorIndices C.cudnnReduceTensorIndices_t

//ReduceTensorIndicesFlag used to pass reduce tensor indices through methods it defaults with reduceTensorNoIndices
const ReduceTensorIndicesFlag ReduceTensorIndices = C.CUDNN_REDUCE_TENSOR_NO_INDICES

//NoIndices returns reduceTensorNoIndices flag
func (r ReduceTensorIndices) NoIndices() ReduceTensorIndices {

	return ReduceTensorIndices(C.CUDNN_REDUCE_TENSOR_NO_INDICES)
}

//FlattenedIndicies returns reduceTensorFlattenedIndicies flag
func (r ReduceTensorIndices) FlattenedIndicies() ReduceTensorIndices {

	return ReduceTensorIndices(C.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES)
}

func (r ReduceTensorIndices) c() C.cudnnReduceTensorIndices_t {
	return C.cudnnReduceTensorIndices_t(r)
}

//IndiciesType are flags
type IndiciesType C.cudnnIndicesType_t

//IndiciesTypeFlag is used to pass IndiciesType flags through methods defaults at indiciesType32Bit
func IndiciesTypeFlag() IndiciesType {
	return IndiciesType(C.CUDNN_32BIT_INDICES)
}

//Type32Bit returns  IndiciesType( C.CUDNN_32BIT_INDICES) flag
func (i IndiciesType) Type32Bit() IndiciesType {
	return IndiciesType(C.CUDNN_32BIT_INDICES)
}

//Type64Bit returns  IndiciesType( C.CUDNN_64BIT_INDICES) flag
func (i IndiciesType) Type64Bit() IndiciesType {
	return IndiciesType(C.CUDNN_64BIT_INDICES)
}

//Type16Bit returns IndiciesType( C.CUDNN_16BIT_INDICES) flag
func (i IndiciesType) Type16Bit() IndiciesType {
	return IndiciesType(C.CUDNN_16BIT_INDICES)
}

//Type8Bit returns  IndiciesType( C.CUDNN_8BIT_INDICES) flag
func (i IndiciesType) Type8Bit() IndiciesType {
	return IndiciesType(C.CUDNN_8BIT_INDICES)
}

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
