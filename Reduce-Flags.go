package gocudnn

/*
#include <cudnn.h>
*/
import "C"

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
