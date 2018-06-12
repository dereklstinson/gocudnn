package gocudnn

/*

#include <cudnn.h>

*/
import "C"

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
		return CInt(x)
	case DataType(C.CUDNN_DATA_INT32):
		return CInt(x)
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
func (t TensorFormat) c() C.cudnnTensorFormat_t { return C.cudnnTensorFormat_t(t) }
