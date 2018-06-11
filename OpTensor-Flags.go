package gocudnn

/*
#include <cudnn.h>
*/
import (
	"C"
)

//OpTensorFlag is used pass OpTensorOp Flags Semi Safely using methods
type OpTensorFlag struct {
}

//OpTensorOp is used for flags for the Optensor functions
type OpTensorOp C.cudnnOpTensorOp_t

//Add returns 	 OpTensorOp(C.CUDNN_OP_TENSOR_ADD)
func (o OpTensorFlag) Add() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_ADD)
}

//Mul returns  OpTensorOp(C.CUDNN_OP_TENSOR_MUL)
func (o OpTensorFlag) Mul() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_MUL)
}

//Min returns OpTensorOp(C.CUDNN_OP_TENSOR_MIN)
func (o OpTensorFlag) Min() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_MIN)
}

//Max returns OpTensorOp(C.CUDNN_OP_TENSOR_MAX)
func (o OpTensorFlag) Max() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_MAX)
}

//Sqrt returns OpTensorOp(C.CUDNN_OP_TENSOR_SQRT)
func (o OpTensorFlag) Sqrt() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_SQRT)
}

//Not returns OpTensorOp(C.CUDNN_OP_TENSOR_NOT)
func (o OpTensorFlag) Not() OpTensorOp {
	return OpTensorOp(C.CUDNN_OP_TENSOR_NOT)
}
func (o OpTensorOp) c() C.cudnnOpTensorOp_t { return C.cudnnOpTensorOp_t(o) }
