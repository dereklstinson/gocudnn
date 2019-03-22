package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//RNNBiasMode handles bias flags for RNN. Flags are exposed through types methods
type RNNBiasMode C.cudnnRNNBiasMode_t

func (b RNNBiasMode) c() C.cudnnRNNBiasMode_t      { return (C.cudnnRNNBiasMode_t)(b) }
func (b *RNNBiasMode) cptr() *C.cudnnRNNBiasMode_t { return (*C.cudnnRNNBiasMode_t)(b) }

//NoBias sets b to and returns RNNBiasMode(C.CUDNN_RNN_NO_BIAS)
func (b *RNNBiasMode) NoBias() RNNBiasMode { *b = RNNBiasMode(C.CUDNN_RNN_NO_BIAS); return *b }

//SingleINP sets b to and returns RNNBiasMode(C.CUDNN_RNN_SINGLE_INP_BIAS)
func (b *RNNBiasMode) SingleINP() RNNBiasMode {
	*b = RNNBiasMode(C.CUDNN_RNN_SINGLE_INP_BIAS)
	return *b
}

//Double sets b to and returns RNNBiasMode(C.CUDNN_RNN_DOUBLE_BIAS)
func (b *RNNBiasMode) Double() RNNBiasMode { *b = RNNBiasMode(C.CUDNN_RNN_DOUBLE_BIAS); return *b }

//SingleREC sets b to and returns RNNBiasMode(C.CUDNN_RNN_SINGLE_REC_BIAS)
func (b *RNNBiasMode) SingleREC() RNNBiasMode {
	*b = RNNBiasMode(C.CUDNN_RNN_SINGLE_REC_BIAS)
	return *b
}

//SetBiasMode sets the bias mode for descriptor
func (r *RNND) SetBiasMode(bmode RNNBiasMode) error {
	return Status(C.cudnnSetRNNBiasMode(r.descriptor, bmode.c())).error("(*RNND)SetBiasMode")
}

//GetBiasMode gets bias mode for descriptor
func (r *RNND) GetBiasMode() (bmode RNNBiasMode, err error) {
	err = Status(C.cudnnGetRNNBiasMode(r.descriptor, bmode.cptr())).error("(*RNND)GetBiasMode")
	return bmode, err
}
