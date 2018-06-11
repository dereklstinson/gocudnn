package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//CTCLossAlgo used to hold flags
type CTCLossAlgo C.cudnnCTCLossAlgo_t

//CTCLossAlgoFlag used to give a semi safe way of exporting CTCLossAlgo flags through methods
type CTCLossAlgoFlag struct {
}

//Deterministic returns CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_DETERMINISTIC)
func (c CTCLossAlgoFlag) Deterministic() CTCLossAlgo {
	return CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_DETERMINISTIC)
}

//NonDeterministic returns   CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC) Flag
func (c CTCLossAlgoFlag) NonDeterministic() CTCLossAlgo {
	return CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC)
}

func (c CTCLossAlgo) c() C.cudnnCTCLossAlgo_t {
	return C.cudnnCTCLossAlgo_t(c)
}
