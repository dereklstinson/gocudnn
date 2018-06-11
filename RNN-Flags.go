package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//RNNmode is used for flags use RNNModeFlag to pass them through methods
type RNNmode C.cudnnRNNMode_t

//RNNModeFlag is used to pass RNNMode flags semi safely through methods.
type RNNModeFlag struct {
}

func (r RNNmode) c() C.cudnnRNNMode_t { return C.cudnnRNNMode_t(r) }

//Relu return RNNMode(C.CUDNN_RNN_RELU)
func (r RNNModeFlag) Relu() RNNmode {
	return RNNmode(C.CUDNN_RNN_RELU)
}

//Tanh returns rnnTanh
func (r RNNModeFlag) Tanh() RNNmode {
	return RNNmode(C.CUDNN_RNN_TANH)
}

//Lstm returns rnnLstm
func (r RNNModeFlag) Lstm() RNNmode {
	return RNNmode(C.CUDNN_LSTM)
}

//Gru returns rnnGru
func (r RNNModeFlag) Gru() RNNmode {
	return RNNmode(C.CUDNN_GRU)
}

//DirectionModeFlag is used to pass DirectionModes through its methods.
type DirectionModeFlag struct {
}

//DirectionMode use DirectionModeFlag to pass them safe-ish.
type DirectionMode C.cudnnDirectionMode_t

func (r DirectionMode) c() C.cudnnDirectionMode_t { return C.cudnnDirectionMode_t(r) }

//Uni returns uniDirectional flag
func (r DirectionModeFlag) Uni() DirectionMode {
	return DirectionMode(C.CUDNN_UNIDIRECTIONAL)
}

//Bi returns biDirectional flag
func (r DirectionModeFlag) Bi() DirectionMode {
	return DirectionMode(C.CUDNN_BIDIRECTIONAL)
}

/*
 *   RNN INPUT MODE FLAGS
 */

//RNNInputModeFlag used to pass RNNInputMode Flags semi-safely through its methods.
type RNNInputModeFlag struct {
}

//RNNInputMode is used for flags
type RNNInputMode C.cudnnRNNInputMode_t

//Linear returns C.CUDNN_LINEAR_INPUT
func (r RNNInputModeFlag) Linear() RNNInputMode {
	return RNNInputMode(C.CUDNN_LINEAR_INPUT)
}

//Skip returns C.CUDNN_SKIP_INPUT
func (r RNNInputModeFlag) Skip() RNNInputMode {
	return RNNInputMode(C.CUDNN_SKIP_INPUT)
}
func (r RNNInputMode) c() C.cudnnRNNInputMode_t { return C.cudnnRNNInputMode_t(r) }

/*
 *   RNN ALGO FLAGS
 */

//RNNAlgoFlag used to pass RNNAlgo flags semi-safely.
type RNNAlgoFlag struct {
}

//RNNAlgo is used for flags
type RNNAlgo C.cudnnRNNAlgo_t

//Standard returns RNNAlgo( C.CUDNN_RNN_ALGO_STANDARD) flag
func (r RNNAlgoFlag) Standard() RNNAlgo {
	return RNNAlgo(C.CUDNN_RNN_ALGO_STANDARD)
}

//PersistStatic returns RNNAlgo( C.CUDNN_RNN_ALGO_PERSIST_STATIC) flag
func (r RNNAlgoFlag) PersistStatic() RNNAlgo {
	return RNNAlgo(C.CUDNN_RNN_ALGO_PERSIST_STATIC)
}

//PersistDynamic returns RNNAlgo( C.CUDNN_RNN_ALGO_PERSIST_DYNAMIC) flag
func (r RNNAlgoFlag) PersistDynamic() RNNAlgo {
	return RNNAlgo(C.CUDNN_RNN_ALGO_PERSIST_DYNAMIC)
}

//Count returns RNNAlgo( C.CUDNN_RNN_ALGO_COUNT) flag
func (r RNNAlgoFlag) Count() RNNAlgo {
	return RNNAlgo(C.CUDNN_RNN_ALGO_COUNT)
}

func (r RNNAlgo) c() C.cudnnRNNAlgo_t { return C.cudnnRNNAlgo_t(r) }
