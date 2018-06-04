package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//RNNMode is used for flags
type RNNMode C.cudnnRNNMode_t

func (r RNNMode) c() C.cudnnRNNMode_t { return C.cudnnRNNMode_t(r) }

//flags for RNNMode
const (
	RnnRelu RNNMode = C.CUDNN_RNN_RELU /* Stock RNN with ReLu activation */
	RnnTanh RNNMode = C.CUDNN_RNN_TANH /* Stock RNN with tanh activation */
	RnnLstm RNNMode = C.CUDNN_LSTM     /* LSTM with no peephole connections */
	RnnGru  RNNMode = C.CUDNN_GRU      /* Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1); */

)

//DirectionMode is a type used for flags
type DirectionMode C.cudnnDirectionMode_t

func (r DirectionMode) c() C.cudnnDirectionMode_t { return C.cudnnDirectionMode_t(r) }

//flags for DirectionMode
const (
	UniDirectional DirectionMode = C.CUDNN_UNIDIRECTIONAL
	BiDirectional  DirectionMode = C.CUDNN_BIDIRECTIONAL
)

//RNNInputMode is used for flags
type RNNInputMode C.cudnnRNNInputMode_t

func (r RNNInputMode) c() C.cudnnRNNInputMode_t { return C.cudnnRNNInputMode_t(r) }

//Flags for RNNInputMode
const (
	LinearInput RNNInputMode = C.CUDNN_LINEAR_INPUT
	SkipInput   RNNInputMode = C.CUDNN_SKIP_INPUT
)

//RNNAlgo is used for flags
type RNNAlgo C.cudnnRNNAlgo_t

//flags for RNNAlgo
const (
	RNNAlgoStandard       RNNAlgo = C.CUDNN_RNN_ALGO_STANDARD
	RNNAlgoPersistStatic  RNNAlgo = C.CUDNN_RNN_ALGO_PERSIST_STATIC
	RNNAlgoPersistDynamic RNNAlgo = C.CUDNN_RNN_ALGO_PERSIST_DYNAMIC
	RNNAlgocCount         RNNAlgo = C.CUDNN_RNN_ALGO_COUNT
)

//AlgorithmD holds the C.cudnnAlgorithmDescriptor_t
type AlgorithmD struct {
	descriptor C.cudnnAlgorithmDescriptor_t
}

//AlgorithmPerformance go typed C.cudnnAlgorithmPerformance_t
type AlgorithmPerformance C.cudnnAlgorithmPerformance_t

//RNND  holdes Rnn descriptor
type RNND struct {
	descriptor C.cudnnRNNDescriptor_t
}

//CreateRNNDescriptor creates an RNND descriptor
func CreateRNNDescriptor() (*RNND, error) {
	var desc C.cudnnRNNDescriptor_t
	err := Status(C.cudnnCreateRNNDescriptor(&desc)).error("CreateRNNDescriptor")
	if err != nil {
		return nil, err
	}
	return &RNND{
		descriptor: desc,
	}, nil
}

func (r *RNND) DestroyDescriptor() error {
	return Status(C.cudnnDestroyRNNDescriptor(r.descriptor)).error("DestroyDescriptor-rnn")
}
