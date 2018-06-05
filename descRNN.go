package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//RNNMode is used for flags
type RNNMode C.cudnnRNNMode_t

func (r RNNMode) c() C.cudnnRNNMode_t { return C.cudnnRNNMode_t(r) }

//RNNModeFlag constant used to pass flags through methods
const RNNModeFlag RNNMode = C.CUDNN_RNN_RELU
const (
	rnnRelu RNNMode = C.CUDNN_RNN_RELU /* Stock RNN with ReLu activation */
	rnnTanh RNNMode = C.CUDNN_RNN_TANH /* Stock RNN with tanh activation */
	rnnLstm RNNMode = C.CUDNN_LSTM     /* LSTM with no peephole connections */
	rnnGru  RNNMode = C.CUDNN_GRU      /* Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1); */
)

//Relu returns rnnRelu
func (r RNNMode) Relu() RNNMode {
	return rnnRelu
}

//Tanh returns rnnTanh
func (r RNNMode) Tanh() RNNMode {
	return rnnTanh
}

//Lstm returns rnnLstm
func (r RNNMode) Lstm() RNNMode {
	return rnnLstm
}

//Gru returns rnnGru
func (r RNNMode) Gru() RNNMode {
	return rnnGru
}

//DirectionMode is a type used for flags
type DirectionMode C.cudnnDirectionMode_t

func (r DirectionMode) c() C.cudnnDirectionMode_t { return C.cudnnDirectionMode_t(r) }

//flags for DirectionMode
const DirectionModeFlag DirectionMode = C.CUDNN_UNIDIRECTIONAL
const (
	uniDirectional DirectionMode = C.CUDNN_UNIDIRECTIONAL
	biDirectional  DirectionMode = C.CUDNN_BIDIRECTIONAL
)

//Uni returns uniDirectional flag
func (r DirectionMode) Uni() DirectionMode {
	return uniDirectional
}

//Bi returns biDirectional flag
func (r DirectionMode) Bi() DirectionMode {
	return biDirectional
}

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

//RNNAlgoFlag is a function that returns an Flag that defaults to RNNAlgo(C.CUDNN_RNN_ALGO_STANDARD)
//It has methods to switch the flag
func RNNAlgoFlag() RNNAlgo {
	return RNNAlgo(C.CUDNN_RNN_ALGO_STANDARD)
}

//Standard returns RNNAlgo( C.CUDNN_RNN_ALGO_STANDARD) flag
func (r RNNAlgo) Standard() RNNAlgo {
	return RNNAlgo(C.CUDNN_RNN_ALGO_STANDARD)
}

//PersistStatic returns RNNAlgo( C.CUDNN_RNN_ALGO_PERSIST_STATIC) flag
func (r RNNAlgo) PersistStatic() RNNAlgo {
	return RNNAlgo(C.CUDNN_RNN_ALGO_PERSIST_STATIC)
}

//PersistDynamic returns RNNAlgo( C.CUDNN_RNN_ALGO_PERSIST_DYNAMIC) flag
func (r RNNAlgo) PersistDynamic() RNNAlgo {
	return RNNAlgo(C.CUDNN_RNN_ALGO_PERSIST_DYNAMIC)
}

//Count returns RNNAlgo( C.CUDNN_RNN_ALGO_COUNT) flag
func (r RNNAlgo) Count() RNNAlgo {
	return RNNAlgo(C.CUDNN_RNN_ALGO_COUNT)
}

func (r RNNAlgo) c() C.cudnnRNNAlgo_t { return C.cudnnRNNAlgo_t(r) }

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

//DestroyDescriptor destroys the descriptor
func (r *RNND) DestroyDescriptor() error {
	return Status(C.cudnnDestroyRNNDescriptor(r.descriptor)).error("DestroyDescriptor-rnn")
}

//SetRNNDescriptor sets the rnndesctiptor
func (r *RNND) SetRNNDescriptor(
	handle *Handle,
	hiddenSize int32,
	numLayers int32,
	doD *DropOutD,
	inputmode RNNInputMode,
	direction DirectionMode,
	rnnmode RNNMode,
	rnnalg RNNAlgo,
	data DataType,

) error {
	return Status(C.cudnnSetRNNDescriptor(
		handle.x,
		r.descriptor,
		C.int(hiddenSize),
		C.int(numLayers),
		doD.descriptor,
		inputmode.c(),
		direction.c(),
		rnnmode.c(),
		rnnalg.c(),
		data.c(),
	)).error("SetRNNDescriptor")
}

//SetRNNProjectionLayers sets the rnnprojection layers
func (r *RNND) SetRNNProjectionLayers(
	handle *Handle,
	recProjsize int32,
	outProjSize int32,
) error {
	return Status(C.cudnnSetRNNProjectionLayers(
		handle.x,
		r.descriptor,
		C.int(recProjsize),
		C.int(outProjSize),
	)).error("SetRNNProjectionLayers")
}

//GetRNNProjectionLayers sets the rnnprojection layers
func (r *RNND) GetRNNProjectionLayers(
	handle *Handle,

) (int32, int32, error) {
	var rec, out C.int

	err := Status(C.cudnnGetRNNProjectionLayers(
		handle.x,
		r.descriptor,
		&rec,
		&out,
	)).error("SetRNNProjectionLayers")
	return int32(rec), int32(out), err
}

//SetRNNAlgorithmDescriptor sets the RNNalgorithm
func (r *RNND) SetRNNAlgorithmDescriptor(
	handle *Handle,
	algo *AlgorithmD,
) error {
	return Status(C.cudnnSetRNNAlgorithmDescriptor(handle.x, r.descriptor, algo.descriptor)).error("SetRNNAlgorithmDescriptor")
}

//GetRNNDescriptor gets algo desctiptor values
func (r *RNND) GetRNNDescriptor() {

}
