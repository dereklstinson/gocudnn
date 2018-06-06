package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//rnnmode is used for flags
type rnnmode C.cudnnRNNMode_t

//RNNModeFlag is a nil struct that is used to pass the RNN flags which are unexported.
//I wanted a way to force the user to use methods in passing the flags so that nothin was passed on accident.
type RNNModeFlags struct {
}

func (r rnnmode) c() C.cudnnRNNMode_t { return C.cudnnRNNMode_t(r) }

//RNNModeFlag func pass an RNNMode type that is used for flags default is RNN_RELU, but it can be changed with RNNMode methods
func CreateRNNModeFlager() RNNModeFlags {
	return RNNModeFlags{}
}

//Relu return RNNMode(C.CUDNN_RNN_RELU)
func (r RNNModeFlags) Relu() rnnmode {
	return rnnmode(C.CUDNN_RNN_RELU)
}

//Tanh returns rnnTanh
func (r RNNModeFlags) Tanh() rnnmode {
	return rnnmode(C.CUDNN_RNN_TANH)
}

//Lstm returns rnnLstm
func (r RNNModeFlags) Lstm() rnnmode {
	return rnnmode(C.CUDNN_LSTM)
}

//Gru returns rnnGru
func (r RNNModeFlags) Gru() rnnmode {
	return rnnmode(C.CUDNN_GRU)
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
	rnnmode rnnmode,
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

//GetRNNDescriptor gets algo desctiptor values returns a ton of stuff
func (r *RNND) GetRNNDescriptor(
	handle *Handle,
) (int32, int32, *DropOutD, RNNInputMode, DirectionMode, rnnmode, RNNAlgo, DataType, error) {
	var hiddensize C.int
	var numLayers C.int
	var dropoutdescriptor C.cudnnDropoutDescriptor_t
	var inputMode C.cudnnRNNInputMode_t
	var direction C.cudnnDirectionMode_t
	var mode C.cudnnRNNMode_t
	var algo C.cudnnRNNAlgo_t
	var dataType C.cudnnDataType_t
	err := Status(C.cudnnGetRNNDescriptor(
		handle.x,
		r.descriptor,
		&hiddensize,
		&numLayers,
		&dropoutdescriptor,
		&inputMode,
		&direction,
		&mode,
		&algo,
		&dataType,
	)).error("GetRNNDescriptor")
	return int32(hiddensize), int32(numLayers), &DropOutD{descriptor: dropoutdescriptor},
		RNNInputMode(inputMode), DirectionMode(direction), rnnmode(mode), RNNAlgo(algo), DataType(dataType), err
}

//SetRNNMatrixMathType Sets the math type for the descriptor
func (r *RNND) SetRNNMatrixMathType(math MathType) error {
	return Status(C.cudnnSetRNNMatrixMathType(r.descriptor, math.c())).error("SetRNNMatrixMathType")
}

//GetRNNMatrixMathType Gets the math type for the descriptor
func (r *RNND) GetRNNMatrixMathType() (MathType, error) {
	var math C.cudnnMathType_t
	err := Status(C.cudnnGetRNNMatrixMathType(r.descriptor, &math)).error("SetRNNMatrixMathType")
	return MathType(math), err
}
