package gocudnn

/*
#include <cudnn.h>

void MakeAlgorithmforRNN(cudnnAlgorithm_t *input,cudnnRNNAlgo_t RNNAlgo ){
	input->algo.RNNAlgo=RNNAlgo;
}
*/
import "C"
import (
	"unsafe"
)

type RNN struct {
	Funcs RNNFuncs
	Flgs  RNNFlags
}

//Algo returns an Algorithm used for
func (r RNNAlgo) Algo() Algos {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforRNN(&algorithm, r.c())
	return Algos(algorithm)
}

//AlgorithmPerformance go typed C.cudnnAlgorithmPerformance_t
type AlgorithmPerformance struct {
	descriptor C.cudnnAlgorithmPerformance_t
	index      C.int
}

//RNND  holdes Rnn descriptor
type RNND struct {
	descriptor C.cudnnRNNDescriptor_t
}

func rrndArrayToCarray(input []RNND) []C.cudnnRNNDescriptor_t {
	array := make([]C.cudnnRNNDescriptor_t, len(input))
	for i := 0; i < len(input); i++ {
		array[i] = input[i].descriptor
	}
	return array
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
	rnnmode RNNmode,
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
) (int32, int32, *DropOutD, RNNInputMode, DirectionMode, RNNmode, RNNAlgo, DataType, error) {
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
		RNNInputMode(inputMode), DirectionMode(direction), RNNmode(mode), RNNAlgo(algo), DataType(dataType), err
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

/* dataType in the RNN descriptor is used to determine math precision */
/* dataType in weight descriptors and input descriptors is used to describe storage */

//GetRNNWorkspaceSize gets the RNN workspace size (WOW!)
func (r *RNND) GetRNNWorkspaceSize(
	handle *Handle,
	seqLength int32,
	xD []*TensorD,
) (SizeT, error) {
	tocxD := tensorDArrayToC(xD)
	var sizeinbytes C.size_t
	err := Status(C.cudnnGetRNNWorkspaceSize(
		handle.x,
		r.descriptor,
		C.int(seqLength),
		&tocxD[0],
		&sizeinbytes,
	)).error("GetRNNWorkspaceSize")
	return SizeT(sizeinbytes), err
}

//GetRNNTrainingReserveSize gets the training reserve size
func (r *RNND) GetRNNTrainingReserveSize(
	handle *Handle,
	seqLength int32,
	xD []*TensorD,
) (SizeT, error) {
	tocxD := tensorDArrayToC(xD)
	var sizeinbytes C.size_t
	err := Status(C.cudnnGetRNNTrainingReserveSize(
		handle.x,
		r.descriptor,
		C.int(seqLength),
		&tocxD[0],
		&sizeinbytes,
	)).error("GetRNNTrainingReserveSize")
	return SizeT(sizeinbytes), err
}

//GetRNNParamsSize gets the training reserve size
func (r *RNND) GetRNNParamsSize(
	handle *Handle,
	xD *TensorD,
	data *DataType,
) (SizeT, error) {
	var sizeinbytes C.size_t
	err := Status(C.cudnnGetRNNParamsSize(
		handle.x,
		r.descriptor,
		xD.descriptor,
		&sizeinbytes,
		data.c(),
	)).error("GetRNNParamsSize")
	return SizeT(sizeinbytes), err
}

//GetRNNLinLayerMatrixParams gets the parameters of the layer matrix
func (r *RNND) GetRNNLinLayerMatrixParams(
	handle *Handle,
	pseudoLayer int32,
	/*
	   The pseudo-layer to query.
	   In uni-directional RNN-s, a pseudo-layer is the same as a "physical" layer
	   (pseudoLayer=0 is the RNN input layer, pseudoLayer=1 is the first hidden layer).

	   In bi-directional RNN-s there are twice as many pseudo-layers in comparison to "physical" layers
	   (pseudoLayer=0 and pseudoLayer=1 are both input layers;
	   pseudoLayer=0 refers to the forward part and pseudoLayer=1 refers to the backward part
	    of the "physical" input layer; pseudoLayer=2 is the forward part of the first hidden layer, and so on).

	*/
	xD *TensorD,
	wD *FilterD,
	w Memer,
	linlayerID int32,
	/*
	   Input. The linear layer to obtain information about:

	   If mode in rnnDesc was set to CUDNN_RNN_RELU or CUDNN_RNN_TANH a value of 0 references the matrix multiplication applied to the input from the previous layer,
	   a value of 1 references the matrix multiplication applied to the recurrent input.

	   If mode in rnnDesc was set to CUDNN_LSTM values of 0-3 reference matrix multiplications applied to the input from the previous layer, value of 4-7 reference matrix multiplications applied to the recurrent input.
	       Values 0 and 4 reference the input gate.
	       Values 1 and 5 reference the forget gate.
	       Values 2 and 6 reference the new memory gate.
	       Values 3 and 7 reference the output gate.
	       Value 8 references the "recurrent" projection matrix when enabled by the cudnnSetRNNProjectionLayers() function.

	   If mode in rnnDesc was set to CUDNN_GRU values of 0-2 reference matrix multiplications applied to the input from the previous layer, value of 3-5 reference matrix multiplications applied to the recurrent input.
	       Values 0 and 3 reference the reset gate.
	       Values 1 and 4 reference the update gate.
	       Values 2 and 5 reference the new memory gate.

	*/

) (FilterD, unsafe.Pointer, error) {
	var linLayerMatDesc FilterD
	var linLayerMat unsafe.Pointer
	err := Status(C.cudnnGetRNNLinLayerMatrixParams(
		handle.x,
		r.descriptor,
		C.int(pseudoLayer),
		xD.descriptor,
		wD.descriptor,
		w.Ptr(),
		C.int(linlayerID),
		linLayerMatDesc.descriptor,
		&linLayerMat,
	)).error("GetRNNLinLayerMatrixParams")
	return linLayerMatDesc, linLayerMat, err
}

//GetRNNLinLayerBiasParams gets the parameters of the layer bias
func (r *RNND) GetRNNLinLayerBiasParams(
	handle *Handle,
	pseudoLayer int32,
	/*
	   The pseudo-layer to query.
	   In uni-directional RNN-s, a pseudo-layer is the same as a "physical" layer
	   (pseudoLayer=0 is the RNN input layer, pseudoLayer=1 is the first hidden layer).

	   In bi-directional RNN-s there are twice as many pseudo-layers in comparison to "physical" layers
	   (pseudoLayer=0 and pseudoLayer=1 are both input layers;
	   pseudoLayer=0 refers to the forward part and pseudoLayer=1 refers to the backward part
	    of the "physical" input layer; pseudoLayer=2 is the forward part of the first hidden layer, and so on).

	*/
	xD *TensorD,
	wD *FilterD,
	w Memer,
	linlayerID int32,
	/*
	   Input. The linear layer to obtain information about:

	   If mode in rnnDesc was set to CUDNN_RNN_RELU or CUDNN_RNN_TANH a value of 0 references the matrix multiplication applied to the input from the previous layer,
	   a value of 1 references the matrix multiplication applied to the recurrent input.

	   If mode in rnnDesc was set to CUDNN_LSTM values of 0-3 reference matrix multiplications applied to the input from the previous layer, value of 4-7 reference matrix multiplications applied to the recurrent input.
	       Values 0 and 4 reference the input gate.
	       Values 1 and 5 reference the forget gate.
	       Values 2 and 6 reference the new memory gate.
	       Values 3 and 7 reference the output gate.
	       Value 8 references the "recurrent" projection matrix when enabled by the cudnnSetRNNProjectionLayers() function.

	   If mode in rnnDesc was set to CUDNN_GRU values of 0-2 reference matrix multiplications applied to the input from the previous layer, value of 3-5 reference matrix multiplications applied to the recurrent input.
	       Values 0 and 3 reference the reset gate.
	       Values 1 and 4 reference the update gate.
	       Values 2 and 5 reference the new memory gate.

	*/

) (FilterD, unsafe.Pointer, error) {
	var linLayerBiasDesc FilterD
	var linLayerBias unsafe.Pointer
	err := Status(C.cudnnGetRNNLinLayerBiasParams(
		handle.x,
		r.descriptor,
		C.int(pseudoLayer),
		xD.descriptor,
		wD.descriptor,
		w.Ptr(),
		C.int(linlayerID),
		linLayerBiasDesc.descriptor,
		&linLayerBias,
	)).error("GetRNNLinLayerBiasParams")
	return linLayerBiasDesc, linLayerBias, err
}

//PersistentRNNPlan holds  C.cudnnPersistentRNNPlan_t
type PersistentRNNPlan struct {
	plan C.cudnnPersistentRNNPlan_t
}

//CreatePersistentRNNPlan creates a PersistentRNNPlan
func (r *RNND) CreatePersistentRNNPlan(minibatch int32, data DataType) (PersistentRNNPlan, error) {
	var plan C.cudnnPersistentRNNPlan_t
	err := Status(C.cudnnCreatePersistentRNNPlan(
		r.descriptor,
		C.int(minibatch),
		data.c(),
		&plan,
	)).error("CreatePersistentRNNPlan")
	return PersistentRNNPlan{
		plan: plan}, err
}

//SetPersistentRNNPlan sets a SetPersistentRNNPlan
func (r *RNND) SetPersistentRNNPlan(plan PersistentRNNPlan) error {
	return Status(C.cudnnSetPersistentRNNPlan(r.descriptor, plan.plan)).error("SetPersistentRNNPlan")
}

//DestroyPersistentRNNPlan destroys the C.cudnnPersistentRNNPlan_t in the PersistentRNNPlan struct
func (p *PersistentRNNPlan) DestroyPersistentRNNPlan() error {
	return Status(C.cudnnDestroyPersistentRNNPlan(p.plan)).error("DestroyPersistentRNNPlan")
}

/*

RNN Fluncs



*/

//RNNFuncs is a nil struct used to call rnn functions
type RNNFuncs struct {
}

//GetRNNForwardInferenceAlgorithmMaxCount returns the maxcount and error
func (rn RNNFuncs) GetRNNForwardInferenceAlgorithmMaxCount(
	handle *Handle,
	rnnD RNND,
) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNForwardInferenceAlgorithmMaxCount(
		handle.x,
		rnnD.descriptor,
		&count,
	)).error("GetRNNForwardInferenceAlgorithmMaxCount")
	return int32(count), err
}

//FindRNNForwardInferenceAlgorithmEx finds the inference algorithmEx
func (rn RNNFuncs) FindRNNForwardInferenceAlgorithmEx(
	handle *Handle,
	rnnD *RNND,
	seqlength int32,
	xD []*TensorD, //Input. An array of fully packed tensor descriptors describing the input to each recurrent iteration (one descriptor per iteration).
	x Memer, //input
	hxD *TensorD, //Input. A fully packed tensor descriptor describing the initial hidden state of the RNN.
	hx Memer, //input
	cxD *TensorD, //Input. A fully packed tensor descriptor describing the initial cell state for LSTM networks.
	cx Memer, //input
	wD *FilterD, //Input. Handle to a previously initialized filter descriptor describing the weights for the RNN.
	w Memer, //Input
	yD []*TensorD, //input An array of fully packed tensor descriptors.
	y Memer, //Output Data pointer to GPU memory associated with the output tensor descriptor yDesc
	hyD *TensorD, //input  A fully packed tensor descriptor describing the final hidden state of the RNN.
	hy Memer, //Output. Data pointer to GPU memory associated with the tensor descriptor hyDesc. If
	cyD *TensorD, //Input. A fully packed tensor descriptor describing the final cell state for LSTM networks.
	cy Memer, //output
	findIntensity float32,
	algocount int32,
	wspace Memer,

) ([]AlgorithmPerformance, error) {
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	var retactAlgoCount C.int
	perfResults := make([]C.cudnnAlgorithmPerformance_t, algocount)
	reqcount := C.int(algocount)
	err := Status(C.cudnnFindRNNForwardInferenceAlgorithmEx(
		handle.x,
		rnnD.descriptor,
		C.int(seqlength),
		&tocxD[0],
		x.Ptr(),
		hxD.descriptor,
		hx.Ptr(),
		cxD.descriptor,
		cx.Ptr(),
		wD.descriptor,
		w.Ptr(),
		&tocyD[0],
		y.Ptr(),
		hyD.descriptor,
		hy.Ptr(),
		cyD.descriptor,
		cy.Ptr(),
		C.float(findIntensity),
		reqcount,
		&retactAlgoCount,
		&perfResults[0],
		wspace.Ptr(),
		wspace.ByteSize().c(),
	)).error("FindRNNForwardInferenceAlgorithmEx")

	return calgoperftogoarray(perfResults), err
}

//GetRNNForwardTrainingAlgorithmMaxCount gets the max number of algorithms for rnnforward training algo
func (rn RNNFuncs) GetRNNForwardTrainingAlgorithmMaxCount(handle *Handle, rnn RNND) (int32, error) {
	var count C.int
	stat := C.cudnnGetRNNForwardTrainingAlgorithmMaxCount(
		handle.x,
		rnn.descriptor,
		&count)
	return int32(count), Status(stat).error("GetRNNForwardTrainingAlgorithmMaxCount")
}

//FindRNNForwardTrainingAlgorithmEx finds and orders the performance of rnn algos for training returns that list with an error
func (rn RNNFuncs) FindRNNForwardTrainingAlgorithmEx(
	handle *Handle,
	rnn *RNND,
	seqLen int32, //input
	xD []*TensorD, //input
	x Memer, //input
	hxD *TensorD, //input: A fully packed tensor descriptor describing the initial hidden state of the RNN.
	hx Memer, //input
	cxD *TensorD, // :input A fully packed tensor descriptor describing the initial cell state for LSTM networks.
	cx Memer, //input
	wD *FilterD, //input
	w Memer, //input
	yD []*TensorD, //Input. An array of fully packed tensor descriptors describing the output from each recurrent iteration (one descriptor per iteration).
	y Memer, //output
	hyD *TensorD, //input
	hy Memer, //output
	cyD *TensorD,
	cy Memer, //output
	findIntensity float32, //input
	reqAlgocount int32, //input
	wspace Memer, ///input
	rspace Memer, //input/output

) ([]AlgorithmPerformance, error) {
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)

	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	err := Status(C.cudnnFindRNNForwardTrainingAlgorithmEx(
		handle.x,
		rnn.descriptor,
		C.int(seqLen),
		&tocxD[0],
		x.Ptr(),
		hxD.descriptor,
		hx.Ptr(),
		cxD.descriptor,
		cx.Ptr(),
		wD.descriptor,
		w.Ptr(),
		&tocyD[0],
		y.Ptr(),
		hyD.descriptor,
		hy.Ptr(),
		cyD.descriptor,
		cy.Ptr(),
		C.float(findIntensity),
		C.int(reqAlgocount),
		&actualcount,
		&perfresults[0],
		wspace.Ptr(),
		wspace.ByteSize().c(),
		rspace.Ptr(),
		rspace.ByteSize().c(),
	)).error("FindRNNForwardTrainingAlgorithmEx")

	return calgoperftogoarray(perfresults), err
}

//GetRNNBackwardDataAlgorithmMaxCount gets the max number of algorithms for the back prop rnn
func (rn RNNFuncs) GetRNNBackwardDataAlgorithmMaxCount(handle *Handle, rnnd *RNND) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNBackwardDataAlgorithmMaxCount(
		handle.x,
		rnnd.descriptor,
		&count,
	)).error("GetRNNBackwardDataAlgorithmMaxCount")
	return int32(count), err
}

//FindRNNBackwardDataAlgorithmEx finds a list of algos for backprop this passes like 26 parameters and pointers and stuff so watch out.
func (rn RNNFuncs) FindRNNBackwardDataAlgorithmEx(
	handle *Handle,
	rnnD *RNND,
	seqLen int32,

	yD []*TensorD, //an array of fully packed tensor descriptors
	y Memer,

	dyD []*TensorD, //an array of fully packed tensor descriptors
	dy Memer,

	dhyD *TensorD, //fully packed tensor descriptor describing the gradients at the final hidden state of the RNN
	dhy Memer,

	dcyD *TensorD, // fully packed tensor descriptor describing the gradients at the final cell state of the RNN.
	dcy Memer,

	wD *FilterD,
	w Memer,

	hxD *TensorD, // A fully packed tensor descriptor describing the initial hidden state of the RNN.
	hx Memer,

	cxD *TensorD, //A fully packed tensor descriptor describing the initial cell state for LSTM networks.
	cx Memer,

	dxD []*TensorD, //
	dx Memer,

	dhxD *TensorD, //A fully packed tensor descriptor describing the gradient at the initial hidden state of the RNN.
	dhx Memer,

	dcxD *TensorD, // A fully packed tensor descriptor describing the gradient at the initial cell state of the RNN.
	dcx Memer,

	findIntensity float32,
	reqAlgocount int32,
	wspace Memer,
	rspace Memer,

) ([]AlgorithmPerformance, error) {
	cyD := tensorDArrayToC(yD)
	cdyD := tensorDArrayToC(dyD)
	cdxD := tensorDArrayToC(dxD)
	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	err := Status(C.cudnnFindRNNBackwardDataAlgorithmEx(
		handle.x,
		rnnD.descriptor,
		C.int(seqLen),

		&cyD[0],
		y.Ptr(),

		&cdyD[0],
		dy.Ptr(),

		dhyD.descriptor,
		dhy.Ptr(),

		dcyD.descriptor,
		dcy.Ptr(),

		wD.descriptor,
		w.Ptr(),

		hxD.descriptor,
		hx.Ptr(),

		cxD.descriptor,
		cx.Ptr(),

		&cdxD[0],
		dx.Ptr(),

		dhxD.descriptor,
		dhx.Ptr(),

		dcxD.descriptor,
		dcx.Ptr(),

		C.float(findIntensity),
		C.int(reqAlgocount),
		&actualcount,
		&perfresults[0],

		wspace.Ptr(),
		wspace.ByteSize().c(),
		rspace.Ptr(),
		rspace.ByteSize().c(),
		//31 total?
	)).error("FindRNNBackwardDataAlgorithmEx")

	return calgoperftogoarray(perfresults), err
}

//GetRNNBackwardWeightsAlgorithmMaxCount gets the max number of algos for weights
func (rn RNNFuncs) GetRNNBackwardWeightsAlgorithmMaxCount(handle *Handle, rnnD *RNND) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
		handle.x,
		rnnD.descriptor,
		&count,
	)).error("GetRNNBackwardWeightsAlgorithmMaxCount")
	return int32(count), err
}

//FindRNNBackwardWeightsAlgorithmEx returns some algos and their performance and stuff
func (rn RNNFuncs) FindRNNBackwardWeightsAlgorithmEx(
	handle *Handle,
	rnnD *RNND,
	seqLen int32,
	xD []*TensorD,
	x Memer,
	hxD *TensorD, //Initial Hidden State
	hx Memer,
	yD []*TensorD,
	y Memer,
	findIntensity float32, //unused for future use
	reqAlgocount int32, //the max number of elements
	wspace Memer,
	dwD *FilterD,
	dw Memer,
	rspace Memer,

) ([]AlgorithmPerformance, error) {
	var actualcount C.int
	inCxD := tensorDArrayToC(xD)
	inCyD := tensorDArrayToC(yD)
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	err := Status(C.cudnnFindRNNBackwardWeightsAlgorithmEx(
		handle.x,
		rnnD.descriptor,
		C.int(seqLen),
		&inCxD[0], //input array
		x.Ptr(),
		hxD.descriptor,
		hx.Ptr(),
		&inCyD[0], //input array
		y.Ptr(),

		C.float(findIntensity),
		C.int(reqAlgocount),
		&actualcount,
		&perfresults[0],

		wspace.Ptr(),
		wspace.ByteSize().c(),

		dwD.descriptor,
		dw.Ptr(),

		rspace.Ptr(),
		rspace.ByteSize().c(),
	)).error("FindRNNBackwardWeightsAlgorithmEx")

	return calgoperftogoarray(perfresults), err
}

//RNNForwardInference is the forward inference
func (rn RNNFuncs) RNNForwardInference(
	handle *Handle,
	rnnd *RNND,
	seqLength int32,
	xD []*TensorD,
	x Memer,
	hxD *TensorD,
	hx Memer,
	cxD *TensorD,
	cx Memer,
	wD *FilterD,
	w Memer,
	yD []*TensorD,
	y Memer,
	hyD TensorD,
	hy Memer,
	cyD TensorD,
	cy Memer,
	wspace Memer,
) error {
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	return Status(C.cudnnRNNForwardInference(
		handle.x,
		rnnd.descriptor,
		C.int(seqLength),
		&tocxD[0],
		x.Ptr(),
		hxD.descriptor,
		hx.Ptr(),
		cxD.descriptor,
		cx.Ptr(),
		wD.descriptor,
		w.Ptr(),
		&tocyD[0],
		y.Ptr(),
		hyD.descriptor,
		hy.Ptr(),
		cyD.descriptor,
		cy.Ptr(),
		w.Ptr(),
		w.ByteSize().c(),
	)).error("RNNForwardInference")
}

//RNNForwardTraining is the forward algo for an RNN
func (rn RNNFuncs) RNNForwardTraining(
	handle *Handle,
	r *RNND,
	seqLen int32,
	xD []*TensorD,
	x Memer,
	hxD *TensorD,
	hx Memer,
	cxD *TensorD,
	cx Memer,
	wD *FilterD,
	w Memer,
	yD []*TensorD,
	y Memer,
	hyD *TensorD,
	hy Memer,
	cyD *TensorD,
	cy Memer,
	wspace Memer,
	rspace Memer,
) error {
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	err := Status(C.cudnnRNNForwardTraining(
		handle.x,
		r.descriptor,
		C.int(seqLen),
		&tocxD[0],
		x.Ptr(),
		hxD.descriptor,
		hx.Ptr(),
		cxD.descriptor,
		cx.Ptr(),
		wD.descriptor,
		w.Ptr(),
		&tocyD[0],
		y.Ptr(),
		hyD.descriptor,
		hy.Ptr(),
		cyD.descriptor,
		cy.Ptr(),
		wspace.Ptr(),
		wspace.ByteSize().c(),
		rspace.Ptr(),
		rspace.ByteSize().c(),
	)).error("RNNForwardTraining")
	return err
}

//RNNBackwardData is the backward algo for an RNN
func (rn RNNFuncs) RNNBackwardData(
	handle *Handle,
	r *RNND,
	seqLen int32,

	yD []*TensorD,
	y Memer,

	dyD []*TensorD,
	dy Memer,

	dhyD *TensorD,
	dhy Memer,

	dcyD *TensorD,
	dcy Memer,

	wD *FilterD,
	w Memer,

	hxD *TensorD,
	hx Memer,

	cxD *TensorD,
	cx Memer,

	dxD []*TensorD,
	dx Memer,

	dhxD *TensorD,
	dhx Memer,

	dcxD *TensorD,
	dcx Memer,

	wspace Memer,
	rspace Memer,
) error {
	tocdxD := tensorDArrayToC(dxD)
	tocdyD := tensorDArrayToC(dyD)
	tocyD := tensorDArrayToC(yD)
	err := Status(C.cudnnRNNBackwardData(
		handle.x,
		r.descriptor,
		C.int(seqLen),
		&tocyD[0],
		y.Ptr(),
		&tocdyD[0],
		dy.Ptr(),
		dhyD.descriptor,
		dhy.Ptr(),
		dcyD.descriptor,
		dcy.Ptr(),
		wD.descriptor,
		w.Ptr(),
		hxD.descriptor,
		hx.Ptr(),
		cxD.descriptor,
		cx.Ptr(),
		&tocdxD[0],
		dx.Ptr(),
		dhxD.descriptor,
		dhx.Ptr(),
		dcxD.descriptor,
		dcx.Ptr(),
		wspace.Ptr(),
		wspace.ByteSize().c(),
		rspace.Ptr(),
		rspace.ByteSize().c(),
	)).error("RNNBackwardData")
	return err
}

//BackwardWeights does the backward weight function
func (rn RNNFuncs) BackwardWeights(
	handle *Handle,
	r *RNND,
	seqLen int32,
	xD []*TensorD,
	x Memer,
	hxD *TensorD,
	hx Memer,
	yD []*TensorD,
	y Memer,
	wspace Memer,
	dwD *FilterD,
	dw Memer,
	rspace Memer,
) error {
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	err := Status(C.cudnnRNNBackwardWeights(
		handle.x,
		r.descriptor,
		C.int(seqLen),
		&tocxD[0],
		x.Ptr(),
		hxD.descriptor,
		hx.Ptr(),
		&tocyD[0],
		y.Ptr(),
		wspace.Ptr(),
		wspace.ByteSize().c(),
		dwD.descriptor,
		dw.Ptr(),
		rspace.Ptr(),
		rspace.ByteSize().c(),
	)).error("BackwardWeights")

	return err
}

/*


FLAGS

*/

type RNNFlags struct {
	Mode      RNNModeFlag
	Algo      RNNAlgoFlag
	Direction DirectionModeFlag
	Input     RNNInputModeFlag
}

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
