package gocudnn

/*
#include <cudnn.h>

void MakeAlgorithmforRNN(cudnnAlgorithm_t *input,cudnnRNNAlgo_t RNNAlgo ){
	input->algo.RNNAlgo=RNNAlgo;
}
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//RNN holds the funcs and flags that are used for RNN stuff it is also used for the creation of an RNND
type RNN struct {
	Flgs RNNFlags
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

func (r *RNND) keepsalive() {
	runtime.KeepAlive(r)
}
func rrndArrayToCarray(input []RNND) []C.cudnnRNNDescriptor_t {
	array := make([]C.cudnnRNNDescriptor_t, len(input))
	for i := 0; i < len(input); i++ {
		array[i] = input[i].descriptor
	}
	return array
}

//CreateRNNDescriptor creates an RNND descriptor
func (rn RNN) CreateRNNDescriptor() (descriptor *RNND, err error) {
	var desc C.cudnnRNNDescriptor_t
	err = Status(C.cudnnCreateRNNDescriptor(&desc)).error("CreateRNNDescriptor")
	if err != nil {
		return nil, err
	}
	descriptor = &RNND{
		descriptor: desc,
	}
	if setfinalizer == true {
		runtime.SetFinalizer(descriptor, destroyrnnddescriptor)
	}

	return descriptor, err
}

//DestroyDescriptor destroys the descriptor
func (r *RNND) DestroyDescriptor() error {
	return destroyrnnddescriptor(r)
}
func destroyrnnddescriptor(r *RNND) error {
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
	if setkeepalive == true {
		keepsalivebuffer(r, handle, doD)
	}

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
	if setkeepalive == true {
		keepsalivebuffer(r, handle)
	}

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
	if setkeepalive == true {
		keepsalivebuffer(r, handle)
	}

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
	if setkeepalive == true {
		keepsalivebuffer(r, handle)
	}
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
	if setkeepalive == true {
		keepsalivebuffer(r, handle)
	}
	return int32(hiddensize), int32(numLayers), &DropOutD{descriptor: dropoutdescriptor},
		RNNInputMode(inputMode), DirectionMode(direction), RNNmode(mode), RNNAlgo(algo), DataType(dataType), err
}

//SetRNNMatrixMathType Sets the math type for the descriptor
func (r *RNND) SetRNNMatrixMathType(math MathType) error {
	if setkeepalive == true {
		keepsalivebuffer(r)
	}
	return Status(C.cudnnSetRNNMatrixMathType(r.descriptor, math.c())).error("SetRNNMatrixMathType")
}

//GetRNNMatrixMathType Gets the math type for the descriptor
func (r *RNND) GetRNNMatrixMathType() (MathType, error) {
	var math C.cudnnMathType_t
	err := Status(C.cudnnGetRNNMatrixMathType(r.descriptor, &math)).error("SetRNNMatrixMathType")
	if setkeepalive == true {
		keepsalivebuffer(r)
	}
	return MathType(math), err
}

/* dataType in the RNN descriptor is used to determine math precision */
/* dataType in weight descriptors and input descriptors is used to describe storage */

//GetRNNWorkspaceSize gets the RNN workspace size (WOW!)
func (r *RNND) GetRNNWorkspaceSize(
	handle *Handle,
	seqLength int32,
	xD []*TensorD,
) (uint, error) {
	tocxD := tensorDArrayToC(xD)
	var sizeinbytes C.size_t
	err := Status(C.cudnnGetRNNWorkspaceSize(
		handle.x,
		r.descriptor,
		C.int(seqLength),
		&tocxD[0],
		&sizeinbytes,
	)).error("GetRNNWorkspaceSize")
	if setkeepalive == true {
		keepsalivebuffer(r, handle)
		for i := range xD {
			xD[i].keepsalive()
		}
	}
	return uint(sizeinbytes), err
}

//GetRNNTrainingReserveSize gets the training reserve size
func (r *RNND) GetRNNTrainingReserveSize(
	handle *Handle,
	seqLength int32,
	xD []*TensorD,
) (uint, error) {
	tocxD := tensorDArrayToC(xD)
	var sizeinbytes C.size_t
	err := Status(C.cudnnGetRNNTrainingReserveSize(
		handle.x,
		r.descriptor,
		C.int(seqLength),
		&tocxD[0],
		&sizeinbytes,
	)).error("GetRNNTrainingReserveSize")
	if setkeepalive == true {
		keepsalivebuffer(r, handle)
		for i := range xD {
			xD[i].keepsalive()
		}
	}
	return uint(sizeinbytes), err
}

//GetRNNParamsSize gets the training reserve size
func (r *RNND) GetRNNParamsSize(
	handle *Handle,
	xD *TensorD,
	data DataType,
) (uint, error) {
	var sizeinbytes C.size_t
	err := Status(C.cudnnGetRNNParamsSize(
		handle.x,
		r.descriptor,
		xD.descriptor,
		&sizeinbytes,
		data.c(),
	)).error("GetRNNParamsSize")
	if setkeepalive == true {
		keepsalivebuffer(r, handle, xD)
	}
	return uint(sizeinbytes), err
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
	w gocu.Mem,
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
	if setkeepalive == true {
		keepsalivebuffer(handle, r, xD, wD, w, linLayerMatDesc, linLayerMat)
	}
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
	w gocu.Mem,
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
	if setkeepalive == true {
		keepsalivebuffer(handle, r, xD, wD, w, linLayerBiasDesc)
	}
	return linLayerBiasDesc, linLayerBias, err
}

//PersistentRNNPlan holds  C.cudnnPersistentRNNPlan_t
type PersistentRNNPlan struct {
	plan C.cudnnPersistentRNNPlan_t
}

//NewPersistentRNNPlan creates and sets a PersistentRNNPlan
func (r *RNND) NewPersistentRNNPlan(minibatch int32, data DataType) (plan *PersistentRNNPlan, err error) {
	var plan1 C.cudnnPersistentRNNPlan_t
	err = Status(C.cudnnCreatePersistentRNNPlan(
		r.descriptor,
		C.int(minibatch),
		data.c(),
		&plan1,
	)).error("CreatePersistentRNNPlan")
	plan = &PersistentRNNPlan{
		plan: plan1}
	err = Status(C.cudnnSetPersistentRNNPlan(r.descriptor, plan.plan)).error("SetPersistentRNNPlan")

	if setfinalizer == true {
		runtime.SetFinalizer(plan, destroypersistantrnnplan)
	}
	return plan, err
}
func (p *PersistentRNNPlan) keepsalive() {
	runtime.KeepAlive(p)
}

//DestroyPersistentRNNPlan destroys the C.cudnnPersistentRNNPlan_t in the PersistentRNNPlan struct
func (p *PersistentRNNPlan) DestroyPersistentRNNPlan() error {
	return destroypersistantrnnplan(p)
}
func destroypersistantrnnplan(p *PersistentRNNPlan) error {
	return Status(C.cudnnDestroyPersistentRNNPlan(p.plan)).error("DestroyPersistentRNNPlan")
}

/*

RNN Fluncs



*/

//GetRNNForwardInferenceAlgorithmMaxCount returns the maxcount and error
func (r *RNND) GetRNNForwardInferenceAlgorithmMaxCount(
	handle *Handle,
) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNForwardInferenceAlgorithmMaxCount(
		handle.x,
		r.descriptor,
		&count,
	)).error("GetRNNForwardInferenceAlgorithmMaxCount")
	if setkeepalive == true {
		keepsalivebuffer(handle, r)
	}
	return int32(count), err
}

//FindRNNForwardInferenceAlgorithmEx finds the inference algorithmEx
func (r *RNND) FindRNNForwardInferenceAlgorithmEx(
	handle *Handle,
	seqlength int32,
	xD []*TensorD, //Input. An array of fully packed tensor descriptors describing the input to each recurrent iteration (one descriptor per iteration).
	x gocu.Mem, //input
	hxD *TensorD, //Input. A fully packed tensor descriptor describing the initial hidden state of the RNN.
	hx gocu.Mem, //input
	cxD *TensorD, //Input. A fully packed tensor descriptor describing the initial cell state for LSTM networks.
	cx gocu.Mem, //input
	wD *FilterD, //Input. Handle to a previously initialized filter descriptor describing the weights for the RNN.
	w gocu.Mem, //Input
	yD []*TensorD, //input An array of fully packed tensor descriptors.
	y gocu.Mem, //Output Data pointer to GPU memory associated with the output tensor descriptor yDesc
	hyD *TensorD, //input  A fully packed tensor descriptor describing the final hidden state of the RNN.
	hy gocu.Mem, //Output. Data pointer to GPU memory associated with the tensor descriptor hyDesc. If
	cyD *TensorD, //Input. A fully packed tensor descriptor describing the final cell state for LSTM networks.
	cy gocu.Mem, //output
	findIntensity float32,
	algocount int32,
	wspace gocu.Mem,
	wspacesize uint,
) ([]AlgorithmPerformance, error) {
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	var retactAlgoCount C.int
	perfResults := make([]C.cudnnAlgorithmPerformance_t, algocount)
	reqcount := C.int(algocount)
	err := Status(C.cudnnFindRNNForwardInferenceAlgorithmEx(
		handle.x,
		r.descriptor,
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
		C.size_t(wspacesize),
	)).error("FindRNNForwardInferenceAlgorithmEx")
	if setkeepalive == true {
		keepsalivebuffer(handle, r, x, hxD, hx, cxD, cx, wD, w, y, hyD, hy, cyD, cy, wspace)
		for i := range xD {
			xD[i].keepsalive()
		}
		for i := range yD {
			yD[i].keepsalive()
		}

	}
	return calgoperftogoarray(perfResults), err
}

//GetRNNForwardTrainingAlgorithmMaxCount gets the max number of algorithms for rnnforward training algo
func (r *RNND) GetRNNForwardTrainingAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	stat := C.cudnnGetRNNForwardTrainingAlgorithmMaxCount(
		handle.x,
		r.descriptor,
		&count)
	if setkeepalive == true {
		keepsalivebuffer(handle, r)
	}
	return int32(count), Status(stat).error("GetRNNForwardTrainingAlgorithmMaxCount")
}

//FindRNNForwardTrainingAlgorithmEx finds and orders the performance of rnn algos for training returns that list with an error
func (r *RNND) FindRNNForwardTrainingAlgorithmEx(
	handle *Handle,
	seqLen int32, //input
	xD []*TensorD, //input
	x gocu.Mem, //input
	hxD *TensorD, //input: A fully packed tensor descriptor describing the initial hidden state of the RNN.
	hx gocu.Mem, //input
	cxD *TensorD, // :input A fully packed tensor descriptor describing the initial cell state for LSTM networks.
	cx gocu.Mem, //input
	wD *FilterD, //input
	w gocu.Mem, //input
	yD []*TensorD, //Input. An array of fully packed tensor descriptors describing the output from each recurrent iteration (one descriptor per iteration).
	y gocu.Mem, //output
	hyD *TensorD, //input
	hy gocu.Mem, //output
	cyD *TensorD,
	cy gocu.Mem, //output
	findIntensity float32, //input
	reqAlgocount int32, //input
	wspace gocu.Mem, ///input
	wspacesize uint,
	rspace gocu.Mem, //input/output
	rspacesize uint,

) ([]AlgorithmPerformance, error) {
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)

	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	if wspace == nil {
		err := Status(C.cudnnFindRNNForwardTrainingAlgorithmEx(
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
			C.float(findIntensity),
			C.int(reqAlgocount),
			&actualcount,
			&perfresults[0],
			nil,
			C.size_t(0),
			rspace.Ptr(),
			C.size_t(rspacesize),
		)).error("FindRNNForwardTrainingAlgorithmEx")

		return calgoperftogoarray(perfresults), err
	}
	err := Status(C.cudnnFindRNNForwardTrainingAlgorithmEx(
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
		C.float(findIntensity),
		C.int(reqAlgocount),
		&actualcount,
		&perfresults[0],
		wspace.Ptr(),
		C.size_t(wspacesize),
		rspace.Ptr(),
		C.size_t(rspacesize),
	)).error("FindRNNForwardTrainingAlgorithmEx")

	return calgoperftogoarray(perfresults), err
}

//GetRNNBackwardDataAlgorithmMaxCount gets the max number of algorithms for the back prop rnn
func (r *RNND) GetRNNBackwardDataAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNBackwardDataAlgorithmMaxCount(
		handle.x,
		r.descriptor,
		&count,
	)).error("GetRNNBackwardDataAlgorithmMaxCount")
	if setkeepalive == true {
		keepsalivebuffer(r, handle)
	}
	return int32(count), err
}

//FindRNNBackwardDataAlgorithmEx finds a list of algos for backprop this passes like 26 parameters and pointers and stuff so watch out.
func (r *RNND) FindRNNBackwardDataAlgorithmEx(
	handle *Handle,

	seqLen int32,

	yD []*TensorD, //an array of fully packed tensor descriptors
	y gocu.Mem,

	dyD []*TensorD, //an array of fully packed tensor descriptors
	dy gocu.Mem,

	dhyD *TensorD, //fully packed tensor descriptor describing the gradients at the final hidden state of the RNN
	dhy gocu.Mem,

	dcyD *TensorD, // fully packed tensor descriptor describing the gradients at the final cell state of the RNN.
	dcy gocu.Mem,

	wD *FilterD,
	w gocu.Mem,

	hxD *TensorD, // A fully packed tensor descriptor describing the initial hidden state of the RNN.
	hx gocu.Mem,

	cxD *TensorD, //A fully packed tensor descriptor describing the initial cell state for LSTM networks.
	cx gocu.Mem,

	dxD []*TensorD, //
	dx gocu.Mem,

	dhxD *TensorD, //A fully packed tensor descriptor describing the gradient at the initial hidden state of the RNN.
	dhx gocu.Mem,

	dcxD *TensorD, // A fully packed tensor descriptor describing the gradient at the initial cell state of the RNN.
	dcx gocu.Mem,

	findIntensity float32,
	reqAlgocount int32,
	wspace gocu.Mem, ///input
	wspacesize uint,
	rspace gocu.Mem, //input/output
	rspacesize uint,

) ([]AlgorithmPerformance, error) {
	cyD := tensorDArrayToC(yD)
	cdyD := tensorDArrayToC(dyD)
	cdxD := tensorDArrayToC(dxD)
	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	if wspace == nil {
		err := Status(C.cudnnFindRNNBackwardDataAlgorithmEx(
			handle.x,
			r.descriptor,
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

			nil,
			C.size_t(0),
			rspace.Ptr(),
			C.size_t(rspacesize),
			//31 total?
		)).error("FindRNNBackwardDataAlgorithmEx")
		return calgoperftogoarray(perfresults), err
	}
	err := Status(C.cudnnFindRNNBackwardDataAlgorithmEx(
		handle.x,
		r.descriptor,
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
		C.size_t(wspacesize),
		rspace.Ptr(),
		C.size_t(rspacesize),
		//31 total?
	)).error("FindRNNBackwardDataAlgorithmEx")
	return calgoperftogoarray(perfresults), err
}

//GetRNNBackwardWeightsAlgorithmMaxCount gets the max number of algos for weights
func (r *RNND) GetRNNBackwardWeightsAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
		handle.x,
		r.descriptor,
		&count,
	)).error("GetRNNBackwardWeightsAlgorithmMaxCount")
	if setkeepalive == true {
		keepsalivebuffer(handle, r)
	}
	return int32(count), err
}

//FindRNNBackwardWeightsAlgorithmEx returns some algos and their performance and stuff
func (r *RNND) FindRNNBackwardWeightsAlgorithmEx(
	handle *Handle,
	seqLen int32,
	xD []*TensorD,
	x gocu.Mem,
	hxD *TensorD, //Initial Hidden State
	hx gocu.Mem,
	yD []*TensorD,
	y gocu.Mem,
	findIntensity float32, //unused for future use
	reqAlgocount int32, //the max number of elements
	wspace gocu.Mem,
	wspacesize uint,
	dwD *FilterD,
	dw gocu.Mem,
	rspace gocu.Mem,
	rspacesize uint,

) ([]AlgorithmPerformance, error) {
	var actualcount C.int
	inCxD := tensorDArrayToC(xD)
	inCyD := tensorDArrayToC(yD)
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	if wspace == nil {
		err := Status(C.cudnnFindRNNBackwardWeightsAlgorithmEx(
			handle.x,
			r.descriptor,
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

			nil,
			C.size_t(0),

			dwD.descriptor,
			dw.Ptr(),

			rspace.Ptr(),
			C.size_t(rspacesize),
		)).error("FindRNNBackwardWeightsAlgorithmEx")

		return calgoperftogoarray(perfresults), err
	}
	err := Status(C.cudnnFindRNNBackwardWeightsAlgorithmEx(
		handle.x,
		r.descriptor,
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
		C.size_t(wspacesize),

		dwD.descriptor,
		dw.Ptr(),

		rspace.Ptr(),
		C.size_t(rspacesize),
	)).error("FindRNNBackwardWeightsAlgorithmEx")

	return calgoperftogoarray(perfresults), err
}

//RNNForwardInference is the forward inference
func (r *RNND) RNNForwardInference(
	handle *Handle,
	seqLength int32,
	xD []*TensorD,
	x gocu.Mem,
	hxD *TensorD,
	hx gocu.Mem,
	cxD *TensorD,
	cx gocu.Mem,
	wD *FilterD,
	w gocu.Mem,
	yD []*TensorD,
	y gocu.Mem,
	hyD TensorD,
	hy gocu.Mem,
	cyD TensorD,
	cy gocu.Mem,
	wspace gocu.Mem,
	wspacesize uint,

) error {
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	if setkeepalive == true {
		keepsalivebuffer(handle, r, x, y, hxD, hx, hxD, hx, wD, w, y, wspace, cy, cyD, hy, hyD, cx, cxD)
		for i := range xD {
			xD[i].keepsalive()
		}
		for i := range yD {
			yD[i].keepsalive()
		}
	}
	if wspace == nil {
		return Status(C.cudnnRNNForwardInference(
			handle.x,
			r.descriptor,
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
			nil,
			C.size_t(0),
		)).error("RNNForwardInference")
	}
	return Status(C.cudnnRNNForwardInference(
		handle.x,
		r.descriptor,
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
		wspace.Ptr(),
		C.size_t(wspacesize),
	)).error("RNNForwardInference")
}

//RNNForwardTraining is the forward algo for an RNN
func (r *RNND) RNNForwardTraining(
	handle *Handle,
	seqLen int32,
	xD []*TensorD,
	x gocu.Mem,
	hxD *TensorD,
	hx gocu.Mem,
	cxD *TensorD,
	cx gocu.Mem,
	wD *FilterD,
	w gocu.Mem,
	yD []*TensorD,
	y gocu.Mem,
	hyD *TensorD,
	hy gocu.Mem,
	cyD *TensorD,
	cy gocu.Mem,
	wspace gocu.Mem,
	wspacesize uint,
	rspace gocu.Mem,
	rspacesize uint,
) error {
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	if wspace == nil {
		return Status(C.cudnnRNNForwardTraining(
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
			nil,
			C.size_t(0),
			rspace.Ptr(),
			C.size_t(rspacesize),
		)).error("RNNForwardTraining")
	}
	return Status(C.cudnnRNNForwardTraining(
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
		C.size_t(wspacesize),
		rspace.Ptr(),
		C.size_t(rspacesize),
	)).error("RNNForwardTraining")

}

//RNNBackwardData is the backward algo for an RNN
func (r *RNND) RNNBackwardData(
	handle *Handle,
	seqLen int32,

	yD []*TensorD,
	y gocu.Mem,

	dyD []*TensorD,
	dy gocu.Mem,

	dhyD *TensorD,
	dhy gocu.Mem,

	dcyD *TensorD,
	dcy gocu.Mem,

	wD *FilterD,
	w gocu.Mem,

	hxD *TensorD,
	hx gocu.Mem,

	cxD *TensorD,
	cx gocu.Mem,

	dxD []*TensorD,
	dx gocu.Mem,

	dhxD *TensorD,
	dhx gocu.Mem,

	dcxD *TensorD,
	dcx gocu.Mem,

	wspace gocu.Mem,
	wspacesize uint,
	rspace gocu.Mem,
	rspacesize uint,
) error {
	tocdxD := tensorDArrayToC(dxD)
	tocdyD := tensorDArrayToC(dyD)
	tocyD := tensorDArrayToC(yD)
	if wspace == nil {
		return Status(C.cudnnRNNBackwardData(
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
			nil,
			C.size_t(wspacesize),
			rspace.Ptr(),
			C.size_t(rspacesize),
		)).error("RNNBackwardData")

	}
	return Status(C.cudnnRNNBackwardData(
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
		C.size_t(wspacesize),
		rspace.Ptr(),
		C.size_t(rspacesize),
	)).error("RNNBackwardData")

}

//BackwardWeights does the backward weight function
func (r *RNND) BackwardWeights(
	handle *Handle,
	seqLen int32,
	xD []*TensorD,
	x gocu.Mem,
	hxD *TensorD,
	hx gocu.Mem,
	yD []*TensorD,
	y gocu.Mem,
	wspace gocu.Mem,
	wspacesize uint,
	dwD *FilterD,
	dw gocu.Mem,
	rspace gocu.Mem,
	rspacesize uint,
) error {
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	if wspace == nil {

		return Status(C.cudnnRNNBackwardWeights(
			handle.x,
			r.descriptor,
			C.int(seqLen),
			&tocxD[0],
			x.Ptr(),
			hxD.descriptor,
			hx.Ptr(),
			&tocyD[0],
			y.Ptr(),
			nil,
			C.size_t(0),
			dwD.descriptor,
			dw.Ptr(),
			rspace.Ptr(),
			C.size_t(rspacesize),
		)).error("BackwardWeights")

	}
	return Status(C.cudnnRNNBackwardWeights(
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
		C.size_t(wspacesize),
		dwD.descriptor,
		dw.Ptr(),
		rspace.Ptr(),
		C.size_t(rspacesize),
	)).error("BackwardWeights")

}

/*


FLAGS

*/

//RNNFlags holds all the RNN flags
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
