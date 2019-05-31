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
	"github.com/dereklstinson/cutil"
)

//Algo returns an Algorithm used for
func (r RNNAlgo) Algo() Algorithm {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforRNN(&algorithm, r.c())
	return Algorithm(algorithm)
}

//AlgorithmPerformance go typed C.cudnnAlgorithmPerformance_t
type AlgorithmPerformance struct {
	descriptor C.cudnnAlgorithmPerformance_t
	index      C.int
	gogc       bool
}

//RNND  holdes Rnn descriptor
type RNND struct {
	descriptor C.cudnnRNNDescriptor_t
	gogc       bool
}

func rrndArrayToCarray(input []RNND) []C.cudnnRNNDescriptor_t {
	array := make([]C.cudnnRNNDescriptor_t, len(input))
	for i := 0; i < len(input); i++ {
		array[i] = input[i].descriptor
	}
	return array
}

//CreateRNNDescriptor creates an RNND descriptor
func CreateRNNDescriptor() (desc *RNND, err error) {
	desc = new(RNND)
	err = Status(C.cudnnCreateRNNDescriptor(&desc.descriptor)).error("CreateRNNDescriptor")
	if err != nil {
		return nil, err
	}

	if setfinalizer == true {
		runtime.SetFinalizer(desc, destroyrnnddescriptor)
	}
	desc.gogc = true
	return desc, err
}

//Destroy destroys the descriptor
//Right now this doesn't work because gocudnn uses go's GC.
func (r *RNND) Destroy() error {
	if setfinalizer || r.gogc {
		return nil
	}
	return destroyrnnddescriptor(r)
}
func destroyrnnddescriptor(r *RNND) error {
	return Status(C.cudnnDestroyRNNDescriptor(r.descriptor)).error("DestroyDescriptor-rnn")
}

//Set sets the rnndesctiptor
func (r *RNND) Set(
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
	)).error("(*RNND)Set")
}

//SetProjectionLayers sets the rnnprojection layers
func (r *RNND) SetProjectionLayers(
	handle *Handle,
	recProjsize int32,
	outProjSize int32,
) error {

	return Status(C.cudnnSetRNNProjectionLayers(
		handle.x,
		r.descriptor,
		C.int(recProjsize),
		C.int(outProjSize),
	)).error("SetProjectionLayers")
}

//GetProjectionLayers sets the rnnprojection layers
func (r *RNND) GetProjectionLayers(
	handle *Handle,
) (int32, int32, error) {

	var rec, out C.int

	err := Status(C.cudnnGetRNNProjectionLayers(
		handle.x,
		r.descriptor,
		&rec,
		&out,
	)).error("GetProjectionLayers")
	return int32(rec), int32(out), err
}

//SetAlgorithmDescriptor sets the RNNalgorithm
func (r *RNND) SetAlgorithmDescriptor(
	handle *Handle,
	algo *AlgorithmD,
) error {

	return Status(C.cudnnSetRNNAlgorithmDescriptor(handle.x, r.descriptor, algo.descriptor)).error("SetAlgorithmDescriptor")
}

//Get gets RNND values that were set
func (r *RNND) Get(
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

//GetWorkspaceSIB gets the RNN workspace size (WOW!)
func (r *RNND) GetWorkspaceSIB(
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

	return uint(sizeinbytes), err
}

//GetReserveSIB gets the training reserve size
func (r *RNND) GetReserveSIB(
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

	return uint(sizeinbytes), err
}

//GetParamsSIB gets the training reserve size
func (r *RNND) GetParamsSIB(
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

	return uint(sizeinbytes), err
}

//GetLinLayerMatrixParams gets the parameters of the layer matrix
func (r *RNND) GetLinLayerMatrixParams(
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
	wD *FilterD, w cutil.Mem,
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

//GetLinLayerMatrixParamsUS is like GetLinLayerMatrixParamsUS but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) GetLinLayerMatrixParamsUS(
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
	wD *FilterD, w unsafe.Pointer,
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

) (FilterD, cutil.Mem, error) {
	var linLayerMatDesc FilterD
	var linLayerMat unsafe.Pointer
	err := Status(C.cudnnGetRNNLinLayerMatrixParams(
		handle.x,
		r.descriptor,
		C.int(pseudoLayer),
		xD.descriptor,
		wD.descriptor, w,
		C.int(linlayerID),
		linLayerMatDesc.descriptor,
		&linLayerMat,
	)).error("GetRNNLinLayerMatrixParams")

	return linLayerMatDesc, gocu.WrapUnsafe(linLayerMat), err
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
	w cutil.Mem,
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

) (BiasD *FilterD, Bias cutil.Mem, err error) {
	BiasD, err = CreateFilterDescriptor()
	if err != nil {
		return nil, nil, err
	}
	Bias = new(gocu.CudaPtr)
	err = Status(C.cudnnGetRNNLinLayerBiasParams(
		handle.x,
		r.descriptor,
		C.int(pseudoLayer),
		xD.descriptor,
		wD.descriptor,
		w.Ptr(),
		C.int(linlayerID),
		BiasD.descriptor,
		Bias.DPtr(),
	)).error("GetRNNLinLayerBiasParams")

	return BiasD, Bias, err
}

//GetRNNLinLayerBiasParamsUS is like GetRNNLinLayerBiasParams but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) GetRNNLinLayerBiasParamsUS(
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
	w unsafe.Pointer,
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

) (BiasD *FilterD, Bias unsafe.Pointer, err error) {
	BiasD, err = CreateFilterDescriptor()
	if err != nil {
		return nil, nil, err
	}
	err = Status(C.cudnnGetRNNLinLayerBiasParams(
		handle.x,
		r.descriptor,
		C.int(pseudoLayer),
		xD.descriptor,
		wD.descriptor, w,
		C.int(linlayerID),
		BiasD.descriptor,
		&Bias,
	)).error("GetRNNLinLayerBiasParams")

	return BiasD, Bias, err
}

//PersistentRNNPlan holds  C.cudnnPersistentRNNPlan_t
type PersistentRNNPlan struct {
	plan C.cudnnPersistentRNNPlan_t
	gogc bool
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

//DestroyPersistentRNNPlan destroys the C.cudnnPersistentRNNPlan_t in the PersistentRNNPlan struct
func (p *PersistentRNNPlan) DestroyPersistentRNNPlan() error {
	if setfinalizer || p.gogc {
		return nil
	}
	return destroypersistantrnnplan(p)
}
func destroypersistantrnnplan(p *PersistentRNNPlan) error {
	return Status(C.cudnnDestroyPersistentRNNPlan(p.plan)).error("DestroyPersistentRNNPlan")
}

/*

RNN Fluncs



*/

//RNNForwardInference is the forward inference
func (r *RNND) RNNForwardInference(
	handle *Handle,
	xD []*TensorD, x cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	cxD *TensorD, cx cutil.Mem,
	wD *FilterD, w cutil.Mem,
	yD []*TensorD, y cutil.Mem,
	hyD TensorD, hy cutil.Mem,
	cyD TensorD, cy cutil.Mem,
	wspace cutil.Mem, wspacesize uint,

) error {
	seqLength := len(xD)
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)

	if wspace == nil {
		return Status(C.cudnnRNNForwardInference(
			handle.x,
			r.descriptor,
			C.int(seqLength),
			&tocxD[0], x.Ptr(),
			hxD.descriptor, hx.Ptr(),
			cxD.descriptor, cx.Ptr(),
			wD.descriptor, w.Ptr(),
			&tocyD[0], y.Ptr(),
			hyD.descriptor, hy.Ptr(),
			cyD.descriptor, cy.Ptr(),
			nil,
			C.size_t(0),
		)).error("RNNForwardInference")
	}
	return Status(C.cudnnRNNForwardInference(
		handle.x,
		r.descriptor,
		C.int(seqLength),
		&tocxD[0], x.Ptr(),
		hxD.descriptor, hx.Ptr(),
		cxD.descriptor, cx.Ptr(),
		wD.descriptor, w.Ptr(),
		&tocyD[0], y.Ptr(),
		hyD.descriptor, hy.Ptr(),
		cyD.descriptor, cy.Ptr(),
		wspace.Ptr(), C.size_t(wspacesize),
	)).error("RNNForwardInference")
}

//RNNForwardInferenceUS is like RNNForwardInference but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) RNNForwardInferenceUS(
	handle *Handle,
	xD []*TensorD, x unsafe.Pointer,
	hxD *TensorD, hx unsafe.Pointer,
	cxD *TensorD, cx unsafe.Pointer,
	wD *FilterD, w unsafe.Pointer,
	yD []*TensorD, y unsafe.Pointer,
	hyD TensorD, hy unsafe.Pointer,
	cyD TensorD, cy unsafe.Pointer,
	wspace unsafe.Pointer, wspacesize uint,

) error {
	seqLength := len(xD)
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)

	return Status(C.cudnnRNNForwardInference(
		handle.x,
		r.descriptor,
		C.int(seqLength),
		&tocxD[0], x,
		hxD.descriptor, hx,
		cxD.descriptor, cx,
		wD.descriptor, w,
		&tocyD[0], y,
		hyD.descriptor, hy,
		cyD.descriptor, cy,
		wspace, C.size_t(wspacesize),
	)).error("RNNForwardInference")
}

//RNNForwardTraining is the forward algo for an RNN
func (r *RNND) RNNForwardTraining(
	handle *Handle,
	xD []*TensorD, x cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	cxD *TensorD, cx cutil.Mem,
	wD *FilterD, w cutil.Mem,
	yD []*TensorD, y cutil.Mem,
	hyD *TensorD, hy cutil.Mem,
	cyD *TensorD, cy cutil.Mem,
	wspace cutil.Mem, wspacesize uint,
	rspace cutil.Mem, rspacesize uint,
) error {
	seqLen := len(xD)
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	if wspace == nil {
		return Status(C.cudnnRNNForwardTraining(
			handle.x,
			r.descriptor,
			C.int(seqLen),
			&tocxD[0], x.Ptr(),
			hxD.descriptor, hx.Ptr(),
			cxD.descriptor, cx.Ptr(),
			wD.descriptor, w.Ptr(),
			&tocyD[0], y.Ptr(),
			hyD.descriptor, hy.Ptr(),
			cyD.descriptor, cy.Ptr(),
			nil, C.size_t(0),
			rspace.Ptr(), C.size_t(rspacesize),
		)).error("RNNForwardTraining")
	}
	return Status(C.cudnnRNNForwardTraining(
		handle.x,
		r.descriptor,
		C.int(seqLen),
		&tocxD[0], x.Ptr(),
		hxD.descriptor, hx.Ptr(),
		cxD.descriptor, cx.Ptr(),
		wD.descriptor, w.Ptr(),
		&tocyD[0], y.Ptr(),
		hyD.descriptor, hy.Ptr(),
		cyD.descriptor, cy.Ptr(),
		wspace.Ptr(), C.size_t(wspacesize),
		rspace.Ptr(), C.size_t(rspacesize),
	)).error("RNNForwardTraining")

}

//RNNForwardTrainingUS is like RNNForwardTraining but using unsafe.Pointer instead of cutil.Mem
func (r *RNND) RNNForwardTrainingUS(
	handle *Handle,
	xD []*TensorD, x unsafe.Pointer,
	hxD *TensorD, hx unsafe.Pointer,
	cxD *TensorD, cx unsafe.Pointer,
	wD *FilterD, w unsafe.Pointer,
	yD []*TensorD, y unsafe.Pointer,
	hyD *TensorD, hy unsafe.Pointer,
	cyD *TensorD, cy unsafe.Pointer,
	wspace unsafe.Pointer, wspacesize uint,
	rspace unsafe.Pointer, rspacesize uint,
) error {
	seqLen := len(xD)
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)

	return Status(C.cudnnRNNForwardTraining(
		handle.x,
		r.descriptor,
		C.int(seqLen),
		&tocxD[0], x,
		hxD.descriptor, hx,
		cxD.descriptor, cx,
		wD.descriptor, w,
		&tocyD[0], y,
		hyD.descriptor, hy,
		cyD.descriptor, cy,
		wspace, C.size_t(wspacesize),
		rspace, C.size_t(rspacesize),
	)).error("RNNForwardTraining")

}

//RNNBackwardData is the backward algo for an RNN
func (r *RNND) RNNBackwardData(
	handle *Handle,
	yD []*TensorD, y cutil.Mem,
	dyD []*TensorD, dy cutil.Mem,
	dhyD *TensorD, dhy cutil.Mem,
	dcyD *TensorD, dcy cutil.Mem,
	wD *FilterD, w cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	cxD *TensorD, cx cutil.Mem,
	dxD []*TensorD, dx cutil.Mem,
	dhxD *TensorD, dhx cutil.Mem,
	dcxD *TensorD, dcx cutil.Mem,
	wspace cutil.Mem, wspacesize uint,
	rspace cutil.Mem, rspacesize uint,
) error {
	seqLen := len(yD)
	tocdxD := tensorDArrayToC(dxD)
	tocdyD := tensorDArrayToC(dyD)
	tocyD := tensorDArrayToC(yD)
	if wspace == nil {
		return Status(C.cudnnRNNBackwardData(
			handle.x,
			r.descriptor,
			C.int(seqLen),
			&tocyD[0], y.Ptr(),
			&tocdyD[0], dy.Ptr(),
			dhyD.descriptor, dhy.Ptr(),
			dcyD.descriptor, dcy.Ptr(),
			wD.descriptor, w.Ptr(),
			hxD.descriptor, hx.Ptr(),
			cxD.descriptor, cx.Ptr(),
			&tocdxD[0], dx.Ptr(),
			dhxD.descriptor, dhx.Ptr(),
			dcxD.descriptor, dcx.Ptr(),
			nil, C.size_t(wspacesize),
			rspace.Ptr(), C.size_t(rspacesize),
		)).error("RNNBackwardData")

	}
	return Status(C.cudnnRNNBackwardData(
		handle.x,
		r.descriptor,
		C.int(seqLen),
		&tocyD[0], y.Ptr(),
		&tocdyD[0], dy.Ptr(),
		dhyD.descriptor, dhy.Ptr(),
		dcyD.descriptor, dcy.Ptr(),
		wD.descriptor, w.Ptr(),
		hxD.descriptor, hx.Ptr(),
		cxD.descriptor, cx.Ptr(),
		&tocdxD[0], dx.Ptr(),
		dhxD.descriptor, dhx.Ptr(),
		dcxD.descriptor, dcx.Ptr(),
		wspace.Ptr(), C.size_t(wspacesize),
		rspace.Ptr(), C.size_t(rspacesize),
	)).error("RNNBackwardData")

}

//RNNBackwardDataUS is like RNNBackwardData but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) RNNBackwardDataUS(
	handle *Handle,
	yD []*TensorD, y unsafe.Pointer,
	dyD []*TensorD, dy unsafe.Pointer,
	dhyD *TensorD, dhy unsafe.Pointer,
	dcyD *TensorD, dcy unsafe.Pointer,
	wD *FilterD, w unsafe.Pointer,
	hxD *TensorD, hx unsafe.Pointer,
	cxD *TensorD, cx unsafe.Pointer,
	dxD []*TensorD, dx unsafe.Pointer,
	dhxD *TensorD, dhx unsafe.Pointer,
	dcxD *TensorD, dcx unsafe.Pointer,
	wspace unsafe.Pointer, wspacesize uint,
	rspace unsafe.Pointer, rspacesize uint,
) error {
	seqLen := len(yD)
	tocdxD := tensorDArrayToC(dxD)
	tocdyD := tensorDArrayToC(dyD)
	tocyD := tensorDArrayToC(yD)

	return Status(C.cudnnRNNBackwardData(
		handle.x,
		r.descriptor,
		C.int(seqLen),
		&tocyD[0], y,
		&tocdyD[0], dy,
		dhyD.descriptor, dhy,
		dcyD.descriptor, dcy,
		wD.descriptor, w,
		hxD.descriptor, hx,
		cxD.descriptor, cx,
		&tocdxD[0], dx,
		dhxD.descriptor, dhx,
		dcxD.descriptor, dcx,
		wspace, C.size_t(wspacesize),
		rspace, C.size_t(rspacesize),
	)).error("RNNBackwardData")

}

//BackwardWeights does the backward weight function
func (r *RNND) BackwardWeights(
	handle *Handle,
	xD []*TensorD, x cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	yD []*TensorD, y cutil.Mem,
	wspace cutil.Mem, wspacesize uint,
	dwD *FilterD, dw cutil.Mem,
	rspace cutil.Mem, rspacesize uint,
) error {
	seqLen := len(yD)
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	if wspace == nil {

		return Status(C.cudnnRNNBackwardWeights(
			handle.x,
			r.descriptor,
			C.int(seqLen),
			&tocxD[0], x.Ptr(),
			hxD.descriptor, hx.Ptr(),
			&tocyD[0], y.Ptr(),
			nil, C.size_t(0),
			dwD.descriptor, dw.Ptr(),
			rspace.Ptr(), C.size_t(rspacesize),
		)).error("BackwardWeights")

	}
	return Status(C.cudnnRNNBackwardWeights(
		handle.x,
		r.descriptor,
		C.int(seqLen),
		&tocxD[0], x.Ptr(),
		hxD.descriptor, hx.Ptr(),
		&tocyD[0], y.Ptr(),
		wspace.Ptr(), C.size_t(wspacesize),
		dwD.descriptor, dw.Ptr(),
		rspace.Ptr(), C.size_t(rspacesize),
	)).error("BackwardWeights")

}

//BackwardWeightsUS is like BackwardWeights but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) BackwardWeightsUS(
	handle *Handle,
	xD []*TensorD, x unsafe.Pointer,
	hxD *TensorD, hx unsafe.Pointer,
	yD []*TensorD, y unsafe.Pointer,
	wspace unsafe.Pointer, wspacesize uint,
	dwD *FilterD, dw unsafe.Pointer,
	rspace unsafe.Pointer, rspacesize uint,
) error {
	seqLen := len(yD)
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)

	return Status(C.cudnnRNNBackwardWeights(
		handle.x,
		r.descriptor,
		C.int(seqLen),
		&tocxD[0], x,
		hxD.descriptor, hx,
		&tocyD[0], y,
		wspace, C.size_t(wspacesize),
		dwD.descriptor, dw,
		rspace, C.size_t(rspacesize),
	)).error("BackwardWeights")

}

/*


FLAGS

*/

//RNNFlags holds all the RNN flags
type RNNFlags struct {
	Mode      RNNmode
	Algo      RNNAlgo
	Direction DirectionMode
	Input     RNNInputMode
}

//RNNmode is used for flags exposing the flags through methods
type RNNmode C.cudnnRNNMode_t

func (r RNNmode) c() C.cudnnRNNMode_t { return C.cudnnRNNMode_t(r) }

//Relu sets r to and returns RNNMode(C.CUDNN_RNN_RELU)
func (r *RNNmode) Relu() RNNmode { *r = RNNmode(C.CUDNN_RNN_RELU); return *r }

//Tanh  sets r to and returns RNNmode(C.CUDNN_RNN_TANH)
func (r *RNNmode) Tanh() RNNmode { *r = RNNmode(C.CUDNN_RNN_RELU); return *r }

//Lstm  sets r to and returns RNNmode(C.CUDNN_LSTM)
func (r *RNNmode) Lstm() RNNmode { *r = RNNmode(C.CUDNN_RNN_RELU); return *r }

//Gru  sets r to and returns RNNmode(C.CUDNN_GRU)
func (r *RNNmode) Gru() RNNmode { *r = RNNmode(C.CUDNN_RNN_RELU); return *r }

//DirectionMode is used for flags and exposes flags of type through types methods
type DirectionMode C.cudnnDirectionMode_t

func (r DirectionMode) c() C.cudnnDirectionMode_t { return C.cudnnDirectionMode_t(r) }

//Uni sets r to and returns DirectionMode(C.CUDNN_UNIDIRECTIONAL)
func (r *DirectionMode) Uni() DirectionMode { *r = DirectionMode(C.CUDNN_UNIDIRECTIONAL); return *r }

//Bi sets r to and returns DirectionMode(C.CUDNN_BIDIRECTIONAL)
func (r *DirectionMode) Bi() DirectionMode { *r = DirectionMode(C.CUDNN_BIDIRECTIONAL); return *r }

/*
 *   RNN INPUT MODE FLAGS
 */

//RNNInputMode is used for flags and exposes the different flags through its methods
type RNNInputMode C.cudnnRNNInputMode_t

//Linear sets r to and returns RNNInputMode(C.CUDNN_LINEAR_INPUT)
func (r *RNNInputMode) Linear() RNNInputMode { *r = RNNInputMode(C.CUDNN_LINEAR_INPUT); return *r }

//Skip sets r to and returns RNNInputMode(C.CUDNN_SKIP_INPUT)
func (r *RNNInputMode) Skip() RNNInputMode      { *r = RNNInputMode(C.CUDNN_SKIP_INPUT); return *r }
func (r RNNInputMode) c() C.cudnnRNNInputMode_t { return C.cudnnRNNInputMode_t(r) }

/*
 *   RNN ALGO FLAGS
 */

//RNNAlgo s used for flags and exposes the different flags through its methods
type RNNAlgo C.cudnnRNNAlgo_t

//Standard sets r to and returns RNNAlgo( C.CUDNN_RNN_ALGO_STANDARD) flag
func (r *RNNAlgo) Standard() RNNAlgo { *r = RNNAlgo(C.CUDNN_RNN_ALGO_STANDARD); return *r }

//PersistStatic sets r to and returns RNNAlgo( C.CUDNN_RNN_ALGO_PERSIST_STATIC) flag
func (r *RNNAlgo) PersistStatic() RNNAlgo { *r = RNNAlgo(C.CUDNN_RNN_ALGO_PERSIST_STATIC); return *r }

//PersistDynamic sets r to and returns RNNAlgo( C.CUDNN_RNN_ALGO_PERSIST_DYNAMIC) flag
func (r *RNNAlgo) PersistDynamic() RNNAlgo { *r = RNNAlgo(C.CUDNN_RNN_ALGO_PERSIST_DYNAMIC); return *r }

//Count sets r to and returns RNNAlgo( C.CUDNN_RNN_ALGO_COUNT) flag
func (r *RNNAlgo) Count() RNNAlgo { *r = RNNAlgo(C.CUDNN_RNN_ALGO_COUNT); return *r }

func (r RNNAlgo) c() C.cudnnRNNAlgo_t { return C.cudnnRNNAlgo_t(r) }
