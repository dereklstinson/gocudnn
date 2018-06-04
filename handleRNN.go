package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//GetRNNForwardInferenceAlgorithmMaxCount returns the maxcount and error
func (handle *Handle) GetRNNForwardInferenceAlgorithmMaxCount(
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
func (handle *Handle) FindRNNForwardInferenceAlgorithmEx(
	rnnD *RNND,
	seqlength int32,
	xD *TensorD,
	x Memer,
	hxD *TensorD,
	hx Memer,
	cxD *TensorD,
	cx Memer,
	wD *FilterD,
	w Memer,
	yD *TensorD,
	y Memer,
	hyD *TensorD,
	hy Memer,
	cyD *TensorD,
	cy Memer,
	findIntensity float32,
	algocount int32,
	wspace Memer,

) ([]AlgorithmPerformance, error) {
	var retactAlgoCount C.int
	perfResults := make([]C.cudnnAlgorithmPerformance_t, algocount)
	reqcount := C.int(algocount)
	err := Status(C.cudnnFindRNNForwardInferenceAlgorithmEx(
		handle.x,
		rnnD.descriptor,
		C.int(seqlength),
		&xD.descriptor,
		x.Ptr(),
		hxD.descriptor,
		hx.Ptr(),
		cxD.descriptor,
		cx.Ptr(),
		wD.descriptor,
		w.Ptr(),
		&yD.descriptor,
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

	results := make([]AlgorithmPerformance, C.int(retactAlgoCount))
	for i := 0; i < len(results); i++ {
		results[i] = AlgorithmPerformance(perfResults[i])
	}
	return results, err
}

//GetRNNForwardTrainingAlgorithmMaxCount gets the max number of algorithms for rnnforward training algo
func (handle *Handle) GetRNNForwardTrainingAlgorithmMaxCount(rnn RNND) (int32, error) {
	var count C.int
	stat := C.cudnnGetRNNForwardTrainingAlgorithmMaxCount(
		handle.x,
		rnn.descriptor,
		&count)
	return int32(count), Status(stat).error("GetRNNForwardTrainingAlgorithmMaxCount")
}

//FindRNNForwardTrainingAlgorithmEx finds and orders the performance of rnn algos for training returns that list with an error
func (handle *Handle) FindRNNForwardTrainingAlgorithmEx(
	rnn *RNND,
	seqLen int32,
	xD *TensorD,
	x Memer,
	hxD *TensorD,
	hx Memer,
	cxD *TensorD,
	cx Memer,
	wD *FilterD,
	w Memer,
	yD *TensorD,
	y Memer,
	hyD *TensorD,
	hy Memer,
	cyD *TensorD,
	cy Memer,
	findIntensity float32,
	reqAlgocount int32,
	wspace Memer,
	rspace Memer,

) ([]AlgorithmPerformance, error) {
	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	err := Status(C.cudnnFindRNNForwardTrainingAlgorithmEx(
		handle.x,
		rnn.descriptor,
		C.int(seqLen),
		&xD.descriptor,
		x.Ptr(),
		hxD.descriptor,
		hx.Ptr(),
		cxD.descriptor,
		cx.Ptr(),
		wD.descriptor,
		w.Ptr(),
		&yD.descriptor,
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
	results := make([]AlgorithmPerformance, actualcount)
	for i := 0; i < len(results); i++ {
		results[i] = AlgorithmPerformance(perfresults[i])
	}
	return results, err
}

//GetRNNBackwardDataAlgorithmMaxCount gets the max number of algorithms for the back prop rnn
func (handle *Handle) GetRNNBackwardDataAlgorithmMaxCount(rnnd *RNND) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNBackwardDataAlgorithmMaxCount(
		handle.x,
		rnnd.descriptor,
		&count,
	)).error("GetRNNBackwardDataAlgorithmMaxCount")
	return int32(count), err
}

//FindRNNBackwardDataAlgorithmEx finds a list of algos for backprop this passes like 26 parameters and pointers and stuff so watch out.
func (handle *Handle) FindRNNBackwardDataAlgorithmEx(
	rnnD *RNND,
	seqLen int32,

	yD *TensorD,
	y Memer,

	dyD *TensorD,
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

	dxD *TensorD,
	dx Memer,

	dhxD *TensorD,
	dhx Memer,

	dcxD *TensorD,
	dcx Memer,

	findIntensity float32,
	reqAlgocount int32,
	wspace Memer,
	rspace Memer,

) ([]AlgorithmPerformance, error) {
	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	err := Status(C.cudnnFindRNNBackwardDataAlgorithmEx(
		handle.x,
		rnnD.descriptor,
		C.int(seqLen),

		&yD.descriptor,
		y.Ptr(),

		&dyD.descriptor,
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

		&dxD.descriptor,
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
	results := make([]AlgorithmPerformance, actualcount)
	for i := 0; i < len(results); i++ {
		results[i] = AlgorithmPerformance(perfresults[i])
	}
	return results, err
}

//GetRNNBackwardWeightsAlgorithmMaxCount gets the max number of algos for weights
func (handle *Handle) GetRNNBackwardWeightsAlgorithmMaxCount(rnnD *RNND) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
		handle.x,
		rnnD.descriptor,
		&count,
	)).error("GetRNNBackwardWeightsAlgorithmMaxCount")
	return int32(count), err
}

//FindRNNBackwardWeightsAlgorithmEx returns some algos and their performance and stuff
func (handle *Handle) FindRNNBackwardWeightsAlgorithmEx(
	rnnD *RNND,
	seqLen int32,
	xD *TensorD,
	x Memer,
	hxD TensorD,
	hx Memer,
	yD *TensorD,
	y Memer,
	findIntensity float32,
	reqAlgocount int32,
	wspace Memer,
	dwD *FilterD,
	dw Memer,
	rspace Memer,

) ([]AlgorithmPerformance, error) {
	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	err := Status(C.cudnnFindRNNBackwardWeightsAlgorithmEx(
		handle.x,
		rnnD.descriptor,
		C.int(seqLen),
		&xD.descriptor,
		x.Ptr(),
		hxD.descriptor,
		hx.Ptr(),
		&yD.descriptor,
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
	results := make([]AlgorithmPerformance, actualcount)
	for i := 0; i < len(results); i++ {
		results[i] = AlgorithmPerformance(perfresults[i])
	}
	return results, err
}
