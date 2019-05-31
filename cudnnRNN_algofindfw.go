package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//FindRNNForwardInferenceAlgorithmEx finds the inference algorithmEx
func (r *RNND) FindRNNForwardInferenceAlgorithmEx(
	handle *Handle,
	xD []*TensorD, //Input. An array of fully packed tensor descriptors describing the input to each recurrent iteration (one descriptor per iteration).
	x cutil.Mem, //input
	hxD *TensorD, //Input. A fully packed tensor descriptor describing the initial hidden state of the RNN.
	hx cutil.Mem, //input
	cxD *TensorD, //Input. A fully packed tensor descriptor describing the initial cell state for LSTM networks.
	cx cutil.Mem, //input
	wD *FilterD, //Input. Handle to a previously initialized filter descriptor describing the weights for the RNN.
	w cutil.Mem, //Input
	yD []*TensorD, //input An array of fully packed tensor descriptors.
	y cutil.Mem, //Output Data pointer to GPU memory associated with the output tensor descriptor yDesc
	hyD *TensorD, //input  A fully packed tensor descriptor describing the final hidden state of the RNN.
	hy cutil.Mem, //Output. Data pointer to GPU memory associated with the tensor descriptor hyDesc. If
	cyD *TensorD, //Input. A fully packed tensor descriptor describing the final cell state for LSTM networks.
	cy cutil.Mem, //output
	findIntensity float32,
	wspace cutil.Mem, wspacesize uint,
) ([]AlgorithmPerformance, error) {
	algocount, err := r.getRNNForwardInferenceAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	seqLength := (C.int)(len(xD))
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	var retactAlgoCount C.int
	perfResults := make([]C.cudnnAlgorithmPerformance_t, algocount)

	err = Status(C.cudnnFindRNNForwardInferenceAlgorithmEx(
		handle.x,
		r.descriptor,
		seqLength,
		&tocxD[0], x.Ptr(),
		hxD.descriptor, hx.Ptr(),
		cxD.descriptor, cx.Ptr(),
		wD.descriptor, w.Ptr(),
		&tocyD[0], y.Ptr(),
		hyD.descriptor, hy.Ptr(),
		cyD.descriptor, cy.Ptr(),
		C.float(findIntensity),
		(C.int)(algocount),
		&retactAlgoCount,
		&perfResults[0],
		wspace.Ptr(), C.size_t(wspacesize),
	)).error("FindRNNForwardInferenceAlgorithmEx")

	return calgoperftogoarray(perfResults, setfinalizer), err
}

//FindRNNForwardInferenceAlgorithmExUS is like FindRNNForwardInferenceAlgorithmEx but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) FindRNNForwardInferenceAlgorithmExUS(
	handle *Handle,
	xD []*TensorD, //Input. An array of fully packed tensor descriptors describing the input to each recurrent iteration (one descriptor per iteration).
	x unsafe.Pointer, //input
	hxD *TensorD, //Input. A fully packed tensor descriptor describing the initial hidden state of the RNN.
	hx unsafe.Pointer, //input
	cxD *TensorD, //Input. A fully packed tensor descriptor describing the initial cell state for LSTM networks.
	cx unsafe.Pointer, //input
	wD *FilterD, //Input. Handle to a previously initialized filter descriptor describing the weights for the RNN.
	w unsafe.Pointer, //Input
	yD []*TensorD, //input An array of fully packed tensor descriptors.
	y unsafe.Pointer, //Output Data pointer to GPU memory associated with the output tensor descriptor yDesc
	hyD *TensorD, //input  A fully packed tensor descriptor describing the final hidden state of the RNN.
	hy unsafe.Pointer, //Output. Data pointer to GPU memory associated with the tensor descriptor hyDesc. If
	cyD *TensorD, //Input. A fully packed tensor descriptor describing the final cell state for LSTM networks.
	cy unsafe.Pointer, //output
	findIntensity float32,
	wspace unsafe.Pointer, wspacesize uint,
) ([]AlgorithmPerformance, error) {
	reqcount, err := r.getRNNForwardInferenceAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	seqLength := (C.int)(len(xD))
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)
	var retactAlgoCount C.int
	perfResults := make([]C.cudnnAlgorithmPerformance_t, reqcount)

	err = Status(C.cudnnFindRNNForwardInferenceAlgorithmEx(
		handle.x,
		r.descriptor,
		seqLength,
		&tocxD[0], x,
		hxD.descriptor, hx,
		cxD.descriptor, cx,
		wD.descriptor, w,
		&tocyD[0], y,
		hyD.descriptor, hy,
		cyD.descriptor, cy,
		C.float(findIntensity),
		(C.int)(reqcount),
		&retactAlgoCount,
		&perfResults[0],
		wspace, C.size_t(wspacesize),
	)).error("FindRNNForwardInferenceAlgorithmEx")

	return calgoperftogoarray(perfResults, setfinalizer), err
}

//GetRNNForwardTrainingAlgorithmMaxCount gets the max number of algorithms for rnnforward training algo
func (r *RNND) GetRNNForwardTrainingAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	stat := C.cudnnGetRNNForwardTrainingAlgorithmMaxCount(
		handle.x,
		r.descriptor,
		&count)

	return int32(count), Status(stat).error("GetRNNForwardTrainingAlgorithmMaxCount")
}

//FindRNNForwardTrainingAlgorithmEx finds and orders the performance of rnn Algorithm for training returns that list with an error
func (r *RNND) FindRNNForwardTrainingAlgorithmEx(
	handle *Handle,
	xD []*TensorD, //input
	x cutil.Mem, //input
	hxD *TensorD, //input: A fully packed tensor descriptor describing the initial hidden state of the RNN.
	hx cutil.Mem, //input
	cxD *TensorD, // :input A fully packed tensor descriptor describing the initial cell state for LSTM networks.
	cx cutil.Mem, //input
	wD *FilterD, //input
	w cutil.Mem, //input
	yD []*TensorD, //Input. An array of fully packed tensor descriptors describing the output from each recurrent iteration (one descriptor per iteration).
	y cutil.Mem, //output
	hyD *TensorD, //input
	hy cutil.Mem, //output
	cyD *TensorD,
	cy cutil.Mem, //output
	findIntensity float32, //input
	reqAlgocount int32, //input
	wspace cutil.Mem, ///input
	wspacesize uint,
	rspace cutil.Mem, //input/output
	rspacesize uint,

) ([]AlgorithmPerformance, error) {
	reqcount, err := r.getRNNForwardInferenceAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	seqLength := (C.int)(len(xD))
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)

	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	if wspace == nil {
		err = Status(C.cudnnFindRNNForwardTrainingAlgorithmEx(
			handle.x,
			r.descriptor,
			seqLength,
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
			C.int(reqcount),
			&actualcount,
			&perfresults[0],
			nil,
			C.size_t(0),
			rspace.Ptr(),
			C.size_t(rspacesize),
		)).error("FindRNNForwardTrainingAlgorithmEx")

		return calgoperftogoarray(perfresults, handle.gogc), err
	}
	err = Status(C.cudnnFindRNNForwardTrainingAlgorithmEx(
		handle.x,
		r.descriptor,
		seqLength,
		&tocxD[0], x.Ptr(),
		hxD.descriptor, hx.Ptr(),
		cxD.descriptor, cx.Ptr(),
		wD.descriptor, w.Ptr(),
		&tocyD[0], y.Ptr(),
		hyD.descriptor, hy.Ptr(),
		cyD.descriptor, cy.Ptr(),
		C.float(findIntensity),
		C.int(reqcount),
		&actualcount, &perfresults[0],
		wspace.Ptr(), C.size_t(wspacesize),
		rspace.Ptr(), C.size_t(rspacesize),
	)).error("FindRNNForwardTrainingAlgorithmEx")

	return calgoperftogoarray(perfresults, handle.gogc), err
}

//FindRNNForwardTrainingAlgorithmExUS is like FindRNNForwardTrainingAlgorithmEx but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) FindRNNForwardTrainingAlgorithmExUS(
	handle *Handle,
	xD []*TensorD, x unsafe.Pointer,
	hxD *TensorD, hx unsafe.Pointer,
	cxD *TensorD, cx unsafe.Pointer,
	wD *FilterD, w unsafe.Pointer,
	yD []*TensorD, y unsafe.Pointer,
	hyD *TensorD, hy unsafe.Pointer,
	cyD *TensorD, cy unsafe.Pointer,
	findIntensity float32, //input
	wspace unsafe.Pointer, wspacesize uint,
	rspace unsafe.Pointer, rspacesize uint,

) ([]AlgorithmPerformance, error) {
	reqcount, err := r.getRNNForwardInferenceAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	seqLength := (C.int)(len(xD))
	tocxD := tensorDArrayToC(xD)
	tocyD := tensorDArrayToC(yD)

	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqcount)

	err = Status(C.cudnnFindRNNForwardTrainingAlgorithmEx(
		handle.x,
		r.descriptor,
		seqLength,
		&tocxD[0], x,
		hxD.descriptor, hx,
		cxD.descriptor, cx,
		wD.descriptor, w,
		&tocyD[0], y,
		hyD.descriptor, hy,
		cyD.descriptor, cy,
		C.float(findIntensity),
		C.int(reqcount),
		&actualcount,
		&perfresults[0],
		wspace, C.size_t(0),
		rspace, C.size_t(rspacesize),
	)).error("FindRNNForwardTrainingAlgorithmEx")

	return calgoperftogoarray(perfresults, handle.gogc), err
}

//GetRNNForwardInferenceAlgorithmMaxCount returns the maxcount and error
func (r *RNND) getRNNForwardInferenceAlgorithmMaxCount(
	handle *Handle,
) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNForwardInferenceAlgorithmMaxCount(
		handle.x,
		r.descriptor,
		&count,
	)).error("GetRNNForwardInferenceAlgorithmMaxCount")

	return int32(count), err
}
