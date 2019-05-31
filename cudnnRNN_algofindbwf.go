package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//GetRNNBackwardWeightsAlgorithmMaxCount gets the max number of Algorithm for weights
func (r *RNND) getRNNBackwardWeightsAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
		handle.x,
		r.descriptor,
		&count,
	)).error("GetRNNBackwardWeightsAlgorithmMaxCount")

	return int32(count), err
}

//FindRNNBackwardWeightsAlgorithmEx returns some Algorithm and their performance and stuff
func (r *RNND) FindRNNBackwardWeightsAlgorithmEx(
	handle *Handle,
	xD []*TensorD, x cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	yD []*TensorD, y cutil.Mem,
	findIntensity float32, //unused for future use
	wspace cutil.Mem, wspacesize uint,
	dwD *FilterD, dw cutil.Mem,
	rspace cutil.Mem, rspacesize uint,

) ([]AlgorithmPerformance, error) {
	reqAlgocount, err := r.getRNNBackwardWeightsAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	seqLength := (C.int)(len(xD))
	var actualcount C.int
	inCxD := tensorDArrayToC(xD)
	inCyD := tensorDArrayToC(yD)
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	if wspace == nil {
		err = Status(C.cudnnFindRNNBackwardWeightsAlgorithmEx(
			handle.x,
			r.descriptor,
			seqLength,
			&inCxD[0], x.Ptr(),
			hxD.descriptor, hx.Ptr(),
			&inCyD[0], y.Ptr(),

			C.float(findIntensity),
			C.int(reqAlgocount),
			&actualcount,
			&perfresults[0],
			nil, C.size_t(0),
			dwD.descriptor, dw.Ptr(),
			rspace.Ptr(), C.size_t(rspacesize),
		)).error("FindRNNBackwardWeightsAlgorithmEx")

		return calgoperftogoarray(perfresults, handle.gogc), err
	}
	err = Status(C.cudnnFindRNNBackwardWeightsAlgorithmEx(
		handle.x,
		r.descriptor,
		seqLength,
		&inCxD[0], x.Ptr(),
		hxD.descriptor, hx.Ptr(),
		&inCyD[0], y.Ptr(),
		C.float(findIntensity),
		C.int(reqAlgocount),
		&actualcount,
		&perfresults[0],
		wspace.Ptr(), C.size_t(wspacesize),
		dwD.descriptor, dw.Ptr(),
		rspace.Ptr(), C.size_t(rspacesize),
	)).error("FindRNNBackwardWeightsAlgorithmEx")

	return calgoperftogoarray(perfresults, handle.gogc), err
}

//FindRNNBackwardWeightsAlgorithmExUS is like FindRNNBackwardWeightsAlgorithmEx but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) FindRNNBackwardWeightsAlgorithmExUS(
	handle *Handle,
	xD []*TensorD, x unsafe.Pointer,
	hxD *TensorD, hx unsafe.Pointer,
	yD []*TensorD, y unsafe.Pointer,
	findIntensity float32, //unused for future use
	wspace unsafe.Pointer, wspacesize uint,
	dwD *FilterD, dw unsafe.Pointer,
	rspace unsafe.Pointer, rspacesize uint,

) ([]AlgorithmPerformance, error) {
	reqAlgocount, err := r.getRNNBackwardWeightsAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	seqLength := (C.int)(len(xD))
	var actualcount C.int
	inCxD := tensorDArrayToC(xD)
	inCyD := tensorDArrayToC(yD)
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)

	err = Status(C.cudnnFindRNNBackwardWeightsAlgorithmEx(
		handle.x,
		r.descriptor,
		seqLength,
		&inCxD[0], x,
		hxD.descriptor, hx,
		&inCyD[0], y,
		C.float(findIntensity),
		C.int(reqAlgocount),
		&actualcount,
		&perfresults[0],
		wspace, C.size_t(wspacesize),
		dwD.descriptor, dw,
		rspace, C.size_t(rspacesize),
	)).error("FindRNNBackwardWeightsAlgorithmEx")

	return calgoperftogoarray(perfresults, handle.gogc), err
}
