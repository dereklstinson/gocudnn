package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//GetRNNBackwardDataAlgorithmMaxCount gets the max number of algorithms for the back prop rnn
func (r *RNND) getRNNBackwardDataAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	err := Status(C.cudnnGetRNNBackwardDataAlgorithmMaxCount(
		handle.x,
		r.descriptor,
		&count,
	)).error("GetRNNBackwardDataAlgorithmMaxCount")

	return int32(count), err
}

//FindRNNBackwardDataAlgorithmEx finds a list of Algorithm for backprop this passes like 26 parameters and pointers and stuff so watch out.
func (r *RNND) FindRNNBackwardDataAlgorithmEx(
	handle *Handle,
	yD []*TensorD, y gocu.Mem,
	dyD []*TensorD, dy gocu.Mem,
	dhyD *TensorD, dhy gocu.Mem,
	dcyD *TensorD, dcy gocu.Mem,
	wD *FilterD, w gocu.Mem,
	hxD *TensorD, hx gocu.Mem,
	cxD *TensorD, cx gocu.Mem,
	dxD []*TensorD, dx gocu.Mem,
	dhxD *TensorD, dhx gocu.Mem,
	dcxD *TensorD, dcx gocu.Mem,
	findIntensity float32,
	wspace gocu.Mem, wspacesize uint,
	rspace gocu.Mem, rspacesize uint,

) ([]AlgorithmPerformance, error) {
	reqAlgocount, err := r.getRNNBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	seqLen := C.int(len(yD))
	cyD := tensorDArrayToC(yD)
	cdyD := tensorDArrayToC(dyD)
	cdxD := tensorDArrayToC(dxD)
	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)
	if wspace == nil {
		err := Status(C.cudnnFindRNNBackwardDataAlgorithmEx(
			handle.x,
			r.descriptor,
			seqLen,
			&cyD[0], y.Ptr(),
			&cdyD[0], dy.Ptr(),
			dhyD.descriptor, dhy.Ptr(),
			dcyD.descriptor, dcy.Ptr(),
			wD.descriptor, w.Ptr(),
			hxD.descriptor, hx.Ptr(),
			cxD.descriptor, cx.Ptr(),
			&cdxD[0], dx.Ptr(),
			dhxD.descriptor, dhx.Ptr(),
			dcxD.descriptor, dcx.Ptr(),
			C.float(findIntensity),
			C.int(reqAlgocount),
			&actualcount,
			&perfresults[0],
			nil,
			C.size_t(0),
			rspace.Ptr(), C.size_t(rspacesize), //31 total?
		)).error("FindRNNBackwardDataAlgorithmEx")
		return calgoperftogoarray(perfresults, handle.gogc), err
	}
	err = Status(C.cudnnFindRNNBackwardDataAlgorithmEx(
		handle.x,
		r.descriptor,
		seqLen,
		&cyD[0], y.Ptr(),
		&cdyD[0], dy.Ptr(),
		dhyD.descriptor, dhy.Ptr(),
		dcyD.descriptor, dcy.Ptr(),
		wD.descriptor, w.Ptr(),
		hxD.descriptor, hx.Ptr(),
		cxD.descriptor, cx.Ptr(),
		&cdxD[0], dx.Ptr(),
		dhxD.descriptor, dhx.Ptr(),
		dcxD.descriptor, dcx.Ptr(),
		C.float(findIntensity),
		C.int(reqAlgocount),
		&actualcount,
		&perfresults[0],
		wspace.Ptr(), C.size_t(wspacesize),
		rspace.Ptr(), C.size_t(rspacesize),
	)).error("FindRNNBackwardDataAlgorithmEx")
	return calgoperftogoarray(perfresults, handle.gogc), err
}

//FindRNNBackwardDataAlgorithmExUS is like FindRNNBackwardDataAlgorithmEx but uses unsafe.Pointer instead of gocu.Mem
func (r *RNND) FindRNNBackwardDataAlgorithmExUS(
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
	findIntensity float32,
	wspace unsafe.Pointer, wspacesize uint,
	rspace unsafe.Pointer, rspacesize uint,

) ([]AlgorithmPerformance, error) {
	reqAlgocount, err := r.getRNNBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	seqLen := C.int(len(yD))
	cyD := tensorDArrayToC(yD)
	cdyD := tensorDArrayToC(dyD)
	cdxD := tensorDArrayToC(dxD)
	var actualcount C.int
	perfresults := make([]C.cudnnAlgorithmPerformance_t, reqAlgocount)

	err = Status(C.cudnnFindRNNBackwardDataAlgorithmEx(
		handle.x,
		r.descriptor,
		seqLen,
		&cyD[0], y,
		&cdyD[0], dy,
		dhyD.descriptor, dhy,
		dcyD.descriptor, dcy,
		wD.descriptor, w,
		hxD.descriptor, hx,
		cxD.descriptor, cx,
		&cdxD[0], dx,
		dhxD.descriptor, dhx,
		dcxD.descriptor, dcx,
		C.float(findIntensity),
		C.int(reqAlgocount),
		&actualcount,
		&perfresults[0],
		wspace, C.size_t(wspacesize),
		rspace, C.size_t(rspacesize),
		//31 total?
	)).error("FindRNNBackwardDataAlgorithmExUS")
	return calgoperftogoarray(perfresults, handle.gogc), err
}
