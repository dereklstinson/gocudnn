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
	rnnD RNND,
	seqlength int32,
	xD TensorD,
	x Memer,
	hxD TensorD,
	hx Memer,
	cxD TensorD,
	cx Memer,
	wD FilterD,
	w Memer,
	yD TensorD,
	y Memer,
	hyD TensorD,
	hy Memer,
	cyD TensorD,
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
	if err != nil {
		return nil, err
	}
	results := make([]AlgorithmPerformance, C.int(retactAlgoCount))
	for i := 0; i < len(results); i++ {
		results[i] = AlgorithmPerformance(perfResults[i])
	}
	return results, nil
}
