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
	results := make([]AlgorithmPerformance, actualcount)
	for i := 0; i < len(results); i++ {
		results[i] = AlgorithmPerformance(perfresults[i])
	}
	return results, err
}

//RNNForwardInference is the forward inference
func (handle *Handle) RNNForwardInference(
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

/*
cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining( cudnnHandle_t              handle,
	const cudnnRNNDescriptor_t    rnnDesc,
	const int                     seqLength,
	const cudnnTensorDescriptor_t *xDesc,
	const void                    *x,
	const cudnnTensorDescriptor_t hxDesc,
	const void                    *hx,
	const cudnnTensorDescriptor_t cxDesc,
	const void                    *cx,
	const cudnnFilterDescriptor_t wDesc,
	const void                    *w,
	const cudnnTensorDescriptor_t *yDesc,
	void                          *y,
	const cudnnTensorDescriptor_t hyDesc,
	void                          *hy,
	const cudnnTensorDescriptor_t cyDesc,
	void                          *cy,
	void                          *workspace,
	size_t                        workSpaceSizeInBytes,
	void *                        reserveSpace,
	size_t                        reserveSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData( cudnnHandle_t                 handle,
	const cudnnRNNDescriptor_t    rnnDesc,
	const int                     seqLength,
	const cudnnTensorDescriptor_t *yDesc,
	const void                    *y,
	const cudnnTensorDescriptor_t *dyDesc,
	const void                    *dy,
	const cudnnTensorDescriptor_t dhyDesc,
	const void                    *dhy,
	const cudnnTensorDescriptor_t dcyDesc,
	const void                    *dcy,
	const cudnnFilterDescriptor_t wDesc,
	const void                    *w,
	const cudnnTensorDescriptor_t hxDesc,
	const void                    *hx,
	const cudnnTensorDescriptor_t cxDesc,
	const void                    *cx,
	const cudnnTensorDescriptor_t *dxDesc,
	void                          *dx,
	const cudnnTensorDescriptor_t dhxDesc,
	void                          *dhx,
	const cudnnTensorDescriptor_t dcxDesc,
	void                          *dcx,
	void                          *workspace,
	size_t                        workSpaceSizeInBytes,
	void *                        reserveSpace,
	size_t                        reserveSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights( cudnnHandle_t              handle,
	const cudnnRNNDescriptor_t    rnnDesc,
	const int                     seqLength,
	const cudnnTensorDescriptor_t *xDesc,
	const void                    *x,
	const cudnnTensorDescriptor_t hxDesc,
	const void                    *hx,
	const cudnnTensorDescriptor_t *yDesc,
	const void                    *y,
	const void                    *workspace,
	size_t                        workSpaceSizeInBytes,
	const cudnnFilterDescriptor_t dwDesc,
	void                          *dw,
	const void                    *reserveSpace,
	size_t                        reserveSpaceSizeInBytes);
*/
