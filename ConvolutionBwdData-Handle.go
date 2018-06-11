package gocudnn

/*

#include <cudnn.h>

void MakeAlgorithmforBWDData(cudnnAlgorithm_t *input,cudnnConvolutionBwdDataAlgo_t algo ){
	input->algo.convBwdDataAlgo=algo;
}

*/
import "C"
import (
	"fmt"
)

/* helper function to provide the convolution algo that fit best the requirement */

//Algo returns an Algorithm struct
func (cbd ConvBwdDataAlgo) Algo() Algorithm {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforBWDData(&algorithm, cbd.c())
	return Algorithm(algorithm)

}

//ConvBwdDataAlgoPerf is used to find the best/fastest algorithms
type ConvBwdDataAlgoPerf C.cudnnConvolutionBwdDataAlgoPerf_t

func (cbd ConvBwdDataAlgo) print() {
	switch cbd {
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0):
		fmt.Println("ConvBwdDataAlgo0")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1):
		fmt.Println("ConvBwdDataAlgo1")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT):
		fmt.Println("ConvBwdDataAlgoFFT")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING):
		fmt.Println("ConvBwdDataAlgoFFTTiling")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD):
		fmt.Println("ConvBwdDataAlgoWinograd")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED):
		fmt.Println("ConvBwdDataAlgoWinoGradNonFused")
	case ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT):
		fmt.Println("ConvBwdDataAlgoCount")

	default:
		fmt.Println("Not supported")
	}
}

//Print prints a human readable copy of the algorithm
func (cbd ConvBwdDataAlgoPerf) Print() {
	ConvBwdFiltAlgo(cbd.algo).print()
	fmt.Println("Status:", Status(cbd.algo).GetErrorString())
	fmt.Println("Time:", cbd.time)
	fmt.Println("Memory:", cbd.memory)
	fmt.Println("Determinism:", cbd.determinism)
	fmt.Println("MathType:", cbd.mathType)
}

//Algo returns the Pref Algo
func (cbd ConvBwdDataAlgoPerf) Algo() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(cbd.algo)
}

//Status returns the status which can be changed to error
func (cbd ConvBwdDataAlgoPerf) Status() Status {
	return Status(cbd.status)
}

//Time returns the time
func (cbd ConvBwdDataAlgoPerf) Time() float32 {
	return float32(cbd.time)
}

//Memory returns the memory
func (cbd ConvBwdDataAlgoPerf) Memory() SizeT {
	return SizeT(cbd.memory)
}

//Determinism returns the determinism
func (cbd ConvBwdDataAlgoPerf) Determinism() Determinism {
	return Determinism(cbd.determinism)
}

//Mathtype returns the mathtype
func (cbd ConvBwdDataAlgoPerf) Mathtype() MathType {
	return MathType(cbd.mathType)
}

//GetConvolutionBackwardDataAlgorithmMaxCount returns the max number of algos
func (handle *Handle) GetConvolutionBackwardDataAlgorithmMaxCount() (int32, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionBackwardDataAlgorithmMaxCount")
	return int32(count), x

}

//FindConvolutionBackwardDataAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (handle *Handle) FindConvolutionBackwardDataAlgorithm(
	w *FilterD,
	dy *TensorD,
	c *ConvolutionD,
	dx *TensorD,
	requestedAlgoCount int32) ([]ConvBwdDataAlgoPerf, error) {
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionBackwardDataAlgorithm(
		handle.x,
		w.descriptor,
		dy.descriptor,
		c.descriptor,
		dx.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0],
	)).error("FindConvolutionBackwardDataAlgorithm")
	results := make([]ConvBwdDataAlgoPerf, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = ConvBwdDataAlgoPerf(perfResults[i])

	}
	return results, err
}

//FindConvolutionBackwardDataAlgorithmEx finds some algorithms with memory
func (handle *Handle) FindConvolutionBackwardDataAlgorithmEx(
	wDesc *FilterD, wMem Memer,
	dyDesc *TensorD, dy Memer,
	conDesc *ConvolutionD,
	dxDesc *TensorD, dx Memer,
	reqAlgoCount int32, wspace Memer) ([]ConvBwdDataAlgoPerf, error) {
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionBackwardDataAlgorithmEx(
		handle.x,
		wDesc.descriptor, wMem.Ptr(),
		dyDesc.descriptor, dy.Ptr(),
		conDesc.descriptor,
		dxDesc.descriptor, dx.Ptr(),
		C.int(reqAlgoCount), &actualalgocount,
		&perfResults[0], wspace.Ptr(), wspace.ByteSize().c())).error("cudnnFindConvolutionBackwardDataAlgorithmEx")

	results := make([]ConvBwdDataAlgoPerf, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = ConvBwdDataAlgoPerf(perfResults[i])

	}
	return results, err
}

//GetConvolutionBackwardDataAlgorithm gives a good algo with the limits given to it
func (handle *Handle) GetConvolutionBackwardDataAlgorithm(
	wDesc *FilterD,
	dyDesc *TensorD,
	convDesc *ConvolutionD,
	dxDesc *TensorD,
	pref ConvBwdDataPref, wsmemlimit SizeT) (ConvBwdDataAlgo, error) {
	var algo C.cudnnConvolutionBwdDataAlgo_t
	err := Status(C.cudnnGetConvolutionBackwardDataAlgorithm(
		handle.x,
		wDesc.descriptor,
		dyDesc.descriptor,
		convDesc.descriptor,
		dxDesc.descriptor,
		pref.c(), wsmemlimit.c(), &algo)).error("GetConvolutionBackwardDataAlgorithm")
	return ConvBwdDataAlgo(algo), err
}

//GetConvolutionBackwardDataAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (handle *Handle) GetConvolutionBackwardDataAlgorithmV7(
	filt *FilterD,
	diff *TensorD,
	c *ConvolutionD,
	grad *TensorD,
	requestedAlgoCount int32) ([]ConvBwdDataAlgoPerf, error) {
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnGetConvolutionBackwardDataAlgorithm_v7(
		handle.x,
		filt.descriptor,
		diff.descriptor,
		c.descriptor,
		grad.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0])).error("GetConvolutionBackwardDataAlgorithmV7")
	results := make([]ConvBwdDataAlgoPerf, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = ConvBwdDataAlgoPerf(perfResults[i])

	}
	return results, err
}

//GetConvolutionBackwardDataWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (handle *Handle) GetConvolutionBackwardDataWorkspaceSize(
	w FilterD,
	dy TensorD,
	c ConvolutionD,
	dx TensorD,
	algo ConvBwdDataAlgo) (SizeT, error) {
	var sizebytes C.size_t
	err := Status(C.cudnnGetConvolutionBackwardDataWorkspaceSize(
		handle.x,
		w.descriptor,
		dy.descriptor,
		c.descriptor,
		dx.descriptor,
		algo.c(),
		&sizebytes)).error("GetConvolutionBackwardDataWorkspaceSize")
	return SizeT(sizebytes), err
}

//ConvolutionBackwardData does the backwards convolution on data
func (handle *Handle) ConvolutionBackwardData(
	alpha CScalar,
	wDesc *FilterD,
	w Memer,
	dyDesc *TensorD,
	dy Memer,
	convDesc *ConvolutionD,
	algo ConvBwdDataAlgo,
	wspace Memer,
	beta CScalar,
	dxDesc *TensorD,
	dx Memer,
) error {
	return Status(C.cudnnConvolutionBackwardData(
		handle.x,
		alpha.CPtr(),
		wDesc.descriptor,
		w.Ptr(),
		dyDesc.descriptor,
		dy.Ptr(),
		convDesc.descriptor,
		algo.c(),
		wspace.Ptr(),
		wspace.ByteSize().c(),
		beta.CPtr(),
		dxDesc.descriptor,
		dx.Ptr(),
	)).error("ConvolutionBackwardData")
}

//Im2Col transformes the multiDim tensors into 2d tensors for speed up in calculation at the cost of memory.
func (handle *Handle) Im2Col(
	xDesc *TensorD,
	x Memer,
	wDesc *FilterD,
	convDesc *ConvolutionD,
	buffer Memer,
) error {
	return Status(C.cudnnIm2Col(
		handle.x,
		xDesc.descriptor,
		x.Ptr(),
		wDesc.descriptor,
		convDesc.descriptor,
		buffer.Ptr(),
	)).error("Im2Col")
}
