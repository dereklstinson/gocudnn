package gocudnn

/*

#include <cudnn.h>

void MakeAlgorithmforBWDFilter(cudnnAlgorithm_t *input,cudnnConvolutionBwdFilterAlgo_t algo ){
	input->algo.convBwdFilterAlgo=algo;
}
*/
import "C"
import "fmt"

//ConvolutionBackwardBias Function to compute the bias gradient for batch convolution db is returned
func (handle *Handle) ConvolutionBackwardBias(alpha CScalar, dyD TensorD, dy Memer, beta CScalar, dbD TensorD, db Memer) error {
	return Status(C.cudnnConvolutionBackwardBias(handle.x, alpha.CPtr(), dyD.descriptor, dy.Ptr(), beta.CPtr(), dbD.descriptor, db.Ptr())).error("ConvolutionBackwardBias")
}

//ConvBwdFilterPref are used for flags for the backwds filters
type ConvBwdFilterPref C.cudnnConvolutionBwdFilterPreference_t

//ConvBwdFilterPrefFlag is used to pass ConvBwdFilterPref flags through methods
type ConvBwdFilterPrefFlag struct {
}

//NoWorkspace return ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
func (c ConvBwdFilterPrefFlag) NoWorkspace() ConvBwdFilterPref {
	return ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
}

//PrefFastest return ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)
func (c ConvBwdFilterPrefFlag) PrefFastest() ConvBwdFilterPref {
	return ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)
}

//SpecifyWorkspaceLimit return ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)
func (c ConvBwdFilterPrefFlag) SpecifyWorkspaceLimit() ConvBwdFilterPref {
	return ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)
}

/*
//These are the backwards filter flags
const (
	ConvBwdFilterNoWorkspace           ConvBwdFilterPref = C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE
	ConvBwdFilterPrefFastest           ConvBwdFilterPref = C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST
	ConvBwdFilterSpecifyWorkspaceLimit ConvBwdFilterPref = C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
)
*/

func (bw ConvBwdFilterPref) c() C.cudnnConvolutionBwdFilterPreference_t {
	return C.cudnnConvolutionBwdFilterPreference_t(bw)
}

//ConvBwdFiltAlgo Used for ConvBwdFiltAlgo flags
type ConvBwdFiltAlgo C.cudnnConvolutionBwdFilterAlgo_t

//ConvBwdFiltAlgoFlag is used to pass ConvBwdFiltAlgo Flags
type ConvBwdFiltAlgoFlag struct {
}

//Algo0 return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0) /* non-deterministic */
func (c ConvBwdFiltAlgoFlag) Algo0() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)
}

//Algo1 return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
func (c ConvBwdFiltAlgoFlag) Algo1() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
}

//FFT return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT)
func (c ConvBwdFiltAlgoFlag) FFT() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT)
}

//Algo3 return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)
func (c ConvBwdFiltAlgoFlag) Algo3() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)
}

//Winograd 	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD)
func (c ConvBwdFiltAlgoFlag) Winograd() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD)
}

//WinogradNonFused return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED)
func (c ConvBwdFiltAlgoFlag) WinogradNonFused() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED)
}

//FFTTiling return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING)
func (c ConvBwdFiltAlgoFlag) FFTTiling() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING)
}

//Count return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)
func (c ConvBwdFiltAlgoFlag) Count() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)
}
func (cb ConvBwdFiltAlgo) c() C.cudnnConvolutionBwdFilterAlgo_t {
	return C.cudnnConvolutionBwdFilterAlgo_t(cb)
}

//Algo returns an Algorithm Struct
func (cb ConvBwdFiltAlgo) Algo() Algorithm {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforBWDFilter(&algorithm, cb.c())
	return Algorithm(algorithm)
}

func (cb ConvBwdFiltAlgo) print() {
	switch cb {
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0):
		fmt.Println("ConvBwdFiltAlgo0")
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1):
		fmt.Println("ConvBwdFiltAlgo1")
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT):
		fmt.Println("ConvBwdFiltAlgoFFT")
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3):
		fmt.Println("ConvBwdFiltAlgo3")
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD):
		fmt.Println("ConvBwdFiltAlgoWinGrad")
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED):
		fmt.Println("ConvBwdFiltAlgoNonFused")
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING):
		fmt.Println("ConvBwdFiltAlgoFFTTiling")
	case ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT):
		fmt.Println("ConvBwdFiltAlgoCount")
	default:
		fmt.Println("Not supported")
	}
}

//ConvBwdFiltAlgoPerf is the return struct in the finding algorithm funcs
type ConvBwdFiltAlgoPerf C.cudnnConvolutionBwdFilterAlgoPerf_t

//Print prints a human readable copy of the algorithm
func (cb ConvBwdFiltAlgoPerf) Print() {
	ConvBwdFiltAlgo(cb.algo).print()
	fmt.Println("Status:", Status(cb.algo).GetErrorString())
	fmt.Println("Time:", cb.time)
	fmt.Println("Memory:", cb.memory)
	fmt.Println("Determinism:", cb.determinism)
	fmt.Println("MathType:", cb.mathType)
}

//PrefAlgo returns the Pref Algo
func (cb ConvBwdFiltAlgoPerf) PrefAlgo() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(cb.algo)
}

//Status returns the status which can be changed to error
func (cb ConvBwdFiltAlgoPerf) Status() Status {
	return Status(cb.status)
}

//Time returns the time
func (cb ConvBwdFiltAlgoPerf) Time() float32 {
	return float32(cb.time)
}

//Memory returns the memory
func (cb ConvBwdFiltAlgoPerf) Memory() SizeT {
	return SizeT(cb.memory)
}

//Determinism returns the determinism
func (cb ConvBwdFiltAlgoPerf) Determinism() Determinism {
	return Determinism(cb.determinism)
}

//Mathtype returns the mathtype
func (cb ConvBwdFiltAlgoPerf) Mathtype() MathType {
	return MathType(cb.mathType)
}

//GetConvolutionBackwardAlgorithmMaxCount returns the max number of algos
func (handle *Handle) GetConvolutionBackwardAlgorithmMaxCount() (int32, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionForwardAlgorithmMaxCount")
	return int32(count), x

}

//FindConvolutionBackwardFilterAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (handle *Handle) FindConvolutionBackwardFilterAlgorithm(
	x *TensorD,
	dy *TensorD,
	c *ConvolutionD,
	dw *FilterD,
	requestedAlgoCount int32,
) ([]ConvBwdFiltAlgoPerf, error) {
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionBackwardFilterAlgorithm(
		handle.x,
		x.descriptor,
		dy.descriptor,
		c.descriptor,
		dw.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0],
	)).error("FindConvolutionBackwardFilterAlgorithm")
	results := make([]ConvBwdFiltAlgoPerf, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = ConvBwdFiltAlgoPerf(perfResults[i])

	}
	return results, err
}

//FindConvolutionBackwardFilterAlgorithmEx finds some algorithms with memory
func (handle *Handle) FindConvolutionBackwardFilterAlgorithmEx(
	xDesc *TensorD, xMem Memer,
	dyDesc *TensorD, y Memer,
	conDesc *ConvolutionD,
	dwDesc *FilterD, dw Memer,
	reqAlgoCount int32, wspace Memer) ([]ConvBwdFiltAlgoPerf, error) {
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionBackwardFilterAlgorithmEx(
		handle.x,
		xDesc.descriptor, xMem.Ptr(),
		dyDesc.descriptor, y.Ptr(),
		conDesc.descriptor,
		dwDesc.descriptor, dw.Ptr(),
		C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), wspace.ByteSize().c())).error("FindConvolutionBackwardFilterAlgorithmEx")

	results := make([]ConvBwdFiltAlgoPerf, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = ConvBwdFiltAlgoPerf(perfResults[i])

	}
	return results, err
}

//GetConvolutionBackwardFilterAlgorithm gives a good algo with the limits given to it
func (handle *Handle) GetConvolutionBackwardFilterAlgorithm(
	xDesc *TensorD,
	dyDesc *TensorD,
	convDesc *ConvolutionD,
	dwDesc *FilterD,
	pref ConvBwdFilterPref, wsmemlimit SizeT) (ConvBwdFiltAlgo, error) {
	var algo C.cudnnConvolutionBwdFilterAlgo_t
	err := Status(C.cudnnGetConvolutionBackwardFilterAlgorithm(
		handle.x,
		xDesc.descriptor,
		dyDesc.descriptor,
		convDesc.descriptor,
		dwDesc.descriptor,
		pref.c(), wsmemlimit.c(), &algo)).error("GetConvolutionBackwardFilterAlgorithm")
	return ConvBwdFiltAlgo(algo), err
}

//GetConvolutionBackwardFilterAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (handle *Handle) GetConvolutionBackwardFilterAlgorithmV7(
	src *TensorD,
	diff *TensorD,
	c *ConvolutionD,
	grad *FilterD,
	requestedAlgoCount int32) ([]ConvBwdFiltAlgoPerf, error) {
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnGetConvolutionBackwardFilterAlgorithm_v7(
		handle.x,
		src.descriptor,
		diff.descriptor,
		c.descriptor,
		grad.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0])).error("GetConvolutionBackwardFilterAlgorithm_v7")
	results := make([]ConvBwdFiltAlgoPerf, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = ConvBwdFiltAlgoPerf(perfResults[i])

	}
	return results, err
}

//GetConvolutionBackwardFilterWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (handle *Handle) GetConvolutionBackwardFilterWorkspaceSize(
	x *TensorD,
	dy *TensorD,
	c *ConvolutionD,
	grad *FilterD,
	algo ConvBwdFiltAlgo) (SizeT, error) {
	var sizebytes C.size_t
	err := Status(C.cudnnGetConvolutionBackwardFilterWorkspaceSize(
		handle.x,
		x.descriptor,
		dy.descriptor,
		c.descriptor,
		grad.descriptor,
		algo.c(),
		&sizebytes)).error("GetConvolutionForwardWorkspaceSize")
	return SizeT(sizebytes), err
}

//ConvolutionBackwardFilter does the backwards convolution
func (handle *Handle) ConvolutionBackwardFilter(
	alpha CScalar,
	xDesc *TensorD,
	x Memer,
	dyDesc *TensorD,
	dy Memer,
	convDesc *ConvolutionD,
	algo ConvBwdFiltAlgo,
	wspace Memer,
	beta CScalar,
	dwDesc *FilterD,
	dw Memer,
) error {
	return Status(C.cudnnConvolutionBackwardFilter(
		handle.x,
		alpha.CPtr(),
		xDesc.descriptor,
		x.Ptr(),
		dyDesc.descriptor,
		dy.Ptr(),
		convDesc.descriptor,
		algo.c(),
		wspace.Ptr(),
		wspace.ByteSize().c(),
		beta.CPtr(),
		dwDesc.descriptor,
		dw.Ptr(),
	)).error("cudnnConvolutionBackwardFilter")
}
