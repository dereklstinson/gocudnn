package gocudnn

/*

#include <cudnn.h>

void MakeAlgorithmforFWD(cudnnAlgorithm_t *input,cudnnConvolutionFwdAlgo_t algo ){
	input->algo.convFwdAlgo=algo;
}
*/
import "C"
import "fmt"

//Algo returns an Algorithm Struct
func (a ConvolutionFwdAlgo) Algo() Algorithm {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforFWD(&algorithm, a.c())
	return Algorithm(algorithm)
}

func (a ConvolutionFwdAlgo) toString() string {
	var x string
	switch a {
	case ConvolutionFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM):
		x = "Implicit Gemm"
	case ConvolutionFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM):
		x = "Implicit Precomp Gemm"
	case ConvolutionFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM):
		x = "Gemm"
	case ConvolutionFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT):
		x = "Direct"
	case ConvolutionFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT):
		x = "FFT"
	case ConvolutionFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING):
		x = "FFT Tiling"
	case ConvolutionFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD):
		x = "WinoGrad"
	case ConvolutionFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED):
		x = "WinoGradNonFused"
	case ConvolutionFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT):
		x = "Count"
	default:
		x = "not supported algo --  to be honest ... I don't know how you got here"

	}
	return x
}

//PrintReadable prints this so that it is readable to a human
func (algoPerf ConvolutionFwdAlgoPerformance) PrintReadable(index int) {
	fmt.Println("")
	fmt.Println("")
	holder := make([]interface{}, 7)
	holder[0] = algoPerf.Algo.toString()
	holder[1] = algoPerf.Stat.GetErrorString()
	holder[2] = algoPerf.Time
	holder[3] = algoPerf.Memory
	holder[4] = algoPerf.Determinism.string()
	holder[5] = algoPerf.Mathtype.string()
	holder[6] = algoPerf.Reserved
	fmt.Println("Algo Perf", index)
	fmt.Println("---------------")
	for i := 0; i < len(holder); i++ {
		fmt.Println(holder[i])
	}
}

//ConvolutionFwdAlgoPerformance is a struct that holds the performance of the algorithm
type ConvolutionFwdAlgoPerformance struct {
	Algo        ConvolutionFwdAlgo
	Stat        Status
	Time        float32
	Memory      uint64
	Determinism Determinism
	Mathtype    MathType
	Reserved    [3]int32
}

func convertConvolutionFwdAlgoPerformance(input C.cudnnConvolutionFwdAlgoPerf_t) ConvolutionFwdAlgoPerformance {
	var x ConvolutionFwdAlgoPerformance
	x.Algo = ConvolutionFwdAlgo(input.algo)
	x.Stat = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint64(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.Mathtype = MathType(input.mathType)
	for i := 0; i < 3; i++ {
		x.Reserved[i] = int32(input.reserved[i])
	}
	return x
}

//GetConvolutionForwardAlgorithmMaxCount returns the max number of algos
func (handle *Handle) GetConvolutionForwardAlgorithmMaxCount() (int32, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionForwardAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionForwardAlgorithmMaxCount")
	return int32(count), x

}

//FindConvolutionForwardAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (handle *Handle) FindConvolutionForwardAlgorithm(
	x *TensorD,
	w *FilterD,
	c *ConvolutionD,
	y *TensorD,
	requestedAlgoCount int32) ([]ConvolutionFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionForwardAlgorithm(handle.x, x.descriptor, w.descriptor, c.descriptor, y.descriptor, C.int(requestedAlgoCount), &actualalgocount, &perfResults[0])).error("FindConvolutionForwardAlgorithm")
	results := make([]ConvolutionFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvolutionFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindConvolutionForwardAlgorithmEx finds some algorithms with memory
func (handle *Handle) FindConvolutionForwardAlgorithmEx(
	xDesc *TensorD,
	xMem Memer,
	wDesc *FilterD,
	wMem Memer,
	conDesc *ConvolutionD,
	yDesc *TensorD,
	yMem Memer,
	reqAlgoCount int32,
	wspace Memer) ([]ConvolutionFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionForwardAlgorithmEx(handle.x, xDesc.descriptor, xMem.Ptr(), wDesc.descriptor, wMem.Ptr(), conDesc.descriptor, yDesc.descriptor, yMem.Ptr(), C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), wspace.ByteSize().c())).error("FindConvolutionForwardAlgorithmEx")

	results := make([]ConvolutionFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvolutionFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionForwardAlgorithm gives a good algo with the limits given to it
func (handle *Handle) GetConvolutionForwardAlgorithm(
	xDesc *TensorD,
	wDesc *FilterD,
	convDesc *ConvolutionD,
	yDesc *TensorD,
	pref ConvolutionFwdPreference,
	wsmemlimit SizeT) (ConvolutionFwdAlgo, error) {
	var algo C.cudnnConvolutionFwdAlgo_t
	err := Status(C.cudnnGetConvolutionForwardAlgorithm(handle.x, xDesc.descriptor, wDesc.descriptor, convDesc.descriptor, yDesc.descriptor, C.cudnnConvolutionFwdPreference_t(pref), C.size_t(wsmemlimit), &algo)).error("GetConvolutionForwardAlgorithm")
	return ConvolutionFwdAlgo(algo), err
}

//GetConvolutionForwardAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (handle *Handle) GetConvolutionForwardAlgorithmV7(
	x *TensorD,
	w *FilterD,
	c *ConvolutionD,
	y *TensorD,
	requestedAlgoCount int32) ([]ConvolutionFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnGetConvolutionForwardAlgorithm_v7(handle.x, x.descriptor, w.descriptor, c.descriptor, y.descriptor, C.int(requestedAlgoCount), &actualalgocount, &perfResults[0])).error("FindConvolutionForwardAlgorithm")
	results := make([]ConvolutionFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvolutionFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionForwardWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (handle *Handle) GetConvolutionForwardWorkspaceSize(
	x *TensorD,
	w *FilterD,
	c *ConvolutionD,
	y *TensorD,
	algo ConvolutionFwdAlgo) (SizeT, error) {
	var sizebytes C.size_t
	err := Status(C.cudnnGetConvolutionForwardWorkspaceSize(handle.x, x.descriptor, w.descriptor, c.descriptor, y.descriptor, algo.c(), &sizebytes)).error("GetConvolutionForwardWorkspaceSize")
	return SizeT(sizebytes), err
}

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//ConvolutionForward Function to perform the forward pass for batch convolution
func (handle *Handle) ConvolutionForward(
	alpha CScalar,
	xD *TensorD,
	x Memer,
	wD *FilterD,
	w Memer,
	cD *ConvolutionD,
	algo ConvolutionFwdAlgo,
	wspace Memer,
	beta CScalar,
	yD *TensorD,
	y Memer) error {
	return Status(C.cudnnConvolutionForward(handle.x, alpha.CPtr(), xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(),
		cD.descriptor, algo.c(), wspace.Ptr(), wspace.ByteSize().c(), beta.CPtr(), yD.descriptor, y.Ptr())).error("ConvolutionForward")

}

//ConvolutionBiasActivationForward passes a lot of stuff so be carefull
/* Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
func (handle *Handle) ConvolutionBiasActivationForward(
	alpha1 CScalar,
	xD *TensorD,
	x Memer,
	wD *FilterD,
	w Memer,
	cD *ConvolutionD,
	algo ConvolutionFwdAlgo,
	wspace Memer,
	alpha2 CScalar,
	zD *TensorD,
	z Memer,
	biasD *TensorD,
	bias Memer,
	aD *ActivationD,
	yD *TensorD,
	y Memer,
) error {
	return Status(
		C.cudnnConvolutionBiasActivationForward(
			handle.x,
			alpha1.CPtr(),
			xD.descriptor,
			x.Ptr(),
			wD.descriptor,
			w.Ptr(),
			cD.descriptor,
			algo.c(),
			wspace.Ptr(),
			wspace.ByteSize().c(),
			alpha2.CPtr(),
			zD.descriptor,
			z.Ptr(),
			biasD.descriptor,
			bias.Ptr(),
			aD.descriptor,
			yD.descriptor,
			y.Ptr(),
		)).error("ConvolutionBiasActivationForward")
}
