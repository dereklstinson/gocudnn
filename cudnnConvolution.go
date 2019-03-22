package gocudnn

/*
#include <cudnn.h>

void MakeAlgorithmforBWDData(cudnnAlgorithm_t *input,cudnnConvolutionBwdDataAlgo_t algo ){
	input->algo.convBwdDataAlgo=algo;
}
void MakeAlgorithmforBWDFilter(cudnnAlgorithm_t *input,cudnnConvolutionBwdFilterAlgo_t algo ){
	input->algo.convBwdFilterAlgo=algo;
}
void MakeAlgorithmforFWD(cudnnAlgorithm_t *input,cudnnConvolutionFwdAlgo_t algo ){
	input->algo.convFwdAlgo=algo;
}
*/
import "C"
import (
	"fmt"
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

/*


Descriptors


*/

//ConvolutionD sets all the convolution info
type ConvolutionD struct {
	descriptor C.cudnnConvolutionDescriptor_t
	dims       C.int
	isconv2d   bool
	gogc       bool
}

const convolutionnd2dtestflag = true

//CreateConvolutionDescriptor creates a convolution descriptor
func CreateConvolutionDescriptor() (*ConvolutionD, error) {
	d := new(ConvolutionD)
	err := Status(C.cudnnCreateConvolutionDescriptor(&d.descriptor)).error("NewConvolution2dDescriptor-create")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		runtime.SetFinalizer(d, destroyconvolutiondescriptor)
	}
	return d, nil
}

//Set sets the convolution descriptor
//Input.Type of the filter layout format. If this input is set to CUDNN_TENSOR_NCHW, which is one of the enumerated values allowed by cudnnTensorFormat_t descriptor, then the layout of the filter is as follows:
//
//For N=4, i.e., for a 4D filter descriptor, the filter layout is in the form of KCRS (K represents the number of output feature maps, C the number of input feature maps, R the number of rows per filter, and S the number of columns per filter.)
//For N=3, i.e., for a 3D filter descriptor, the number S (number of columns per filter) is omitted.
//For N=5 and greater, the layout of the higher dimensions immediately follow RS.
//On the other hand, if this input is set to CUDNN_TENSOR_NHWC, then the layout of the filter is as follows:
//for N=4, i.e., for a 4D filter descriptor, the filter layout is in the form of KRSC.
//For N=3, i.e., for a 3D filter descriptor, the number S (number of columns per filter) is omitted, and the layout of C immediately follows R.
//For N=5 and greater, the layout of the higher dimensions are inserted between S and C. See also the description for cudnnTensorFormat_t.
func (c *ConvolutionD) Set(mode ConvolutionMode, data DataType, pad, stride, dialation []int32) error {
	cdata := data.c()
	cmode := mode.c()
	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdialation := int32Tocint(dialation)
	dims := C.int(len(pad))
	return Status(C.cudnnSetConvolutionNdDescriptor(c.descriptor, dims, &cpad[0], &cstride[0], &cdialation[0], cmode, cdata)).error("NewConvolutionNdDescriptor-set")

}

//Get gets returns the values used to make the convolution descriptor
func (c *ConvolutionD) Get() (ConvolutionMode, DataType, []int32, []int32, []int32, error) {
	if !convolutionnd2dtestflag {
		var padh C.int
		var padw C.int
		var u C.int
		var v C.int
		var dh C.int
		var dw C.int
		var mode C.cudnnConvolutionMode_t
		var dtype C.cudnnDataType_t

		err := Status(C.cudnnGetConvolution2dDescriptor(c.descriptor, &padh, &padw, &u, &v, &dh, &dw, &mode, &dtype)).error("Get2dDescripter")
		pads := make([]int32, 2)
		uv := make([]int32, 2)
		dilat := make([]int32, 2)
		pads[0] = int32(padh)
		pads[1] = int32(padw)
		uv[0] = int32(u)
		uv[1] = int32(v)
		dilat[0] = int32(dh)
		dilat[1] = int32(dw)

		return ConvolutionMode(mode), DataType(dtype), pads, uv, dilat, err

	}

	pad := make([]C.int, c.dims)
	stride := make([]C.int, c.dims)
	dilation := make([]C.int, c.dims)
	var actual C.int
	var mode C.cudnnConvolutionMode_t
	var dtype C.cudnnDataType_t
	err := Status(C.cudnnGetConvolutionNdDescriptor(c.descriptor, c.dims, &actual, &pad[0], &stride[0], &dilation[0], &mode, &dtype)).error("GetndDescriptor")

	return ConvolutionMode(mode), DataType(dtype), cintToint32(pad), cintToint32(stride), cintToint32(dilation), err

}

//SetGroupCount sets the Group Count
func (c *ConvolutionD) SetGroupCount(groupCount int32) error {

	err := Status(C.cudnnSetConvolutionGroupCount(c.descriptor, C.int(groupCount))).error("SetGroupCountandMathtype-Group")

	return err

}

//SetMathType sets the mathtype
func (c *ConvolutionD) SetMathType(mathtype MathType) error {

	x := Status(C.cudnnSetConvolutionMathType(c.descriptor, C.cudnnMathType_t(mathtype)))

	return x.error("SetGroupCountandMathtype-Math")
}

//GetOutputDims is a helper function to give the size of the output of of a COnvolutionNDForward
//Each dimension of the (nbDims-2)-D images of the output tensor is computed as followed:
//
//    outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride;
//
func (c *ConvolutionD) GetOutputDims(input *TensorD, filter *FilterD) ([]int32, error) {
	dims := make([]C.int, int32(input.dims))
	x := Status(C.cudnnGetConvolutionNdForwardOutputDim(c.descriptor, input.descriptor, filter.descriptor, input.dims, &dims[0])).error("GetConvolutionNdForwardOutputDim")

	return cintToint32(dims), x
}

//Destroy destroys the ConvolutionDescriptor. If GC is set then it only returns nil.
//Currently GC is set with no option to turn off
func (c *ConvolutionD) Destroy() error {
	if setfinalizer || c.gogc {
		return nil
	}
	return destroyconvolutiondescriptor(c)
}
func destroyconvolutiondescriptor(c *ConvolutionD) error {
	return Status(C.cudnnDestroyConvolutionDescriptor(c.descriptor)).error("DestroyConvolutionDescriptor")
}

/* helper function to provide the convolution algo that fit best the requirement */

//Algo returns an Algorithm struct
func (c ConvBwdDataAlgo) Algo() Algorithm {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforBWDData(&algorithm, c.c())
	return Algorithm(algorithm)

}

//ConvBwdDataAlgoPerf is used to find the best/fastest algorithms
//type ConvBwdDataAlgoPerformance C.cudnnConvolutionBwdDataAlgoPerf_t

//ConvBwdDataAlgoPerformance is the return struct in the finding algorithm funcs
type ConvBwdDataAlgoPerformance struct {
	Algo        ConvBwdDataAlgo `json:"algo,omitempty"`
	Status      Status          `json:"status,omitempty"`
	Time        float32         `json:"time,omitempty"`
	Memory      uint            `json:"memory,omitempty"`
	Determinism Determinism     `json:"determinism,omitempty"`
	MathType    MathType        `json:"math_type,omitempty"`
}

func convertConvBwdDataAlgoPerformance(input C.cudnnConvolutionBwdDataAlgoPerf_t) ConvBwdDataAlgoPerformance {
	var x ConvBwdDataAlgoPerformance
	x.Algo = ConvBwdDataAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)
	return x
}

func (c ConvBwdDataAlgo) print() {
	switch c {
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
func (cbd ConvBwdDataAlgoPerformance) Print() {
	ConvBwdFiltAlgo(cbd.Algo).print()
	fmt.Println("Status:", Status(cbd.Algo).GetErrorString())
	fmt.Println("Time:", cbd.Time)
	fmt.Println("Memory:", cbd.Memory)
	fmt.Println("Determinism:", cbd.Determinism)
	fmt.Println("MathType:", cbd.MathType)
}

//GetConvolutionBackwardDataAlgorithmMaxCount returns the max number of Algorithm
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardDataAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionBackwardDataAlgorithmMaxCount")

	return int32(count), x

}

//FindConvolutionBackwardDataAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (cbf ConvolutionBwdFuncs) FindConvolutionBackwardDataAlgorithm(
	handle *Handle,
	w *FilterD,
	dy *TensorD,
	c *ConvolutionD,
	dx *TensorD,
	requestedAlgoCount int32) ([]ConvBwdDataAlgoPerformance, error) {
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

	results := make([]ConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdDataAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindConvolutionBackwardDataAlgorithmEx finds some algorithms with memory
func (cbf ConvolutionBwdFuncs) FindConvolutionBackwardDataAlgorithmEx(
	handle *Handle,
	wD *FilterD, w gocu.Mem,
	dyD *TensorD, dy gocu.Mem,
	cD *ConvolutionD,
	dxD *TensorD, dx gocu.Mem,
	reqAlgoCount int32, wspace gocu.Mem, wspacesize uint) ([]ConvBwdDataAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionBackwardDataAlgorithmEx(
		handle.x,
		wD.descriptor, w.Ptr(),
		dyD.descriptor, dy.Ptr(),
		cD.descriptor,
		dxD.descriptor, dx.Ptr(),
		C.int(reqAlgoCount), &actualalgocount,
		&perfResults[0], wspace.Ptr(), C.size_t(wspacesize))).error("cudnnFindConvolutionBackwardDataAlgorithmEx")

	results := make([]ConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdDataAlgoPerformance(perfResults[i])

	}

	return results, err
}

//GetConvolutionBackwardDataAlgorithm gives a good algo with the limits given to it
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardDataAlgorithm(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	cD *ConvolutionD,
	dxD *TensorD,
	pref ConvBwdDataPref, wsmemlimit uint) (ConvBwdDataAlgo, error) {
	var algo C.cudnnConvolutionBwdDataAlgo_t
	err := Status(C.cudnnGetConvolutionBackwardDataAlgorithm(
		handle.x,
		wD.descriptor,
		dyD.descriptor,
		cD.descriptor,
		dxD.descriptor,
		pref.c(), (C.size_t)(wsmemlimit), &algo)).error("GetConvolutionBackwardDataAlgorithm")

	return ConvBwdDataAlgo(algo), err
}

//GetConvolutionBackwardDataAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardDataAlgorithmV7(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	cD *ConvolutionD,
	dxD *TensorD,
	requestedAlgoCount int32) ([]ConvBwdDataAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionBwdDataAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnGetConvolutionBackwardDataAlgorithm_v7(
		handle.x,
		wD.descriptor,
		dyD.descriptor,
		cD.descriptor,
		dxD.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0])).error("GetConvolutionBackwardDataAlgorithmV7")
	results := make([]ConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdDataAlgoPerformance(perfResults[i])

	}

	return results, err
}

//GetConvolutionBackwardDataWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardDataWorkspaceSize(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	cD *ConvolutionD,
	dxD *TensorD,
	algo ConvBwdDataAlgo) (uint, error) {
	var sizebytes C.size_t
	err := Status(C.cudnnGetConvolutionBackwardDataWorkspaceSize(
		handle.x,
		wD.descriptor,
		dyD.descriptor,
		cD.descriptor,
		dxD.descriptor,
		algo.c(),
		&sizebytes)).error("GetConvolutionBackwardDataWorkspaceSize")

	return uint(sizebytes), err
}

//BackwardData does the backwards convolution on data
func (c *ConvolutionD) BackwardData(
	handle *Handle,
	alpha float64,
	wD *FilterD,
	w gocu.Mem,
	dyD *TensorD,
	dy gocu.Mem,
	algo ConvBwdDataAlgo,
	wspace gocu.Mem,
	wspacesize uint,
	beta float64,
	dxD *TensorD,
	dx gocu.Mem,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	if wspace == nil {

		return Status(C.cudnnConvolutionBackwardData(
			handle.x,
			a.CPtr(),
			wD.descriptor,
			w.Ptr(),
			dyD.descriptor,
			dy.Ptr(),
			c.descriptor,
			algo.c(),
			nil,
			(C.size_t)(wspacesize),
			b.CPtr(),
			dxD.descriptor,
			dx.Ptr(),
		)).error("ConvolutionBackwardData")
	}

	return Status(C.cudnnConvolutionBackwardData(
		handle.x,
		a.CPtr(),
		wD.descriptor,
		w.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		c.descriptor,
		algo.c(),
		wspace.Ptr(),
		(C.size_t)(wspacesize),
		b.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
	)).error("ConvolutionBackwardData")
}

//Im2Col transformes the multiDim tensors into 2d tensors for speed up in calculation at the cost of memory.
func (cbf ConvolutionBwdFuncs) Im2Col(
	handle *Handle,
	xD *TensorD,
	x gocu.Mem,
	wD *FilterD,
	cD *ConvolutionD,
	buffer gocu.Mem,
) error {

	return Status(C.cudnnIm2Col(
		handle.x,
		xD.descriptor,
		x.Ptr(),
		wD.descriptor,
		cD.descriptor,
		buffer.Ptr(),
	)).error("Im2Col")
}

//Algo returns an Algorithm Struct
func (c ConvBwdFiltAlgo) Algo() Algorithm {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforBWDFilter(&algorithm, c.c())
	return Algorithm(algorithm)
}

func (c ConvBwdFiltAlgo) print() {
	switch c {
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

//ConvBwdFiltAlgoPerformance is the return struct in the finding algorithm funcs
type ConvBwdFiltAlgoPerformance struct {
	Algo        ConvBwdFiltAlgo `json:"algo,omitempty"`
	Status      Status          `json:"status,omitempty"`
	Time        float32         `json:"time,omitempty"`
	Memory      uint            `json:"memory,omitempty"`
	Determinism Determinism     `json:"determinism,omitempty"`
	MathType    MathType        `json:"math_type,omitempty"`
}

func convertConvBwdFiltAlgoPerformance(input C.cudnnConvolutionBwdFilterAlgoPerf_t) ConvBwdFiltAlgoPerformance {
	var x ConvBwdFiltAlgoPerformance
	x.Algo = ConvBwdFiltAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)
	return x
}

//Print prints a human readable copy of the algorithm
func (cb ConvBwdFiltAlgoPerformance) Print() {
	fmt.Println("Convolution Backward FIlter Algorithm Performance")
	fmt.Println("-------------------------------------------------")
	ConvBwdFiltAlgo(cb.Algo).print()
	fmt.Println("Status:", Status(cb.Algo).GetErrorString())
	fmt.Println("Time:", cb.Time)
	fmt.Println("Memory:", cb.Memory)
	fmt.Println("Determinism:", cb.Determinism)
	fmt.Println("MathType:", cb.MathType)
}

//ConvolutionBwdFuncs is an empty struct that is used to organize and call backward convolution functions and helper functions
type ConvolutionBwdFuncs struct {
}

//GetConvolutionBackwardFilterAlgorithmMaxCount returns the max number of Algorithm
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardFilterAlgorithmMaxCount(handle *Handle) (int32, error) {

	var count C.int
	x := Status(C.cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionForwardAlgorithmMaxCount")

	return int32(count), x

}

//ConvolutionBackwardBias Function to compute the bias gradient for batch convolution db is returned
func (cbf ConvolutionBwdFuncs) ConvolutionBackwardBias(
	handle *Handle,
	alpha float64,
	dyD *TensorD,
	dy gocu.Mem,
	beta float64,
	dbD *TensorD,
	db gocu.Mem) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	return Status(C.cudnnConvolutionBackwardBias(handle.x, a.CPtr(), dyD.descriptor, dy.Ptr(), b.CPtr(), dbD.descriptor, db.Ptr())).error("ConvolutionBackwardBias")
}

//FindConvolutionBackwardFilterAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (cbf ConvolutionBwdFuncs) FindConvolutionBackwardFilterAlgorithm(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	cD *ConvolutionD,
	dwD *FilterD,
	requestedAlgoCount int32,
) ([]ConvBwdFiltAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionBackwardFilterAlgorithm(
		handle.x,
		xD.descriptor,
		dyD.descriptor,
		cD.descriptor,
		dwD.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0],
	)).error("FindConvolutionBackwardFilterAlgorithm")

	results := make([]ConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindConvolutionBackwardFilterAlgorithmEx finds some algorithms with memory
func (cbf ConvolutionBwdFuncs) FindConvolutionBackwardFilterAlgorithmEx(
	handle *Handle,
	xD *TensorD, x gocu.Mem,
	dyD *TensorD, dy gocu.Mem,
	cD *ConvolutionD,
	dwD *FilterD, dw gocu.Mem,
	reqAlgoCount int32,
	wspace gocu.Mem, wspacesize uint) ([]ConvBwdFiltAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionBackwardFilterAlgorithmEx(
		handle.x,
		xD.descriptor, x.Ptr(),
		dyD.descriptor, dy.Ptr(),
		cD.descriptor,
		dwD.descriptor, dw.Ptr(),
		C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), C.size_t(wspacesize))).error("FindConvolutionBackwardFilterAlgorithmEx")

	results := make([]ConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionBackwardFilterAlgorithm gives a good algo with the limits given to it
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardFilterAlgorithm(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	cD *ConvolutionD,
	dwD *FilterD,
	pref ConvBwdFilterPref, wsmemlimit uint) (ConvBwdFiltAlgo, error) {
	var algo C.cudnnConvolutionBwdFilterAlgo_t
	err := Status(C.cudnnGetConvolutionBackwardFilterAlgorithm(
		handle.x,
		xD.descriptor,
		dyD.descriptor,
		cD.descriptor,
		dwD.descriptor,
		pref.c(), C.size_t(wsmemlimit), &algo)).error("GetConvolutionBackwardFilterAlgorithm")

	return ConvBwdFiltAlgo(algo), err
}

//GetConvolutionBackwardFilterAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardFilterAlgorithmV7(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	cD *ConvolutionD,
	dwD *FilterD,
	requestedAlgoCount int32) ([]ConvBwdFiltAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnGetConvolutionBackwardFilterAlgorithm_v7(
		handle.x,
		xD.descriptor,
		dyD.descriptor,
		cD.descriptor,
		dwD.descriptor,
		C.int(requestedAlgoCount),
		&actualalgocount,
		&perfResults[0])).error("GetConvolutionBackwardFilterAlgorithm_v7")
	results := make([]ConvBwdFiltAlgoPerformance, int32(actualalgocount))

	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionBackwardFilterWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardFilterWorkspaceSize(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	cD *ConvolutionD,
	dwD *FilterD,
	algo ConvBwdFiltAlgo) (uint, error) {
	var sizebytes C.size_t
	err := Status(C.cudnnGetConvolutionBackwardFilterWorkspaceSize(
		handle.x,
		xD.descriptor,
		dyD.descriptor,
		cD.descriptor,
		dwD.descriptor,
		algo.c(),
		&sizebytes)).error("GetConvolutionForwardWorkspaceSize")

	return uint(sizebytes), err
}

//BackwardFilter does the backwards convolution
func (c *ConvolutionD) BackwardFilter(
	handle *Handle,
	alpha float64,
	xD *TensorD,
	x gocu.Mem,
	dyD *TensorD,
	dy gocu.Mem,

	algo ConvBwdFiltAlgo,
	wspace gocu.Mem,
	wspacesize uint,
	beta float64,
	dwD *FilterD,
	dw gocu.Mem,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	if wspace == nil {

		return Status(C.cudnnConvolutionBackwardFilter(
			handle.x,
			a.CPtr(),
			xD.descriptor,
			x.Ptr(),
			dyD.descriptor,
			dy.Ptr(),
			c.descriptor,
			algo.c(),
			nil,
			C.size_t(wspacesize),
			b.CPtr(),
			dwD.descriptor,
			dw.Ptr(),
		)).error("cudnnConvolutionBackwardFilter")

	}

	return Status(C.cudnnConvolutionBackwardFilter(
		handle.x,
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		c.descriptor,
		algo.c(),
		wspace.Ptr(),
		C.size_t(wspacesize),
		b.CPtr(),
		dwD.descriptor,
		dw.Ptr(),
	)).error("cudnnConvolutionBackwardFilter")
}

//Algo returns an Algorithm Struct
func (c ConvFwdAlgo) Algo() Algorithm {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforFWD(&algorithm, c.c())
	return Algorithm(algorithm)
}

func (c ConvFwdAlgo) toString() string {
	var x string
	switch c {
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM):
		x = "Implicit Gemm"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM):
		x = "Implicit Precomp Gemm"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM):
		x = "Gemm"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT):
		x = "Direct"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT):
		x = "FFT"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING):
		x = "FFT Tiling"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD):
		x = "WinoGrad"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED):
		x = "WinoGradNonFused"
	case ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT):
		x = "Count"
	default:
		x = "not supported algo --  to be honest ... I don't know how you got here"

	}
	return x
}

//Print prints a human readable copy of the algorithm
func (algoPerf ConvFwdAlgoPerformance) Print() {
	fmt.Println("Convolution Forward Filter Algorithm Performance")
	fmt.Println("-------------------------------------------------")
	ConvBwdFiltAlgo(algoPerf.Algo).print()
	fmt.Println("Status:", Status(algoPerf.Algo).GetErrorString())
	fmt.Println("Time:", algoPerf.Time)
	fmt.Println("Memory:", algoPerf.Memory)
	fmt.Println("Determinism:", algoPerf.Determinism)
	fmt.Println("MathType:", algoPerf.MathType)
}

//ConvFwdAlgoPerformance is a struct that holds the performance of the algorithm
type ConvFwdAlgoPerformance struct {
	Algo        ConvFwdAlgo `json:"algo,omitempty"`
	Status      Status      `json:"status,omitempty"`
	Time        float32     `json:"time,omitempty"`
	Memory      uint        `json:"memory,omitempty"`
	Determinism Determinism `json:"determinism,omitempty"`
	MathType    MathType    `json:"math_type,omitempty"`
}

func convertConvFwdAlgoPerformance(input C.cudnnConvolutionFwdAlgoPerf_t) ConvFwdAlgoPerformance {
	var x ConvFwdAlgoPerformance
	x.Algo = ConvFwdAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)

	return x
}

//ConvolutionFwdFuncs is an empty struct that is used to call Convolution Functions either operations or helpers
type ConvolutionFwdFuncs struct {
}

//GetConvolutionForwardAlgorithmMaxCount returns the max number of Algorithm
func (cfo ConvolutionFwdFuncs) GetConvolutionForwardAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionForwardAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionForwardAlgorithmMaxCount")

	return int32(count), x

}

//FindConvolutionForwardAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (cfo ConvolutionFwdFuncs) FindConvolutionForwardAlgorithm(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	cD *ConvolutionD,
	yD *TensorD,
	requestedAlgoCount int32) ([]ConvFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionForwardAlgorithm(handle.x, xD.descriptor, wD.descriptor, cD.descriptor, yD.descriptor, C.int(requestedAlgoCount), &actualalgocount, &perfResults[0])).error("FindConvolutionForwardAlgorithm")

	results := make([]ConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindConvolutionForwardAlgorithmEx finds some algorithms with memory
func (cfo ConvolutionFwdFuncs) FindConvolutionForwardAlgorithmEx(
	handle *Handle,
	xD *TensorD,
	x gocu.Mem,
	wD *FilterD,
	w gocu.Mem,
	cD *ConvolutionD,
	yD *TensorD,
	y gocu.Mem,
	reqAlgoCount int32,
	wspace gocu.Mem,
	wspacesize uint) ([]ConvFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	var err error
	if wspace == nil {
		err = Status(C.cudnnFindConvolutionForwardAlgorithmEx(handle.x, xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(), cD.descriptor, yD.descriptor, y.Ptr(), C.int(reqAlgoCount), &actualalgocount, &perfResults[0], nil, C.size_t(0))).error("FindConvolutionForwardAlgorithmEx")

	} else {
		err = Status(C.cudnnFindConvolutionForwardAlgorithmEx(handle.x, xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(), cD.descriptor, yD.descriptor, y.Ptr(), C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), C.size_t(wspacesize))).error("FindConvolutionForwardAlgorithmEx")

	}

	results := make([]ConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionForwardAlgorithm gives a good algo with the limits given to it
func (cfo ConvolutionFwdFuncs) GetConvolutionForwardAlgorithm(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	cD *ConvolutionD,
	yD *TensorD,
	pref ConvolutionForwardPref,
	wsmemlimit uint) (ConvFwdAlgo, error) {
	var algo C.cudnnConvolutionFwdAlgo_t
	err := Status(C.cudnnGetConvolutionForwardAlgorithm(handle.x, xD.descriptor, wD.descriptor, cD.descriptor, yD.descriptor, pref.c(), C.size_t(wsmemlimit), &algo)).error("GetConvolutionForwardAlgorithm")

	return ConvFwdAlgo(algo), err
}

//GetConvolutionForwardAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (cfo ConvolutionFwdFuncs) GetConvolutionForwardAlgorithmV7(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	cD *ConvolutionD,
	yD *TensorD,
	requestedAlgoCount int32) ([]ConvFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnGetConvolutionForwardAlgorithm_v7(handle.x, xD.descriptor, wD.descriptor, cD.descriptor, yD.descriptor, C.int(requestedAlgoCount), &actualalgocount, &perfResults[0])).error("FindConvolutionForwardAlgorithm")

	results := make([]ConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionForwardWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (cfo ConvolutionFwdFuncs) GetConvolutionForwardWorkspaceSize(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	cD *ConvolutionD,
	yD *TensorD,
	algo ConvFwdAlgo) (uint, error) {
	var sizebytes C.size_t
	err := Status(C.cudnnGetConvolutionForwardWorkspaceSize(handle.x, xD.descriptor, wD.descriptor, cD.descriptor, yD.descriptor, algo.c(), &sizebytes)).error("GetConvolutionForwardWorkspaceSize")

	return uint(sizebytes), err
}

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//Forward Function to perform the forward pass for batch convolution
func (c *ConvolutionD) Forward(
	handle *Handle,
	alpha float64,
	xD *TensorD,
	x gocu.Mem,
	wD *FilterD,
	w gocu.Mem,
	algo ConvFwdAlgo,
	wspace gocu.Mem,
	wspacesize uint,
	beta float64,
	yD *TensorD,
	y gocu.Mem) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if wspace == nil {

		return Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(),
			c.descriptor, algo.c(), nil, C.size_t(wspacesize), b.CPtr(), yD.descriptor, y.Ptr())).error("ConvolutionForward")
	}

	err := Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(),
		c.descriptor, algo.c(), wspace.Ptr(), C.size_t(wspacesize), b.CPtr(), yD.descriptor, y.Ptr())).error("ConvolutionForward")
	if err != nil {
		fmt.Println("\nError for ConvForward\n", "alpha: ", a, "\nbeta: ", b, "\nDesc: ", xD, "\nx :", x, "\nwD :", wD, "\nw: ", w, "\nwspace: ", wspace, "\nwspacesize: ", wspacesize, "\nyD: ", yD, "\ny: ", y)

		panic(err)
	}
	return err
}

//BiasActivationForward passes a lot of stuff so be carefull
/* Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
func (c *ConvolutionD) BiasActivationForward(
	handle *Handle,
	alpha1 float64,
	xD *TensorD,
	x gocu.Mem,
	wD *FilterD,
	w gocu.Mem,
	algo ConvFwdAlgo,
	wspace gocu.Mem,
	wspacesize uint,
	alpha2 float64,
	zD *TensorD,
	z gocu.Mem,
	biasD *TensorD,
	bias gocu.Mem,
	aD *ActivationD,
	yD *TensorD,
	y gocu.Mem,
) error {
	a1 := cscalarbydatatype(yD.dtype, alpha1)
	a2 := cscalarbydatatype(yD.dtype, alpha2)

	if wspace == nil {

		return Status(
			C.cudnnConvolutionBiasActivationForward(
				handle.x,
				a1.CPtr(),
				xD.descriptor,
				x.Ptr(),
				wD.descriptor,
				w.Ptr(),
				c.descriptor,
				algo.c(),
				nil,
				C.size_t(wspacesize),
				a2.CPtr(),
				zD.descriptor,
				z.Ptr(),
				biasD.descriptor,
				bias.Ptr(),
				aD.descriptor,
				yD.descriptor,
				y.Ptr(),
			)).error("ConvolutionBiasActivationForward")
	}

	return Status(
		C.cudnnConvolutionBiasActivationForward(
			handle.x,
			a1.CPtr(),
			xD.descriptor,
			x.Ptr(),
			wD.descriptor,
			w.Ptr(),
			c.descriptor,
			algo.c(),
			wspace.Ptr(),
			C.size_t(wspacesize),
			a2.CPtr(),
			zD.descriptor,
			z.Ptr(),
			biasD.descriptor,
			bias.Ptr(),
			aD.descriptor,
			yD.descriptor,
			y.Ptr(),
		)).error("ConvolutionBiasActivationForward")
}

/*

Flags


*/

//ConvolutionFlags is used to store the different Convolution Flags.  Hopefully it can make it easier when
//Using something like VSCode.
type ConvolutionFlags struct {
	Mode ConvolutionMode
	Bwd  ConvolutionBwdFlags
	Fwd  ConvolutionFwdFlags
}

//ConvolutionFwdFlags holds the different flags used for the convlution fwd
type ConvolutionFwdFlags struct {
	Pref ConvolutionForwardPref
	Algo ConvFwdAlgo
}

//ConvolutionBwdFlags holds the different type of BwdConvolutionFlags
type ConvolutionBwdFlags struct {
	DataPref ConvBwdDataPref
	DataAlgo ConvBwdDataAlgo
	FltrPref ConvBwdFilterPref
	FltrAlgo ConvBwdFiltAlgo
}

/*
*
*
*       ConvolutionMode
*
*
 */

//ConvolutionMode is the type to describe the convolution mode flags
type ConvolutionMode C.cudnnConvolutionMode_t

//Convolution sets and returns value of c to ConvolutionMode(C.CUDNN_CONVOLUTION)
func (c *ConvolutionMode) Convolution() ConvolutionMode {
	*c = ConvolutionMode(C.CUDNN_CONVOLUTION)
	return *c
}

// CrossCorrelation n sets and returns value of c to  ConvolutionMode(C.CUDNN_CROSS_CORRELATION)
func (c *ConvolutionMode) CrossCorrelation() ConvolutionMode {
	*c = ConvolutionMode(C.CUDNN_CROSS_CORRELATION)
	return *c
}

func (c ConvolutionMode) c() C.cudnnConvolutionMode_t { return C.cudnnConvolutionMode_t(c) }

/*
*
*
*       ConvBwdDataPrefFlag
*
*
 */

//ConvBwdDataPref used for flags on bwddatapref exposing them through methods
type ConvBwdDataPref C.cudnnConvolutionBwdDataPreference_t

//NoWorkSpace sets c to returns ConvBwdDataPref( C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE) and returns value of c
func (c *ConvBwdDataPref) NoWorkSpace() ConvBwdDataPref {
	*c = ConvBwdDataPref(C.CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE)
	return *c
}

//PreferFastest  sets c to ConvBwdDataPref( C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST) and returns value of c
func (c *ConvBwdDataPref) PreferFastest() ConvBwdDataPref {
	*c = ConvBwdDataPref(C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
	return *c
}

//SpecifyWorkSpaceLimit  sets c to ConvBwdDataPref( C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)and returns value of c
func (c *ConvBwdDataPref) SpecifyWorkSpaceLimit() ConvBwdDataPref {
	*c = ConvBwdDataPref(C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
	return *c
}

func (c ConvBwdDataPref) c() C.cudnnConvolutionBwdDataPreference_t {
	return C.cudnnConvolutionBwdDataPreference_t(c)
}

/*
*
*
*       ConvBwdDataAlgoFlag
*
*
 */

//ConvBwdDataAlgo used for flags in the bacward data algorithms  exposing them through methods
type ConvBwdDataAlgo C.cudnnConvolutionBwdDataAlgo_t

//Algo0  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)  and returns value of c /* non-deterministic */
func (c *ConvBwdDataAlgo) Algo0() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
	return *c
}

//Algo1  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)  and returns value of c
func (c *ConvBwdDataAlgo) Algo1() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)
	return *c
}

//FFT  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)  and returns value of c
func (c *ConvBwdDataAlgo) FFT() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)
	return *c
}

//FFTTiling  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)  and returns value of c
func (c *ConvBwdDataAlgo) FFTTiling() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)
	return *c
}

//Winograd 	 sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)  and returns value of c
func (c *ConvBwdDataAlgo) Winograd() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
	return *c
}

//WinogradNonFused  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)  and returns value of c
func (c *ConvBwdDataAlgo) WinogradNonFused() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)
	return *c
}

//Count  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)  and returns value of c
func (c *ConvBwdDataAlgo) Count() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)
	return *c
}
func (c ConvBwdDataAlgo) c() C.cudnnConvolutionBwdDataAlgo_t {
	return C.cudnnConvolutionBwdDataAlgo_t(c)
}

/*
*
*
*       ConvBwdFilterPrefFlag
*
*
 */

//ConvBwdFilterPref are used for flags for the backwds filters  exposing them through methods
type ConvBwdFilterPref C.cudnnConvolutionBwdFilterPreference_t

//NoWorkSpace sets c to  ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)  and returns value of c
func (c *ConvBwdFilterPref) NoWorkSpace() ConvBwdFilterPref {
	*c = ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
	return *c
}

//PrefFastest sets c to  ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)  and returns value of c
func (c *ConvBwdFilterPref) PrefFastest() ConvBwdFilterPref {
	*c = ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
	return *c
}

//SpecifyWorkSpaceLimit sets c to  ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)  and returns value of c
func (c *ConvBwdFilterPref) SpecifyWorkSpaceLimit() ConvBwdFilterPref {
	*c = ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
	return *c
}

func (c ConvBwdFilterPref) c() C.cudnnConvolutionBwdFilterPreference_t {
	return C.cudnnConvolutionBwdFilterPreference_t(c)
}

/*
*
*
*       ConvBwdFiltAlgoFlag
*
*
 */

//ConvBwdFiltAlgo Used for ConvBwdFiltAlgo flags  exposing them through methods
type ConvBwdFiltAlgo C.cudnnConvolutionBwdFilterAlgo_t

//Algo0 sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0) and returns value of c /* non-deterministic */
func (c *ConvBwdFiltAlgo) Algo0() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)
	return *c
}

//Algo1 sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1) and returns value of c
func (c *ConvBwdFiltAlgo) Algo1() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
	return *c
}

//FFT sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT) and returns value of c
func (c *ConvBwdFiltAlgo) FFT() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT)
	return *c
}

//Algo3 sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3) and returns value of c
func (c *ConvBwdFiltAlgo) Algo3() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)
	return *c
}

//Winograd 	sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD) and returns value of c
func (c *ConvBwdFiltAlgo) Winograd() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD)
	return *c
}

//WinogradNonFused sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED) and returns value of c
func (c *ConvBwdFiltAlgo) WinogradNonFused() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED)
	return *c
}

//FFTTiling sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING) and returns value of c
func (c *ConvBwdFiltAlgo) FFTTiling() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING)
	return *c
}

//Count sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT) and returns value of c
func (c *ConvBwdFiltAlgo) Count() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)
	return *c
}
func (c ConvBwdFiltAlgo) c() C.cudnnConvolutionBwdFilterAlgo_t {
	return C.cudnnConvolutionBwdFilterAlgo_t(c)
}

/*
*
*
*       ConvolutionFwdPrefFlag
*
*
 */

// ConvolutionForwardPref used for flags  exposing them through methods
type ConvolutionForwardPref C.cudnnConvolutionFwdPreference_t

func (c ConvolutionForwardPref) c() C.cudnnConvolutionFwdPreference_t {
	return C.cudnnConvolutionFwdPreference_t(c)
}

//NoWorkSpace sets c to ConvolutionForwardPref( C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE) and returns value of c
func (c *ConvolutionForwardPref) NoWorkSpace() ConvolutionForwardPref {
	*c = ConvolutionForwardPref(C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE)
	return *c
}

//PreferFastest returns ConvolutionForwardPref( C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
func (c *ConvolutionForwardPref) PreferFastest() ConvolutionForwardPref {
	*c = ConvolutionForwardPref(C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
	return *c
}

//SpecifyWorkSpaceLimit returns ConvolutionForwardPref( C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
func (c *ConvolutionForwardPref) SpecifyWorkSpaceLimit() ConvolutionForwardPref {
	*c = ConvolutionForwardPref(C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
	return *c
}

/*
*
*
*       ConvFwdAlgoFlag
*
*
 */

//ConvFwdAlgo flags for cudnnConvFwdAlgo_t  exposing them through methods
type ConvFwdAlgo C.cudnnConvolutionFwdAlgo_t

//ImplicitGemm sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) and returns value of c
func (c *ConvFwdAlgo) ImplicitGemm() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
	return *c
}

//ImplicitPrecompGemm sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) and returns value of c
func (c *ConvFwdAlgo) ImplicitPrecompGemm() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
	return *c
}

//Gemm sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM) and returns value of c
func (c *ConvFwdAlgo) Gemm() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
	return *c
}

//Direct sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT) and returns value of c
func (c *ConvFwdAlgo) Direct() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
	return *c
}

//FFT sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_FFT) and returns value of c
func (c *ConvFwdAlgo) FFT() ConvFwdAlgo { *c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT); return *c }

//FFTTiling sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) and returns value of c
func (c *ConvFwdAlgo) FFTTiling() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
	return *c
}

//WinoGrad sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD) and returns value of c
func (c *ConvFwdAlgo) WinoGrad() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
	return *c
}

//WinoGradNonFused sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED) and returns value of c
func (c *ConvFwdAlgo) WinoGradNonFused() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
	return *c
}

//Count sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT) and returns value of c
func (c *ConvFwdAlgo) Count() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
	return *c
}

func (c ConvFwdAlgo) c() C.cudnnConvolutionFwdAlgo_t {
	return C.cudnnConvolutionFwdAlgo_t(c)
}

/*
//Set2D sets convolution descriptor to 2D
func (c *ConvolutionD) Set2D(mode ConvolutionMode, data DataType, pad, stride, dialation []int32) error {
	if len(pad) != len(stride) || len(pad) != len(dialation) || len(pad) != 2 {
		return errors.New("pad stride and dialation need to be size 2")
	}
	cdata := data.c()
	cmode := mode.c()
	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdialation := int32Tocint(dialation)
	c.isconv2d = true
	if convolutionnd2dtestflag {
		dims := C.int(len(pad))
		return Status(C.cudnnSetConvolutionNdDescriptor(c.descriptor, dims, &cpad[0], &cstride[0], &cdialation[0], cmode, cdata)).error("NewConvolutionNdDescriptor-Set2D")

	}
	return Status(C.cudnnSetConvolution2dDescriptor(c.descriptor, cpad[0], cpad[1], cstride[0], cstride[1], cdialation[0], cdialation[1], cmode, cdata)).error("NewConvolution2dDescriptor-set")
}

//SetND sets the convolution descriptor to ND
func (c *ConvolutionD) SetND(mode ConvolutionMode, data DataType, pad, stride, dialation []int32) error {
	cdata := data.c()
	cmode := mode.c()
	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdialation := int32Tocint(dialation)
	dims := C.int(len(pad))
	return Status(C.cudnnSetConvolutionNdDescriptor(c.descriptor, dims, &cpad[0], &cstride[0], &cdialation[0], cmode, cdata)).error("NewConvolutionNdDescriptor-set")

}
*/
/*
//GetConvolution2dForwardOutputDim is a helper func that will output the shape of the convolution
func (c *ConvolutionD) GetConvolution2dForwardOutputDim(input *TensorD, filter *FilterD) ([]int32, error) {
	var shape [4]C.int
	x := Status(C.cudnnGetConvolution2dForwardOutputDim(c.descriptor, input.descriptor, filter.descriptor,
		&shape[0], &shape[1], &shape[2], &shape[3]))
	retshap := cintToint32(shape[:4])

	return retshap, x.error("GetConvolution2dForwardOutputDim")

}
*/
