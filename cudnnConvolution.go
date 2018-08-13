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
	"errors"
	"fmt"
)

//Convolution is a helper struct that is helpfun when coding in something like vs code. It will allow methods and flags to be
//easily accessed through intellisense.
type Convolution struct {
	Funcs ConvolutionFuncs
	Flgs  ConvolutionFlags
}

//ConvolutionFuncs contains the different operations that can be done with convolution
type ConvolutionFuncs struct {
	Fwd ConvolutionFwdFuncs
	Bwd ConvolutionBwdFuncs
}

/*


Descriptors


*/

//ConvolutionD sets all the convolution info
type ConvolutionD struct {
	descriptor C.cudnnConvolutionDescriptor_t
	dims       C.int
	flag       descflag
}

/*
//Pads is a convienence func
func Pads(pad ...int32) []int32 {
	return pad
}

//Dialation is a convienence func
func Dialation(dialation ...int32) []int32 {
	return dialation
}
*/

//NewConvolution2dDescriptor creates and sets a 2d convolution descriptor
func (conv Convolution) NewConvolution2dDescriptor(mode ConvolutionMode, data DataType, pad, stride, dialation []int32) (*ConvolutionD, error) {
	if len(pad) != len(stride) || len(pad) != len(dialation) || len(pad) != 2 {
		return nil, errors.New("pad stride and dialation need to be size 2")
	}
	var descriptor C.cudnnConvolutionDescriptor_t
	err := Status(C.cudnnCreateConvolutionDescriptor(&descriptor)).error("NewConvolution2dDescriptor-create")
	if err != nil {
		return nil, err
	}
	cdata := data.c()
	cmode := mode.c()
	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdialation := int32Tocint(dialation)

	err = Status(C.cudnnSetConvolution2dDescriptor(descriptor, cpad[0], cpad[1], cstride[0], cstride[1], cdialation[0], cdialation[1], cmode, cdata)).error("NewConvolution2dDescriptor-set")
	if err != nil {
		return nil, err
	}
	return &ConvolutionD{descriptor: descriptor, dims: 2, flag: t2d}, nil
}

//NewConvolutionNdDescriptor creates and sets a new Convolution ND descriptor
func (conv Convolution) NewConvolutionNdDescriptor(mode ConvolutionMode, data DataType, pad, stride, dialation []int32) (*ConvolutionD, error) {
	if len(pad) != len(stride) || len(pad) != len(dialation) || len(pad) < 2 {
		return nil, errors.New("pad stride and dialation need to be size 2 or greater")
	}
	var descriptor C.cudnnConvolutionDescriptor_t
	err := Status(C.cudnnCreateConvolutionDescriptor(&descriptor)).error("NewConvolution2dDescriptor-create")
	if err != nil {
		return nil, err
	}
	cdata := C.cudnnDataType_t(data)
	cmode := C.cudnnConvolutionMode_t(mode)
	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdialation := int32Tocint(dialation)
	//var holder C.int
	dims := C.int(len(pad))
	err = Status(C.cudnnSetConvolutionNdDescriptor(descriptor, dims, &cpad[0], &cstride[0], &cdialation[0], cmode, cdata)).error("NewConvolutionNdDescriptor-set")
	if err != nil {
		return nil, err
	}
	return &ConvolutionD{descriptor: descriptor, dims: dims, flag: tnd}, nil
}

//GetDescriptor gets returns the values used to make the convolution descriptor
func (c *ConvolutionD) GetDescriptor() (ConvolutionMode, DataType, []int32, []int32, []int32, error) {
	if c.flag == t2d {
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

//GetConvolution2dForwardOutputDim is a helper func that will output the shape of the convolution
func (c *ConvolutionD) GetConvolution2dForwardOutputDim(input *TensorD, filter *FilterD) ([]int32, error) {
	var shape [4]C.int
	x := Status(C.cudnnGetConvolution2dForwardOutputDim(c.descriptor, input.descriptor, filter.descriptor,
		&shape[0], &shape[1], &shape[2], &shape[3]))
	retshap := cintToint32(shape[:4])

	return retshap, x.error("GetConvolution2dForwardOutputDim")

}

//GetConvolutionNdForwardOutputDim is a helper function to give the size of the output of of a COnvolutionNDForward
func (c *ConvolutionD) GetConvolutionNdForwardOutputDim(input *TensorD, filter *FilterD) ([]int32, error) {
	dims := make([]C.int, int32(c.dims))
	x := Status(C.cudnnGetConvolutionNdForwardOutputDim(c.descriptor, input.descriptor, filter.descriptor, c.dims, &dims[0])).error("GetConvolutionNdForwardOutputDim")

	return cintToint32(dims), x
}

//DestroyDescriptor destroys the ConvolutionDescriptor
func (c *ConvolutionD) DestroyDescriptor() error {
	x := Status(C.cudnnDestroyConvolutionDescriptor(c.descriptor)).error("DestroyConvolutionDescriptor")
	return x
}

/* helper function to provide the convolution algo that fit best the requirement */

//Algo returns an Algorithm struct
func (cbd ConvBwdDataAlgo) Algo() Algos {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforBWDData(&algorithm, cbd.c())
	return Algos(algorithm)

}

//ConvBwdDataAlgoPerf is used to find the best/fastest algorithms
//type ConvBwdDataAlgoPerformance C.cudnnConvolutionBwdDataAlgoPerf_t

//ConvBwdDataAlgoPerformance is the return struct in the finding algorithm funcs
type ConvBwdDataAlgoPerformance struct {
	Algo        ConvBwdDataAlgo
	Status      Status
	Time        float32
	Memory      SizeT
	Determinism Determinism
	MathType    MathType
}

func convertConvBwdDataAlgoPerformance(input C.cudnnConvolutionBwdDataAlgoPerf_t) ConvBwdDataAlgoPerformance {
	var x ConvBwdDataAlgoPerformance
	x.Algo = ConvBwdDataAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = SizeT(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)
	return x
}

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

/*
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
*/

//GetConvolutionBackwardDataAlgorithmMaxCount returns the max number of algos
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
	wDesc *FilterD, wMem Memer,
	dyDesc *TensorD, dy Memer,
	conDesc *ConvolutionD,
	dxDesc *TensorD, dx Memer,
	reqAlgoCount int32, wspace Memer) ([]ConvBwdDataAlgoPerformance, error) {
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

	results := make([]ConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdDataAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionBackwardDataAlgorithm gives a good algo with the limits given to it
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardDataAlgorithm(
	handle *Handle,
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
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardDataAlgorithmV7(
	handle *Handle,
	filt *FilterD,
	diff *TensorD,
	c *ConvolutionD,
	grad *TensorD,
	requestedAlgoCount int32) ([]ConvBwdDataAlgoPerformance, error) {
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
	results := make([]ConvBwdDataAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdDataAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionBackwardDataWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardDataWorkspaceSize(
	handle *Handle,
	w *FilterD,
	dy *TensorD,
	c *ConvolutionD,
	dx *TensorD,
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
func (cbf ConvolutionBwdFuncs) ConvolutionBackwardData(
	handle *Handle,
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
	if wspace == nil {
		return Status(C.cudnnConvolutionBackwardData(
			handle.x,
			alpha.CPtr(),
			wDesc.descriptor,
			w.Ptr(),
			dyDesc.descriptor,
			dy.Ptr(),
			convDesc.descriptor,
			algo.c(),
			nil,
			SizeT(0).c(),
			beta.CPtr(),
			dxDesc.descriptor,
			dx.Ptr(),
		)).error("ConvolutionBackwardData")
	}
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
func (cbf ConvolutionBwdFuncs) Im2Col(
	handle *Handle,
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

//Algo returns an Algorithm Struct
func (cb ConvBwdFiltAlgo) Algo() Algos {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforBWDFilter(&algorithm, cb.c())
	return Algos(algorithm)
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

//ConvBwdFiltAlgoPerformance is the return struct in the finding algorithm funcs
type ConvBwdFiltAlgoPerformance struct {
	Algo        ConvBwdFiltAlgo
	Status      Status
	Time        float32
	Memory      SizeT
	Determinism Determinism
	MathType    MathType
}

func convertConvBwdFiltAlgoPerformance(input C.cudnnConvolutionBwdFilterAlgoPerf_t) ConvBwdFiltAlgoPerformance {
	var x ConvBwdFiltAlgoPerformance
	x.Algo = ConvBwdFiltAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = SizeT(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.MathType = MathType(input.mathType)
	return x
}

/*
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
*/

//ConvolutionBwdFuncs is an empty struct that is used to organize and call backward convolution functions and helper functions
type ConvolutionBwdFuncs struct {
}

//GetConvolutionBackwardFilterAlgorithmMaxCount returns the max number of algos
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardFilterAlgorithmMaxCount(handle *Handle) (int32, error) {

	var count C.int
	x := Status(C.cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionForwardAlgorithmMaxCount")
	return int32(count), x

}

//ConvolutionBackwardBias Function to compute the bias gradient for batch convolution db is returned
func (cbf ConvolutionBwdFuncs) ConvolutionBackwardBias(
	handle *Handle,
	alpha CScalar,
	dyD *TensorD,
	dy Memer,
	beta CScalar,
	dbD *TensorD,
	db Memer) error {
	return Status(C.cudnnConvolutionBackwardBias(handle.x, alpha.CPtr(), dyD.descriptor, dy.Ptr(), beta.CPtr(), dbD.descriptor, db.Ptr())).error("ConvolutionBackwardBias")
}

//FindConvolutionBackwardFilterAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (cbf ConvolutionBwdFuncs) FindConvolutionBackwardFilterAlgorithm(
	handle *Handle,
	x *TensorD,
	dy *TensorD,
	c *ConvolutionD,
	dw *FilterD,
	requestedAlgoCount int32,
) ([]ConvBwdFiltAlgoPerformance, error) {
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
	results := make([]ConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindConvolutionBackwardFilterAlgorithmEx finds some algorithms with memory
func (cbf ConvolutionBwdFuncs) FindConvolutionBackwardFilterAlgorithmEx(
	handle *Handle,
	xDesc *TensorD, xMem Memer,
	dyDesc *TensorD, y Memer,
	conDesc *ConvolutionD,
	dwDesc *FilterD, dw Memer,
	reqAlgoCount int32, wspace Memer) ([]ConvBwdFiltAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionBwdFilterAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionBackwardFilterAlgorithmEx(
		handle.x,
		xDesc.descriptor, xMem.Ptr(),
		dyDesc.descriptor, y.Ptr(),
		conDesc.descriptor,
		dwDesc.descriptor, dw.Ptr(),
		C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), wspace.ByteSize().c())).error("FindConvolutionBackwardFilterAlgorithmEx")

	results := make([]ConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionBackwardFilterAlgorithm gives a good algo with the limits given to it
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardFilterAlgorithm(
	handle *Handle,
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
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardFilterAlgorithmV7(
	handle *Handle,
	src *TensorD,
	diff *TensorD,
	c *ConvolutionD,
	grad *FilterD,
	requestedAlgoCount int32) ([]ConvBwdFiltAlgoPerformance, error) {
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
	results := make([]ConvBwdFiltAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvBwdFiltAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionBackwardFilterWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (cbf ConvolutionBwdFuncs) GetConvolutionBackwardFilterWorkspaceSize(
	handle *Handle,
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
func (cbf ConvolutionBwdFuncs) ConvolutionBackwardFilter(
	handle *Handle,
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
	if wspace == nil {
		return Status(C.cudnnConvolutionBackwardFilter(
			handle.x,
			alpha.CPtr(),
			xDesc.descriptor,
			x.Ptr(),
			dyDesc.descriptor,
			dy.Ptr(),
			convDesc.descriptor,
			algo.c(),
			nil,
			SizeT(0).c(),
			beta.CPtr(),
			dwDesc.descriptor,
			dw.Ptr(),
		)).error("cudnnConvolutionBackwardFilter")
	}
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

//Algo returns an Algorithm Struct
func (a ConvFwdAlgo) Algo() Algos {
	var algorithm C.cudnnAlgorithm_t
	C.MakeAlgorithmforFWD(&algorithm, a.c())
	return Algos(algorithm)
}

func (a ConvFwdAlgo) toString() string {
	var x string
	switch a {
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

//PrintReadable prints this so that it is readable to a human
func (algoPerf ConvFwdAlgoPerformance) PrintReadable(index int) {
	fmt.Println("")
	fmt.Println("")
	holder := make([]interface{}, 6)
	holder[0] = algoPerf.Algo.toString()
	holder[1] = algoPerf.Status.GetErrorString()
	holder[2] = algoPerf.Time
	holder[3] = algoPerf.Memory
	holder[4] = algoPerf.Determinism.string()
	holder[5] = algoPerf.Mathtype.string()

	fmt.Println("Algo Perf", index)
	fmt.Println("---------------")
	for i := 0; i < len(holder); i++ {
		fmt.Println(holder[i])
	}
}

//ConvFwdAlgoPerformance is a struct that holds the performance of the algorithm
type ConvFwdAlgoPerformance struct {
	Algo        ConvFwdAlgo
	Status      Status
	Time        float32
	Memory      SizeT
	Determinism Determinism
	Mathtype    MathType
}

func convertConvFwdAlgoPerformance(input C.cudnnConvolutionFwdAlgoPerf_t) ConvFwdAlgoPerformance {
	var x ConvFwdAlgoPerformance
	x.Algo = ConvFwdAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = SizeT(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.Mathtype = MathType(input.mathType)

	return x
}

//ConvolutionFwdFuncs is an empty struct that is used to call Convolution Functions either operations or helpers
type ConvolutionFwdFuncs struct {
}

//GetConvolutionForwardAlgorithmMaxCount returns the max number of algos
func (cfo ConvolutionFwdFuncs) GetConvolutionForwardAlgorithmMaxCount(handle *Handle) (int32, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionForwardAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionForwardAlgorithmMaxCount")
	return int32(count), x

}

//FindConvolutionForwardAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (cfo ConvolutionFwdFuncs) FindConvolutionForwardAlgorithm(
	handle *Handle,
	x *TensorD,
	w *FilterD,
	c *ConvolutionD,
	y *TensorD,
	requestedAlgoCount int32) ([]ConvFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionForwardAlgorithm(handle.x, x.descriptor, w.descriptor, c.descriptor, y.descriptor, C.int(requestedAlgoCount), &actualalgocount, &perfResults[0])).error("FindConvolutionForwardAlgorithm")
	results := make([]ConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindConvolutionForwardAlgorithmEx finds some algorithms with memory
func (cfo ConvolutionFwdFuncs) FindConvolutionForwardAlgorithmEx(
	handle *Handle,
	xDesc *TensorD,
	xMem Memer,
	wDesc *FilterD,
	wMem Memer,
	conDesc *ConvolutionD,
	yDesc *TensorD,
	yMem Memer,
	reqAlgoCount int32,
	wspace Memer) ([]ConvFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionForwardAlgorithmEx(handle.x, xDesc.descriptor, xMem.Ptr(), wDesc.descriptor, wMem.Ptr(), conDesc.descriptor, yDesc.descriptor, yMem.Ptr(), C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), wspace.ByteSize().c())).error("FindConvolutionForwardAlgorithmEx")

	results := make([]ConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionForwardAlgorithm gives a good algo with the limits given to it
func (cfo ConvolutionFwdFuncs) GetConvolutionForwardAlgorithm(
	handle *Handle,
	xDesc *TensorD,
	wDesc *FilterD,
	convDesc *ConvolutionD,
	yDesc *TensorD,
	pref ConvolutionFwdPreference,
	wsmemlimit SizeT) (ConvFwdAlgo, error) {
	var algo C.cudnnConvolutionFwdAlgo_t
	err := Status(C.cudnnGetConvolutionForwardAlgorithm(handle.x, xDesc.descriptor, wDesc.descriptor, convDesc.descriptor, yDesc.descriptor, C.cudnnConvolutionFwdPreference_t(pref), C.size_t(wsmemlimit), &algo)).error("GetConvolutionForwardAlgorithm")
	return ConvFwdAlgo(algo), err
}

//GetConvolutionForwardAlgorithmV7 will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (cfo ConvolutionFwdFuncs) GetConvolutionForwardAlgorithmV7(
	handle *Handle,
	x *TensorD,
	w *FilterD,
	c *ConvolutionD,
	y *TensorD,
	requestedAlgoCount int32) ([]ConvFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnGetConvolutionForwardAlgorithm_v7(handle.x, x.descriptor, w.descriptor, c.descriptor, y.descriptor, C.int(requestedAlgoCount), &actualalgocount, &perfResults[0])).error("FindConvolutionForwardAlgorithm")
	results := make([]ConvFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionForwardWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (cfo ConvolutionFwdFuncs) GetConvolutionForwardWorkspaceSize(
	handle *Handle,
	x *TensorD,
	w *FilterD,
	c *ConvolutionD,
	y *TensorD,
	algo ConvFwdAlgo) (SizeT, error) {
	var sizebytes C.size_t
	err := Status(C.cudnnGetConvolutionForwardWorkspaceSize(handle.x, x.descriptor, w.descriptor, c.descriptor, y.descriptor, algo.c(), &sizebytes)).error("GetConvolutionForwardWorkspaceSize")
	return SizeT(sizebytes), err
}

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//ConvolutionForward Function to perform the forward pass for batch convolution
func (cfo ConvolutionFwdFuncs) ConvolutionForward(
	handle *Handle,
	alpha CScalar,
	xD *TensorD,
	x Memer,
	wD *FilterD,
	w Memer,
	cD *ConvolutionD,
	algo ConvFwdAlgo,
	wspace Memer,
	beta CScalar,
	yD *TensorD,
	y Memer) error {
	if wspace == nil {
		return Status(C.cudnnConvolutionForward(handle.x, alpha.CPtr(), xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(),
			cD.descriptor, algo.c(), nil, SizeT(0).c(), beta.CPtr(), yD.descriptor, y.Ptr())).error("ConvolutionForward")
	}
	return Status(C.cudnnConvolutionForward(handle.x, alpha.CPtr(), xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(),
		cD.descriptor, algo.c(), wspace.Ptr(), wspace.ByteSize().c(), beta.CPtr(), yD.descriptor, y.Ptr())).error("ConvolutionForward")

}

//ConvolutionBiasActivationForward passes a lot of stuff so be carefull
/* Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
func (cfo ConvolutionFwdFuncs) ConvolutionBiasActivationForward(
	handle *Handle,
	alpha1 CScalar,
	xD *TensorD,
	x Memer,
	wD *FilterD,
	w Memer,
	cD *ConvolutionD,
	algo ConvFwdAlgo,
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
	if wspace == nil {
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
				nil,
				SizeT(0).c(),
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

/*

Flags


*/

//ConvolutionFlags is used to store the different Convolution Flags.  Hopefully it can make it easier when
//Using something like VSCode.
type ConvolutionFlags struct {
	Mode ConvolutionModeFlag
	Bwd  ConvolutionBwdFlags
	Fwd  ConvolutionFwdFlags
}

//ConvolutionFwdFlags holds the different flags used for the convlution fwd
type ConvolutionFwdFlags struct {
	Pref ConvolutionFwdPrefFlag
	Algo ConvFwdAlgoFlag
}

//ConvolutionBwdFlags holds the different type of BwdConvolutionFlags
type ConvolutionBwdFlags struct {
	DataPref ConvBwdDataPrefFlag
	DataAlgo ConvBwdDataAlgoFlag
	FltrPref ConvBwdFilterPrefFlag
	FltrAlgo ConvBwdFiltAlgoFlag
}

/*
*
*
*       ConvolutionModeFlag
*
*
 */

//ConvolutionModeFlag is used to pass Convolution Mode Flags in a
//semi-safe way for human users by using methods
type ConvolutionModeFlag struct {
}

//ConvolutionMode is the type to describe the convolution mode flags
type ConvolutionMode C.cudnnConvolutionMode_t

//Convolution returns  ConvolutionMode(C.CUDNN_CONVOLUTION)
func (c ConvolutionModeFlag) Convolution() ConvolutionMode {
	return ConvolutionMode(C.CUDNN_CONVOLUTION)
}

// CrossCorrelation returns ConvolutionMode(C.CUDNN_CROSS_CORRELATION)
func (c ConvolutionModeFlag) CrossCorrelation() ConvolutionMode {
	return ConvolutionMode(C.CUDNN_CROSS_CORRELATION)
}

func (c ConvolutionMode) c() C.cudnnConvolutionMode_t { return C.cudnnConvolutionMode_t(c) }

/*
*
*
*       ConvBwdDataPrefFlag
*
*
 */

//ConvBwdDataPrefFlag used to pass ConvBwdDataPref flags through methods
type ConvBwdDataPrefFlag struct {
}

//ConvBwdDataPref used for flags on bwddatapref
type ConvBwdDataPref C.cudnnConvolutionBwdDataPreference_t

//NoWorkSpace returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE)
func (c ConvBwdDataPrefFlag) NoWorkSpace() ConvBwdDataPref {
	return ConvBwdDataPref(C.CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE)
}

//PreferFastest returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
func (c ConvBwdDataPrefFlag) PreferFastest() ConvBwdDataPref {
	return ConvBwdDataPref(C.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST)
}

//SpecifyWorkSpaceLimit returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
func (c ConvBwdDataPrefFlag) SpecifyWorkSpaceLimit() ConvBwdDataPref {
	return ConvBwdDataPref(C.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT)
}

func (cbd ConvBwdDataPref) c() C.cudnnConvolutionBwdDataPreference_t {
	return C.cudnnConvolutionBwdDataPreference_t(cbd)
}

/*
*
*
*       ConvBwdDataAlgoFlag
*
*
 */

//ConvBwdDataAlgoFlag is used to pass ConvBwdDataAlgo Flags
type ConvBwdDataAlgoFlag struct {
}

//ConvBwdDataAlgo used for flags in the bacward data algorithms
type ConvBwdDataAlgo C.cudnnConvolutionBwdDataAlgo_t

//Algo0 return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0) /* non-deterministic */
func (c ConvBwdDataAlgoFlag) Algo0() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
}

//Algo1 return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)
func (c ConvBwdDataAlgoFlag) Algo1() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)
}

//FFT return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)
func (c ConvBwdDataAlgoFlag) FFT() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)
}

//FFTTiling return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)
func (c ConvBwdDataAlgoFlag) FFTTiling() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)
}

//Winograd 	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
func (c ConvBwdDataAlgoFlag) Winograd() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
}

//WinogradNonFused return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)
func (c ConvBwdDataAlgoFlag) WinogradNonFused() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)
}

//Count return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)
func (c ConvBwdDataAlgoFlag) Count() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)
}

func (cbd ConvBwdDataAlgo) c() C.cudnnConvolutionBwdDataAlgo_t {
	return C.cudnnConvolutionBwdDataAlgo_t(cbd)
}

/*
*
*
*       ConvBwdFilterPrefFlag
*
*
 */

//ConvBwdFilterPref are used for flags for the backwds filters
type ConvBwdFilterPref C.cudnnConvolutionBwdFilterPreference_t

//ConvBwdFilterPrefFlag is used to pass ConvBwdFilterPref flags through methods
type ConvBwdFilterPrefFlag struct {
}

//NoWorkSpace return ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
func (c ConvBwdFilterPrefFlag) NoWorkSpace() ConvBwdFilterPref {
	return ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
}

//PrefFastest return ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)
func (c ConvBwdFilterPrefFlag) PrefFastest() ConvBwdFilterPref {
	return ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)
}

//SpecifyWorkSpaceLimit return ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)
func (c ConvBwdFilterPrefFlag) SpecifyWorkSpaceLimit() ConvBwdFilterPref {
	return ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)
}

func (bw ConvBwdFilterPref) c() C.cudnnConvolutionBwdFilterPreference_t {
	return C.cudnnConvolutionBwdFilterPreference_t(bw)
}

/*
*
*
*       ConvBwdFiltAlgoFlag
*
*
 */

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

/*
*
*
*       ConvolutionFwdPrefFlag
*
*
 */

// ConvolutionFwdPreference used for flags
type ConvolutionFwdPreference C.cudnnConvolutionFwdPreference_t

/* helper function to provide the convolution algo that fit best the requirement */
//these are flags for ConvolutionFwdPreference

//ConvolutionFwdPrefFlag transfer flags for ConvolutionFwdPreference through methods
type ConvolutionFwdPrefFlag struct {
}

//NoWorkSpace returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE)
func (c ConvolutionFwdPrefFlag) NoWorkSpace() ConvolutionFwdPreference {
	return ConvolutionFwdPreference(C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE)
}

//PreferFastest returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
func (c ConvolutionFwdPrefFlag) PreferFastest() ConvolutionFwdPreference {
	return ConvolutionFwdPreference(C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
}

//SpecifyWorkSpaceLimit returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
func (c ConvolutionFwdPrefFlag) SpecifyWorkSpaceLimit() ConvolutionFwdPreference {
	return ConvolutionFwdPreference(C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
}

/*
*
*
*       ConvFwdAlgoFlag
*
*
 */

//ConvFwdAlgo flags for cudnnConvFwdAlgo_t
type ConvFwdAlgo C.cudnnConvolutionFwdAlgo_t

//ConvFwdAlgoFlag transfer flags for ConvFwdAlgo through methods
type ConvFwdAlgoFlag struct {
}

//ImplicitGemm returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
func (c ConvFwdAlgoFlag) ImplicitGemm() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
}

//ImplicitPrecompGemm returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
func (c ConvFwdAlgoFlag) ImplicitPrecompGemm() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
}

//Gemm returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
func (c ConvFwdAlgoFlag) Gemm() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
}

//Direct returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
func (c ConvFwdAlgoFlag) Direct() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
}

//FFT returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_FFT)
func (c ConvFwdAlgoFlag) FFT() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT)
}

//FFTTiling returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
func (c ConvFwdAlgoFlag) FFTTiling() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
}

//WinoGrad  returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
func (c ConvFwdAlgoFlag) WinoGrad() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
}

//WinoGradNonFused   returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
func (c ConvFwdAlgoFlag) WinoGradNonFused() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
}

//Count    returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
func (c ConvFwdAlgoFlag) Count() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
}

func (a ConvFwdAlgo) c() C.cudnnConvolutionFwdAlgo_t {
	return C.cudnnConvolutionFwdAlgo_t(a)
}
