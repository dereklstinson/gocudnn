package gocudnn

/*
#include <cudnn.h>
*/
import "C"

import (
	"errors"
	"fmt"
)

//ConvolutionMode is the type to describe the convolution mode flags
type ConvolutionMode C.cudnnConvolutionMode_t

//flags for convolution mode
const (
	Convolution      ConvolutionMode = C.CUDNN_CONVOLUTION
	CrossCorrelation ConvolutionMode = C.CUDNN_CROSS_CORRELATION
)

//FilterD is the struct holding discriptor information for cudnnFilterDescriptor_t
type FilterD struct {
	descriptor C.cudnnFilterDescriptor_t
	dims       C.int
	flags      descflag
}

//CreateFilterDescriptor creates a filter distriptor
/*
 The Basic 4d shape is shape[0] = # of output feature maps
					   shape[1] = # of input feature maps
					   shape[2] = height of each filter
					   shape[3] = width of each input filter
*/

//NewFilter4dDescriptor Creates and sets an Filter 4d Descpripter
func NewFilter4dDescriptor(data DataType, format TensorFormat, shape []int32) (*FilterD, error) {
	if len(shape) != 4 {
		return nil, errors.New("length of shape != 4")
	}
	var descriptor C.cudnnFilterDescriptor_t
	err := Status(C.cudnnCreateFilterDescriptor(&descriptor)).error("NewFilter4dDescriptor-create")
	if err != nil {
		return nil, err
	}
	cshape := int32Tocint(shape)
	err = Status(C.cudnnSetFilter4dDescriptor(descriptor, C.cudnnDataType_t(data), C.cudnnTensorFormat_t(format), cshape[0], cshape[1], cshape[2], cshape[3])).error("NewFilter4dDescriptor-set")
	if err != nil {
		return nil, err
	}
	return &FilterD{descriptor: descriptor, dims: C.int(4), flags: t4d}, nil
}

//NewFilterNdDescriptor creates and sets an FilterNDDescriptor
func NewFilterNdDescriptor(data DataType, format TensorFormat, shape []int32) (*FilterD, error) {
	if len(shape) < 4 {
		return nil, errors.New("length of shape >= 4")
	}
	var descriptor C.cudnnFilterDescriptor_t
	err := Status(C.cudnnCreateFilterDescriptor(&descriptor)).error("NewFilter4dDescriptor-create")
	if err != nil {
		return nil, err
	}
	cshape := int32Tocint(shape)
	dims := C.int(len(shape))
	err = Status(C.cudnnSetFilterNdDescriptor(descriptor, C.cudnnDataType_t(data), C.cudnnTensorFormat_t(format), dims, &cshape[0])).error("NewFilter4dDescriptor-set")
	if err != nil {
		return nil, err
	}
	return &FilterD{descriptor: descriptor, dims: dims, flags: tnd}, nil
}

//GetDeiscriptor returns a copy of the ConvolutionD
func (f *FilterD) GetDeiscriptor() (DataType, TensorFormat, []int32, error) {
	var data C.cudnnDataType_t
	var format C.cudnnTensorFormat_t
	var err error
	shape := make([]C.int, f.dims)
	if f.flags == t4d {
		err = Status(C.cudnnGetFilter4dDescriptor(f.descriptor, &data, &format, &shape[0], &shape[1], &shape[2], &shape[3])).error("GetFilter4dDescriptor")
	} else if f.flags == tnd {
		var holder C.int
		err = Status(C.cudnnGetFilterNdDescriptor(f.descriptor, f.dims, &data, &format, &holder, &shape[0])).error("GetFilterNdDescriptor")
	} else {
		err = errors.New("Unsupported flag for descriptor")
	}

	return DataType(data), TensorFormat(format), cintToint32(shape), err
}

//DestroyFilterDescriptor Destroys Filter Descriptor
func (f *FilterD) DestroyDescriptor() error {
	return Status(C.cudnnDestroyFilterDescriptor(f.descriptor)).error("DestroyConvolutionDescriptor")
}

//ConvolutionD sets all the convolution info
type ConvolutionD struct {
	descriptor C.cudnnConvolutionDescriptor_t
	dims       C.int
	flag       descflag
}

//Pads is a convienence func
func Pads(pad ...int32) []int32 {
	return pad
}

//Dialation is a convienence func
func Dialation(dialation ...int32) []int32 {
	return dialation
}

//NewConvolution2dDescriptor creates and sets a 2d convolution descriptor
func NewConvolution2dDescriptor(mode ConvolutionMode, data DataType, pad, stride, dialation []int32) (*ConvolutionD, error) {
	if len(pad) != len(stride) || len(pad) != len(dialation) || len(pad) != 2 {
		return nil, errors.New("pad stride and dialation need to be size 2")
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

	err = Status(C.cudnnSetConvolution2dDescriptor(descriptor, cpad[0], cpad[1], cstride[0], cstride[1], cdialation[0], cdialation[1], cmode, cdata)).error("NewConvolution2dDescriptor-set")
	if err != nil {
		return nil, err
	}
	return &ConvolutionD{descriptor: descriptor, dims: 2, flag: t2d}, nil
}
func NewConvolutionNdDescriptor(mode ConvolutionMode, data DataType, pad, stride, dialation []int32) (*ConvolutionD, error) {
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

/*
func NewConvoltuion(mode ConvolutionMode, data DataType, pad, stride, dialation []int) (*ConvolutionD, error) {
	if len(pad) != len(stride) || len(stride) != len(dialation) {
		return nil, errors.New("not matching inputs")
	}
	if len(pad) < 2 {
		return nil, errors.New("pad,stride,dialation need to be at least 2 in length")
	}
}
*/

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

//DestroyConvolutionDescriptor destroys the ConvolutionDescriptor
func (c *ConvolutionD) DestroyConvolutionDescriptor() error {
	x := Status(C.cudnnDestroyConvolutionDescriptor(c.descriptor)).error("DestroyConvolutionDescriptor")
	return x
}

// ConvolutionFwdPreference used for flags
type ConvolutionFwdPreference C.cudnnConvolutionFwdPreference_t

/* helper function to provide the convolution algo that fit best the requirement */
//these are flags for ConvolutionFwdPreference
const (
	ConvolutionFwdNoWorkSpace           ConvolutionFwdPreference = C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
	ConvolutionFwdPreferFastest         ConvolutionFwdPreference = C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
	ConvolutionFwdSpecifyWorkspaceLimit ConvolutionFwdPreference = C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
) //cudnnConvolutionFwdPreference_t;

//ConvolutionFwdAlgo flags for cudnnConvolutionFwdAlgo_t
type ConvolutionFwdAlgo C.cudnnConvolutionFwdAlgo_t

//Flags used for algorithm
const (
	ConvolutionFwdAlgoImplicitGemm        ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
	ConvolutionFwdAlgoImplicitPrecompGemm ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
	ConvolutionFwdAlgoGemm                ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM
	ConvolutionFwdAlgoDirect              ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
	ConvolutionFwdAlgoFFT                 ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_FFT
	ConvolutionFwdAlgoFFTTiling           ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
	ConvolutionFwdAlgoWinoGrad            ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
	ConvolutionFwdAlgoWinoGradNonFused    ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
	ConvolutionFwdAlgoCount               ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT
) // cudnnConvolutionFwdAlgo_t;

func (a ConvolutionFwdAlgo) toString() string {
	var x string
	switch a {
	case ConvolutionFwdAlgoImplicitGemm:
		x = "Implicit Gemm"
	case ConvolutionFwdAlgoImplicitPrecompGemm:
		x = "Implicit Precomp Gemm"
	case ConvolutionFwdAlgoGemm:
		x = "Gemm"
	case ConvolutionFwdAlgoDirect:
		x = "Direct"
	case ConvolutionFwdAlgoFFT:
		x = "FFT"
	case ConvolutionFwdAlgoFFTTiling:
		x = "FFT Tiling"
	case ConvolutionFwdAlgoWinoGrad:
		x = "WinoGrad"
	case ConvolutionFwdAlgoWinoGradNonFused:
		x = "WinoGradNonFused"
	case ConvolutionFwdAlgoCount:
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
func (handle *Handle) FindConvolutionForwardAlgorithm(x *TensorD, w *FilterD, c *ConvolutionD, y *TensorD, requestedAlgoCount int32) ([]ConvolutionFwdAlgoPerformance, error) {
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
func (handle *Handle) FindConvolutionForwardAlgorithmEx(xDesc *TensorD, xMem Memer, wDesc *FilterD, wMem Memer, conDesc *ConvolutionD, yDesc *TensorD, yMem Memer, reqAlgoCount int32, wspace Memer, wspacebytes SizeT) ([]ConvolutionFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionForwardAlgorithmEx(handle.x, xDesc.descriptor, xMem.Ptr(), wDesc.descriptor, wMem.Ptr(), conDesc.descriptor, yDesc.descriptor, yMem.Ptr(), C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), C.size_t(wspacebytes))).error("FindConvolutionForwardAlgorithmEx")

	results := make([]ConvolutionFwdAlgoPerformance, int32(actualalgocount))
	for i := int32(0); i < int32(actualalgocount); i++ {
		results[i] = convertConvolutionFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//GetConvolutionForwardAlgorithm gives a good algo with the limits given to it
func (handle *Handle) GetConvolutionForwardAlgorithm(xDesc *TensorD, wDesc *FilterD, convDesc *ConvolutionD, yDesc *TensorD, pref ConvolutionFwdPreference, wsmemlimit SizeT) (ConvolutionFwdAlgo, error) {
	var algo C.cudnnConvolutionFwdAlgo_t
	err := Status(C.cudnnGetConvolutionForwardAlgorithm(handle.x, xDesc.descriptor, wDesc.descriptor, convDesc.descriptor, yDesc.descriptor, C.cudnnConvolutionFwdPreference_t(pref), C.size_t(wsmemlimit), &algo)).error("GetConvolutionForwardAlgorithm")
	return ConvolutionFwdAlgo(algo), err
}

/*

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                cudnnConvolutionFwdPreference_t     preference,
                                size_t                              memoryLimitInBytes,
                                cudnnConvolutionFwdAlgo_t          *algo );


cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(
                                cudnnHandle_t                      handle,
                                const cudnnTensorDescriptor_t      srcDesc,
                                const cudnnFilterDescriptor_t      filterDesc,
                                const cudnnConvolutionDescriptor_t convDesc,
                                const cudnnTensorDescriptor_t      destDesc,
                                const int                          requestedAlgoCount,
                                int                               *returnedAlgoCount,
                                cudnnConvolutionFwdAlgoPerf_t     *perfResults);
*/
