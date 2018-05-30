package cudnn

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
	datatype   C.cudnnDataType_t
	format     C.cudnnTensorFormat_t
	shape      []C.int
	dims       C.int
}

//CreateFilterDescriptor creates a filter distriptor
/*
 The Basic 4d shape is shape[0] = # of output feature maps
					   shape[1] = # of input feature maps
					   shape[2] = height of each filter
					   shape[3] = width of each input filter
*/
func CreateFilterDescriptor(data DataType, format TensorFormat, shape []int) (FilterD, error) {
	var x error
	var filter FilterD
	if len(shape) < 4 || len(shape) > DimMax {
		return filter, errors.New("shape is less than 4 or greater than DimMax")
	}
	filter.datatype = C.cudnnDataType_t(data)
	filter.shape = intTocint(shape)
	filter.format = C.cudnnTensorFormat_t(format)
	filter.dims = C.int(len(filter.shape))
	x = Status(C.cudnnCreateFilterDescriptor(&filter.descriptor)).error("CreateFilterDescriptor")
	if x != nil {
		return filter, x
	}
	if len(shape) == 4 {
		x = filter.setFilter4dDescriptor()
	} else {
		x = filter.setFilterNdDescriptor()
	}
	return filter, x
}

//SetFilter4dDescriptor sets the info passed in the create filter descriptor
func (f *FilterD) setFilter4dDescriptor() error {
	x := Status(C.cudnnSetFilter4dDescriptor(f.descriptor, f.datatype, f.format, f.shape[0], f.shape[1], f.shape[2], f.shape[3]))
	return x.error("SetFilter4dDescriptor")
}

//DataType returns the Datatype flag
func (f *FilterD) DataType() DataType { return DataType(f.datatype) }

//TensorFormat returns the tensor format for FilterD
func (f *FilterD) TensorFormat() TensorFormat { return TensorFormat(f.format) }

//Shape returns the shape arrayh of the filterD
func (f *FilterD) Shape() []int { return cintToint(f.shape) }

/*
//GetFilter4dDescriptor returns a copy of the ConvolutionD
func (f *FilterD) GetFilter4dDescriptor() (FilterD, error) {
	var filter FilterD
	filter.descriptor = f.descriptor
	filter.shape = make([]C.int, len(f.shape))
	x := Status(C.cudnnGetFilter4dDescriptor(filter.descriptor, &filter.datatype, &filter.format, &filter.shape[0], &filter.shape[1], &filter.shape[2], &filter.shape[3]))
	return filter, x.error("GetFilter4dDescriptor")
}
*/

//SetFilterNdDescriptor sets the FilterNdDescriptor
func (f *FilterD) setFilterNdDescriptor() error {
	x := Status(C.cudnnSetFilterNdDescriptor(f.descriptor, f.datatype, f.format, f.dims, &f.shape[0]))
	return x.error("SetFilter4dDescriptor")
}

/*
//GetFilterNdDescriptor returns a copy of the ConvolutionD
func (f *FilterD) GetFilterNdDescriptor() (FilterD, error) {
	var filter FilterD
	dimsreq := C.int(len(f.shape))
	filter.descriptor = f.descriptor
	x := Status(C.cudnnGetFilterNdDescriptor(filter.descriptor, dimsreq, &filter.datatype, &filter.format, &filter.dims, &filter.shape[0]))
	return filter, x.error("GetFilterNdDescriptor")
}
*/

//DestroyFilterDescriptor Destroys Filter Descriptor
func (f *FilterD) DestroyFilterDescriptor() error {
	return Status(C.cudnnDestroyFilterDescriptor(f.descriptor)).error("DestroyConvolutionDescriptor")
}

//ConvolutionD sets all the convolution info
type ConvolutionD struct {
	descriptor C.cudnnConvolutionDescriptor_t
	mathtype   C.cudnnMathType_t
	mode       C.cudnnConvolutionMode_t
	data       C.cudnnDataType_t
	groupCount C.int
	pad        []C.int
	stride     []C.int
	dialation  []C.int
	dims       C.int
}

//Pads is a convienence func
func Pads(pad ...int) []int {
	return pad
}

//Dialation is a convienence func
func Dialation(dialation ...int) []int {
	return dialation
}

//MathType returns convolutionD's mathtype
func (c *ConvolutionD) MathType() MathType { return MathType(c.mathtype) }

//GroupCount returns convoltionD's GroupCount
func (c *ConvolutionD) GroupCount() int { return int(c.groupCount) }

//ConvolutionMode returns the convolutionMode
func (c *ConvolutionD) ConvolutionMode() ConvolutionMode { return ConvolutionMode(c.mode) }

//DataType returns the ConvolutionD datatype
func (c *ConvolutionD) DataType() DataType { return DataType(c.data) }

//Pads returns the padding array
func (c *ConvolutionD) Pads() []int { return cintToint(c.pad) }

//Strides returns the stride array
func (c *ConvolutionD) Strides() []int { return cintToint(c.stride) }

//Dialations returns the dialation array
func (c *ConvolutionD) Dialations() []int { return cintToint(c.dialation) }

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
//CreateConvolutionDescriptor Creats a ConvolutionD struct
func CreateConvolutionDescriptor(mode ConvolutionMode,
	data DataType, pad, stride, dialation []int) (ConvolutionD, error) {

	var c ConvolutionD
	if len(pad) != len(stride) || len(pad) != len(dialation) {
		return c, errors.New("pad,stride,dialation dims not same size")
	}
	if len(pad) < 2 || len(pad) > DimMax || len(stride) < 2 || len(stride) > DimMax || len(dialation) < 2 || len(dialation) > DimMax {
		return c, errors.New("length of pad || stride ||dialation arrays not correrct sizes")
	}
	c.data = C.cudnnDataType_t(data)
	c.pad = intTocint(pad)
	c.stride = intTocint(stride)
	c.dialation = intTocint(dialation)

	c.mode = C.cudnnConvolutionMode_t(mode)
	c.dims = C.int(len(stride))
	x := Status(C.cudnnCreateConvolutionDescriptor(&c.descriptor)).error("CreateConvolutionDescriptor,cudnnCreateConvolutionDescriptor")
	if x != nil {
		return c, x
	}
	if len(pad) == 2 {
		x = c.setConvolution2dDescriptor()
		return c, x
	}
	x = c.setConvolutionNdDescriptor()
	return c, x

}

//SetGroupCount sets the Group Count
func (c *ConvolutionD) SetGroupCount(groupCount int) error {

	c.groupCount = C.int(groupCount)
	err := Status(C.cudnnSetConvolutionGroupCount(c.descriptor, c.groupCount)).error("SetGroupCountandMathtype-Group")
	return err

}

//SetMathType sets the mathtype
func (c *ConvolutionD) SetMathType(mathtype MathType) error {
	c.mathtype = C.cudnnMathType_t(mathtype)
	x := Status(C.cudnnSetConvolutionMathType(c.descriptor, c.mathtype))
	return x.error("SetGroupCountandMathtype-Math")
}

//SetConvolution2dDescriptor sets the 2dDescriptor
func (c *ConvolutionD) setConvolution2dDescriptor() error {

	x := Status(C.cudnnSetConvolution2dDescriptor(c.descriptor, c.pad[0], c.pad[1], c.stride[0], c.stride[1], c.dialation[0], c.dialation[1], c.mode, c.data))
	return x.error("SetConvolution2dDescriptor")
}

//GetConvolution2dForwardOutputDim is a helper func that will output the shape of the convolution
func (c *ConvolutionD) GetConvolution2dForwardOutputDim(input *TensorD, filter *FilterD) ([]int, error) {
	var shape [4]C.int
	x := Status(C.cudnnGetConvolution2dForwardOutputDim(c.descriptor, input.descriptor, filter.descriptor,
		&shape[0], &shape[1], &shape[2], &shape[3]))
	retshap := make([]int, 4)
	for i := 0; i < 4; i++ {
		retshap[i] = int(shape[i])
	}
	return retshap, x.error("GetConvolution2dForwardOutputDim")

}
func (c *ConvolutionD) setConvolutionNdDescriptor() error {
	x := Status(C.cudnnSetConvolutionNdDescriptor(c.descriptor, c.dims, &c.pad[0], &c.stride[0], &c.dialation[0], c.mode, c.data)).error("error")
	return x
}

//GetConvolutionNdForwardOutputDim is a helper function to give the size of the output of of a COnvolutionNDForward
func (c *ConvolutionD) GetConvolutionNdForwardOutputDim(input *TensorD, filter *FilterD) ([]int, error) {
	dims := make([]C.int, int(c.dims))
	x := Status(C.cudnnGetConvolutionNdForwardOutputDim(c.descriptor, input.descriptor, filter.descriptor, c.dims, &dims[0])).error("GetConvolutionNdForwardOutputDim")

	return cintToint(dims), x
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
	holder[1] = algoPerf.Status.GetErrorString()
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
	Status      Status
	Time        float32
	Memory      uint64
	Determinism Determinism
	Mathtype    MathType
	Reserved    [3]int
}

func convertConvolutionFwdAlgoPerformance(input C.cudnnConvolutionFwdAlgoPerf_t) ConvolutionFwdAlgoPerformance {
	var x ConvolutionFwdAlgoPerformance
	x.Algo = ConvolutionFwdAlgo(input.algo)
	x.Status = Status(input.status)
	x.Time = float32(input.time)
	x.Memory = uint64(input.memory)
	x.Determinism = Determinism(input.determinism)
	x.Mathtype = MathType(input.mathType)
	for i := 0; i < 3; i++ {
		x.Reserved[i] = int(input.reserved[i])
	}
	return x
}

//GetConvolutionForwardAlgorithmMaxCount returns the max number of algos
func (handle *Handle) GetConvolutionForwardAlgorithmMaxCount() (int, error) {
	var count C.int
	x := Status(C.cudnnGetConvolutionForwardAlgorithmMaxCount(handle.x, &count)).error("GetConvolutionForwardAlgorithmMaxCount")
	return int(count), x

}

//FindConvolutionForwardAlgorithm will find the top performing algoriths and return the best algorithms in accending order they are limited to the number passed in requestedAlgoCount.
//So if 4 is passed through in requestedAlgoCount, then it will return the top 4 performers in the ConvolutionFwdAlgoPerformance struct.  using this could possible give the user cheat level performance :-)
func (handle *Handle) FindConvolutionForwardAlgorithm(x *TensorD, w *FilterD, c *ConvolutionD, y *TensorD, requestedAlgoCount int) ([]ConvolutionFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionForwardAlgorithm(handle.x, x.descriptor, w.descriptor, c.descriptor, y.descriptor, C.int(requestedAlgoCount), &actualalgocount, &perfResults[0])).error("FindConvolutionForwardAlgorithm")
	results := make([]ConvolutionFwdAlgoPerformance, int(actualalgocount))
	for i := 0; i < int(actualalgocount); i++ {
		results[i] = convertConvolutionFwdAlgoPerformance(perfResults[i])

	}
	return results, err
}

//FindConvolutionForwardAlgorithmEx finds some algorithms with memory
func (handle *Handle) FindConvolutionForwardAlgorithmEx(xDesc *TensorD, xMem Memer, wDesc *FilterD, wMem Memer, conDesc *ConvolutionD, yDesc *TensorD, yMem Memer, reqAlgoCount int, wspace Memer, wspacebytes SizeT) ([]ConvolutionFwdAlgoPerformance, error) {
	perfResults := make([]C.cudnnConvolutionFwdAlgoPerf_t, reqAlgoCount)
	var actualalgocount C.int
	err := Status(C.cudnnFindConvolutionForwardAlgorithmEx(handle.x, xDesc.descriptor, xMem.Ptr(), wDesc.descriptor, wMem.Ptr(), conDesc.descriptor, yDesc.descriptor, yMem.Ptr(), C.int(reqAlgoCount), &actualalgocount, &perfResults[0], wspace.Ptr(), C.size_t(wspacebytes))).error("FindConvolutionForwardAlgorithmEx")

	results := make([]ConvolutionFwdAlgoPerformance, int(actualalgocount))
	for i := 0; i < int(actualalgocount); i++ {
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
