package gocudnn

/*
#include <cudnn.h>
*/
import "C"

import (
	"errors"
)

//ConvolutionMode is the type to describe the convolution mode flags
type ConvolutionMode C.cudnnConvolutionMode_t

//flags for convolution mode
const (
	Convolution      ConvolutionMode = C.CUDNN_CONVOLUTION
	CrossCorrelation ConvolutionMode = C.CUDNN_CROSS_CORRELATION
)

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

//NewConvolutionNdDescriptor creates and sets a new Convolution ND descriptor
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
