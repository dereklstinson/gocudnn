package gocudnn

/*

#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"runtime"
)

//Pooling is used to hold flags and funcs for pooling operations
type Pooling struct {
	PFlg PoolingModeFlag
	NFlg PropagationNANFlag
}

//PoolingD handles the pooling descriptor
type PoolingD struct {
	descriptor C.cudnnPoolingDescriptor_t
	dims       C.int
}

//NewPooling2dDescriptor creates and sets a pooling 2d Descriptor
func (pool Pooling) NewPooling2dDescriptor(
	mode PoolingMode,
	nan PropagationNAN,
	window []int32, // height and wideth
	padding []int32, //height and width
	stride []int32, //height and width
) (descriptor *PoolingD, err error) {
	if len(window) != len(padding) || len(window) != len(stride) || len(window) != 2 {
		return nil, errors.New("Window Padding and Stride array lengths need to be 2")
	}
	var desc C.cudnnPoolingDescriptor_t
	err = Status(C.cudnnCreatePoolingDescriptor(&desc)).error("NewPooling2dDescriptor-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetPooling2dDescriptor(
		desc,
		mode.c(),
		nan.c(),
		C.int(window[0]),
		C.int(window[1]),
		C.int(padding[0]),
		C.int(padding[1]),
		C.int(stride[0]),
		C.int(stride[1]),
	)).error("NewPooling2dDescriptor-set")
	if err != nil {
		return nil, err
	}
	descriptor = &PoolingD{descriptor: desc, dims: C.int(2)}
	if setfinalizer {
		runtime.SetFinalizer(descriptor, destroypoolingdescriptor)
	}
	return descriptor, nil
}
func (p *PoolingD) keepsalive() {
	runtime.KeepAlive(p)
}

//GetPoolingDescriptor returns the pooling descriptors and the error
func (p *PoolingD) GetPoolingDescriptor() (PoolingMode, PropagationNAN, []int32, []int32, []int32, error) {
	var mode C.cudnnPoolingMode_t
	var nan C.cudnnNanPropagation_t
	window := make([]C.int, p.dims)
	padding := make([]C.int, p.dims)
	stride := make([]C.int, p.dims)
	if p.dims == C.int(2) {
		err := Status(C.cudnnGetPooling2dDescriptor(
			p.descriptor,
			&mode,
			&nan,
			&window[0],
			&window[1],
			&padding[0],
			&padding[1],
			&stride[0],
			&stride[1],
		)).error("GetPooling2dDescriptor-2d")
		if setkeepalive {
			p.keepsalive()
		}
		return PoolingMode(mode), PropagationNAN(nan), cintToint32(window), cintToint32(padding), cintToint32(stride), err
	}
	var actual C.int
	err := Status(C.cudnnGetPoolingNdDescriptor(
		p.descriptor,
		p.dims,
		&mode,
		&nan,
		&actual,
		&window[0],
		&padding[0],
		&stride[0],
	)).error("GetPoolingDescriptor-nd")
	if setkeepalive {
		p.keepsalive()
	}
	return PoolingMode(mode), PropagationNAN(nan), cintToint32(window), cintToint32(padding), cintToint32(stride), err
}

//CreatePoolingNdDescriptor Creates and sets a pooling nd descriptor
func (pool Pooling) CreatePoolingNdDescriptor(
	mode PoolingMode,
	nan PropagationNAN,
	dims int32,
	window []int32, // height and wideth
	padding []int32, //height and width
	stride []int32, //height and width
) (descriptor *PoolingD, err error) {
	if len(window) != len(padding) || len(window) != len(stride) || len(window) < 3 {
		return nil, errors.New("Window Padding and Stride array lengths need to be equal and 3 or greater")
	}
	var desc C.cudnnPoolingDescriptor_t
	err = Status(C.cudnnCreatePoolingDescriptor(&desc)).error("NewPoolingNdDescriptor-create")
	if err != nil {
		return nil, err
	}
	cwindow := int32Tocint(window)
	cpadding := int32Tocint(padding)
	cstride := int32Tocint(stride)
	err = Status(C.cudnnSetPoolingNdDescriptor(
		desc,
		mode.c(),
		nan.c(),
		C.int(dims),
		&cwindow[0],
		&cpadding[0],
		&cstride[0],
	)).error("NewPoolingNdDescriptor-set")
	if err != nil {
		return nil, err
	}
	descriptor = &PoolingD{
		descriptor: desc,
		dims:       C.int(dims),
	}
	if setfinalizer {
		runtime.SetFinalizer(descriptor, destroypoolingdescriptor)
	}
	return descriptor, nil
}

//GetPoolingForwardOutputDim will return the forward output dims from the pooling desc, and the tensor passed
func (p *PoolingD) GetPoolingForwardOutputDim(
	input *TensorD,
) ([]int32, error) {
	/*
		if input.dims != p.dims {
			inp := strconv.Itoa(int(input.dims))
			pop := strconv.Itoa(int(p.dims))
			return nil, errors.New("input dims != pooling dims" + " input dims: " + inp + " pooling dims: " + pop)
		}
	*/

	if p.dims > 2 {
		outputdims := make([]C.int, p.dims)
		err := Status(C.cudnnGetPoolingNdForwardOutputDim(
			p.descriptor,
			input.descriptor,
			p.dims,
			&outputdims[0],
		)).error("GetPoolingForwardOutputDim-nd")
		if setkeepalive {
			keepsalivebuffer(p, input)
		}
		return cintToint32(outputdims), err
	}
	outputdims := make([]C.int, 4)
	err := Status(C.cudnnGetPooling2dForwardOutputDim(
		p.descriptor,
		input.descriptor,
		&outputdims[0],
		&outputdims[1],
		&outputdims[2],
		&outputdims[3],
	)).error("GetPoolingForwardOutputDim-2d")
	if setkeepalive {
		keepsalivebuffer(p, input)
	}
	return cintToint32(outputdims), err
}

//DestroyDescriptor destroys the pooling descriptor
func (p *PoolingD) DestroyDescriptor() error {
	return destroypoolingdescriptor(p)
}
func destroypoolingdescriptor(p *PoolingD) error {
	return Status(C.cudnnDestroyPoolingDescriptor(p.descriptor)).error("DestroyDescriptor")
}

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//PoolingForward does the poolingForward operation
func (p *PoolingD) PoolingForward(
	handle *Handle,
	alpha CScalar,
	xD *TensorD,
	x *Malloced,
	beta CScalar,
	yD *TensorD,
	y *Malloced,
) error {
	if setkeepalive {
		keepsalivebuffer(p, handle, xD, x, yD, y)
	}
	return Status(C.cudnnPoolingForward(
		handle.x,
		p.descriptor,
		alpha.CPtr(),
		xD.descriptor,
		x.Ptr(),
		beta.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("PoolingForward")
}

//PoolingBackward does the backward pooling operation
func (p *PoolingD) PoolingBackward(
	handle *Handle,
	alpha CScalar,
	yD *TensorD,
	y *Malloced,
	dyD *TensorD,
	dy *Malloced,
	xD *TensorD,
	x *Malloced,
	beta CScalar,
	dxD *TensorD,
	dx *Malloced,
) error {
	if setkeepalive {
		keepsalivebuffer(p, handle, xD, x, yD, y, dyD, dy, dxD, dx)
	}
	return Status(C.cudnnPoolingBackward(
		handle.x,
		p.descriptor,
		alpha.CPtr(),
		yD.descriptor,
		y.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		xD.descriptor,
		x.Ptr(),
		beta.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
	)).error("PoolingBackward")
}

/*
 *  pooling mode
 */

//PoolingModeFlag is used to pass PoolingMode flags for human users semi-safely using methods
type PoolingModeFlag struct {
}

//PoolingMode is used for flags in pooling
type PoolingMode C.cudnnPoolingMode_t

//Max returns PoolingMode(C.CUDNN_POOLING_MAX) flag
func (p PoolingModeFlag) Max() PoolingMode {
	return PoolingMode(C.CUDNN_POOLING_MAX)
}

//AverageCountIncludePadding returns PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) flag
func (p PoolingModeFlag) AverageCountIncludePadding() PoolingMode {
	return PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
}

//AverageCountExcludePadding returns PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) flag
func (p PoolingModeFlag) AverageCountExcludePadding() PoolingMode {
	return PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
}

//MaxDeterministic returns PoolingMode(C.CUDNN_POOLING_MAX_DETERMINISTIC) flag
func (p PoolingModeFlag) MaxDeterministic() PoolingMode {
	return PoolingMode(C.CUDNN_POOLING_MAX_DETERMINISTIC)
}

func (p PoolingMode) c() C.cudnnPoolingMode_t { return C.cudnnPoolingMode_t(p) }
