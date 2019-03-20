package gocudnn

/*

#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//PoolingD handles the pooling descriptor
type PoolingD struct {
	descriptor C.cudnnPoolingDescriptor_t
	dims       C.int
	gogc       bool
}

//CreatePoolingDescriptor creates a pooling descriptor.
func CreatePoolingDescriptor() (*PoolingD, error) {
	p := new(PoolingD)
	err := Status(C.cudnnCreatePoolingDescriptor(&p.descriptor)).error("NewPooling2dDescriptor-create")
	if setfinalizer || p.gogc {
		runtime.SetFinalizer(p, destroypoolingdescriptor)
	}
	return p, err

}

//Set2D sets pooling descriptor to 2d
func (p *PoolingD) Set2D(mode PoolingMode, nan NANProp, Hwindow, Wwindow, Vpadding, Hpadding, Vstride, Hstride int32) error {
	return Status(C.cudnnSetPooling2dDescriptor(
		p.descriptor,
		mode.c(),
		nan.c(),
		C.int(window[0]),
		C.int(window[1]),
		C.int(padding[0]),
		C.int(padding[1]),
		C.int(stride[0]),
		C.int(stride[1]),
	)).error("(*PoolingD)Set2Dex")
}

//Set2DEx sets up a 2D pooling descriptor but with the window padding and stride are in slices.
func (p *PoolingD) Set2DEx(mode PoolingMode, nan NANProp, window, padding, stride []int32) error {
	if len(window) != len(padding) || len(window) != len(stride) || len(window) != 2 {
		return nil, errors.New("Window Padding and Stride array lengths need to be 2")
	}
	return Status(C.cudnnSetPooling2dDescriptor(
		p.descriptor,
		mode.c(),
		nan.c(),
		C.int(window[0]),
		C.int(window[1]),
		C.int(padding[0]),
		C.int(padding[1]),
		C.int(stride[0]),
		C.int(stride[1]),
	)).error("(*PoolingD)Set2Dex")

}

//SetND sets pooling descriptor to values passed
func (p *PoolingD) SetND(mode PoolingMode, nan NANProp, window, padding, stride []int32) error {
	cwindow := int32Tocint(window)
	cpadding := int32Tocint(padding)
	cstride := int32Tocint(stride)
	return Status(C.cudnnSetPoolingNdDescriptor(
		p.descriptor,
		mode.c(),
		nan.c(),
		C.int(dims),
		&cwindow[0],
		&cpadding[0],
		&cstride[0],
	)).error("(*PoolingD)SetND")
}

func (p *PoolingD) Get2D() (mode PoolingMode, nan NANProp, Hwindow, Wwindow, Vpadding, Hpadding, Vstride, Hstride int32, err error) {
	var hw, ww, vp, hp, vs, hs C.int
	err = Status(C.cudnnGetPooling2dDescriptor(
		p.descriptor,
		mode.cptr(), nan.cptr(),
		&hw, &ww, &vp, &hp, &vs, &hs,
	)).error("GetPooling2dDescriptor-2d")
	(C.int)()
	Hwindow, Wwindow, Vpadding, Hpadding, Vstride, Hstride = (C.int)(hw), (C.int)(ww), (C.int)(vp), (C.int)(hp), (C.int)(vs), (C.int)(hs)
	return
}

//GetPoolingDescriptor returns the pooling descriptors and the error
func (p *PoolingD) GetPoolingDescriptor() (PoolingMode, NANProp, []int32, []int32, []int32, error) {
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
	alpha float64,
	xD *TensorD,
	x gocu.Mem,
	beta float64,
	yD *TensorD,
	y gocu.Mem,
) error {

	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnPoolingForward(
		handle.x,
		p.descriptor,
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		b.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("PoolingForward")
}

//PoolingBackward does the backward pooling operation
func (p *PoolingD) PoolingBackward(
	handle *Handle,
	alpha float64,
	yD *TensorD,
	y gocu.Mem,
	dyD *TensorD,
	dy gocu.Mem,
	xD *TensorD,
	x gocu.Mem,
	beta float64,
	dxD *TensorD,
	dx gocu.Mem,
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnPoolingBackward(handle.x, p.descriptor, a.CPtr(), yD.descriptor, y.Ptr(), dyD.descriptor, dy.Ptr(), xD.descriptor, x.Ptr(), b.CPtr(), dxD.descriptor, dx.Ptr())).error("PoolingBackward")
}

/*
 *  pooling mode
 */

//PoolingMode is used for flags in pooling
type PoolingMode C.cudnnPoolingMode_t

//Max returns PoolingMode(C.CUDNN_POOLING_MAX) flag
func (p *PoolingMode) Max() PoolingMode { *p = PoolingMode(C.CUDNN_POOLING_MAX); return *p }

//AverageCountIncludePadding returns PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) flag
func (p *PoolingMode) AverageCountIncludePadding() PoolingMode {
	*p = PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
	return *p
}

//AverageCountExcludePadding returns PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) flag
func (p *PoolingMode) AverageCountExcludePadding() PoolingMode {
	*p = PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
	return *p
}

//MaxDeterministic returns PoolingMode(C.CUDNN_POOLING_MAX_DETERMINISTIC) flag
func (p *PoolingMode) MaxDeterministic() PoolingMode {
	*p = PoolingMode(C.CUDNN_POOLING_MAX_DETERMINISTIC)
	return *p
}

func (p PoolingMode) c() C.cudnnPoolingMode_t      { return C.cudnnPoolingMode_t(p) }
func (p *PoolingMode) cptr() *C.cudnnPoolingMode_t { return (*C.cudnnPoolingMode_t)(p) }
