package gocudnn

/*

#include <cudnn.h>
*/
import "C"
import (
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

//Set sets pooling descriptor to values passed
func (p *PoolingD) Set(mode PoolingMode, nan NANProp, window, padding, stride []int32) error {
	cwindow := int32Tocint(window)
	cpadding := int32Tocint(padding)
	cstride := int32Tocint(stride)
	p.dims = (C.int)(len(window))
	return Status(C.cudnnSetPoolingNdDescriptor(
		p.descriptor,
		mode.c(),
		nan.c(),
		p.dims,
		&cwindow[0],
		&cpadding[0],
		&cstride[0],
	)).error("(*PoolingD)SetND")
}

//Get gets the descriptor values for pooling
func (p *PoolingD) Get() (mode PoolingMode, nan NANProp, window, padding, stride []int32, err error) {
	windowc := make([]C.int, p.dims)
	paddingc := make([]C.int, p.dims)
	stridec := make([]C.int, p.dims)
	var actual C.int
	err = Status(C.cudnnGetPoolingNdDescriptor(
		p.descriptor,
		p.dims,
		mode.cptr(),
		nan.cptr(),
		&actual,
		&windowc[0],
		&paddingc[0],
		&stridec[0],
	)).error("GetPoolingDescriptor-nd")
	window, padding, stride = cintToint32(windowc), cintToint32(paddingc), cintToint32(stridec)
	return mode, nan, window, padding, stride, err
}

//GetOutputDims will return the forward output dims from the pooling desc, and the tensor passed
func (p *PoolingD) GetOutputDims(
	input *TensorD,
) ([]int32, error) {
	outputdims := make([]C.int, input.dims)
	err := Status(C.cudnnGetPoolingNdForwardOutputDim(
		p.descriptor,
		input.descriptor,
		input.dims,
		&outputdims[0],
	)).error("GetPoolingForwardOutputDim-nd")

	return cintToint32(outputdims), err
}

//Destroy destroys the pooling descriptor.
//
//Right now gocudnn is handle by the go GC exclusivly, but sometime in the future
//user of package will be be able to toggle it.
func (p *PoolingD) Destroy() error {
	if setfinalizer || p.gogc {
		return nil
	}
	return destroypoolingdescriptor(p)
}
func destroypoolingdescriptor(p *PoolingD) error {
	return Status(C.cudnnDestroyPoolingDescriptor(p.descriptor)).error("DestroyDescriptor")
}

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//Forward does the poolingForward operation
func (p *PoolingD) Forward(
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

//Backward does the backward pooling operation
func (p *PoolingD) Backward(
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

/*
//Set2D sets pooling descriptor to 2d
func (p *PoolingD) Set2D(mode PoolingMode, nan NANProp, Hwindow, Wwindow, Vpadding, Hpadding, Vstride, Hstride int32) error {
	return Status(C.cudnnSetPooling2dDescriptor(
		p.descriptor,
		mode.c(),
		nan.c(),
		C.int(Hwindow),
		C.int(Wwindow),
		C.int(Vpadding),
		C.int(Hpadding),
		C.int(Vstride),
		C.int(Hstride),
	)).error("(*PoolingD)Set2Dex")
}

//Set2DEx sets up a 2D pooling descriptor but with the window padding and stride are in slices.
func (p *PoolingD) Set2DEx(mode PoolingMode, nan NANProp, window, padding, stride []int32) error {
	if len(window) != len(padding) || len(window) != len(stride) || len(window) != 2 {
		return errors.New("Window Padding and Stride array lengths need to be 2")
	}
	p.dims = (C.int)(len(window))
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
	p.dims = (C.int)(len(window))
	return Status(C.cudnnSetPoolingNdDescriptor(
		p.descriptor,
		mode.c(),
		nan.c(),
		p.dims,
		&cwindow[0],
		&cpadding[0],
		&cstride[0],
	)).error("(*PoolingD)SetND")
}
//Get2D get 2d gets the 2D descriptor values
func (p *PoolingD) Get2D() (mode PoolingMode, nan NANProp, Hwindow, Wwindow, Vpadding, Hpadding, Vstride, Hstride int32, err error) {
	var hw, ww, vp, hp, vs, hs C.int
	err = Status(C.cudnnGetPooling2dDescriptor(
		p.descriptor,
		mode.cptr(), nan.cptr(),
		&hw, &ww, &vp, &hp, &vs, &hs,
	)).error("GetPooling2dDescriptor-2d")

	Hwindow, Wwindow, Vpadding, Hpadding, Vstride, Hstride = (int32)(hw), (int32)(ww), (int32)(vp), (int32)(hp), (int32)(vs), (int32)(hs)
	return mode, nan, Hwindow, Wwindow, Vpadding, Hpadding, Vstride, Hstride, err
}

//Get2DEx gets the 2d descriptor values in ND mode.
func (p *PoolingD) Get2DEx() (mode PoolingMode, nan NANProp, window, padding, stride []int32, err error) {

	window = make([]int32, 2)
	padding = make([]int32, 2)
	stride = make([]int32, 2)
	hw, ww, vp, hp, vs, hs := (*C.int)(&window[0]), (*C.int)(&window[1]), (*C.int)(&padding[0]), (*C.int)(&padding[1]), (*C.int)(&stride[0]), (*C.int)(&stride[1])
	err = Status(C.cudnnGetPooling2dDescriptor(
		p.descriptor,
		mode.cptr(), nan.cptr(),
		hw, ww, vp, hp, vs, hs,
	)).error("GetPooling2dDescriptor-2d")

	return mode, nan, window, padding, stride, err
}

//GetND gets the nd descriptor for pooling
func (p *PoolingD) GetND() (mode PoolingMode, nan NANProp, window, padding, stride []int32, err error) {
	windowc := make([]C.int, p.dims)
	paddingc := make([]C.int, p.dims)
	stridec := make([]C.int, p.dims)
	var actual C.int
	err = Status(C.cudnnGetPoolingNdDescriptor(
		p.descriptor,
		p.dims,
		mode.cptr(),
		nan.cptr(),
		&actual,
		&windowc[0],
		&paddingc[0],
		&stridec[0],
	)).error("GetPoolingDescriptor-nd")
	window, padding, stride = cintToint32(windowc), cintToint32(paddingc), cintToint32(stridec)
	return mode, nan, window, padding, stride, err
}
//GetOutputDim will return the forward output dims from the pooling desc, and the tensor passed
func (p *PoolingD) GetOutputDim(
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

*/
