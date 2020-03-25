package gocudnn

/*

#include <cudnn.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/dereklstinson/cutil"
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
	)).error("(p *PoolingD) Set")
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
	)).error("(p *PoolingD) Get()")
	window, padding, stride = cintToint32(windowc), cintToint32(paddingc), cintToint32(stridec)
	return mode, nan, window, padding, stride, err
}
func (p *PoolingD) String() string {
	mode, nan, w, pad, s, err := p.Get()
	if err != nil {
		return fmt.Sprintf("PoolingD{\nError: %v\n}\n", err)
	}

	return fmt.Sprintf("PoolingD{\n%v,\n%v,\nWindow: %v,\nPadding: %v,\nStride: %v,\n}\n", mode, nan, w, pad, s)
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
	)).error("(p *PoolingD) GetOutputDims")

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
	xD *TensorD, x cutil.Mem,
	beta float64,
	yD *TensorD, y cutil.Mem,
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnPoolingForward(
				handle.x,
				p.descriptor,
				a.CPtr(),
				xD.descriptor,
				x.Ptr(),
				b.CPtr(),
				yD.descriptor,
				y.Ptr(),
			)).error("(p *PoolingD) Forward")
		})
	}
	return Status(C.cudnnPoolingForward(
		handle.x,
		p.descriptor,
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		b.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("(p *PoolingD) Forward")
}

//ForwardUS is like Forward but uses unsafe.Pointer instead of cutil.Mem
func (p *PoolingD) ForwardUS(
	handle *Handle,
	alpha float64,
	xD *TensorD, x unsafe.Pointer,
	beta float64,
	yD *TensorD, y unsafe.Pointer,
) error {

	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnPoolingForward(
				handle.x,
				p.descriptor,
				a.CPtr(),
				xD.descriptor, x,
				b.CPtr(),
				yD.descriptor, y,
			)).error("(p *PoolingD) ForwardUS")
		})
	}
	return Status(C.cudnnPoolingForward(
		handle.x,
		p.descriptor,
		a.CPtr(),
		xD.descriptor, x,
		b.CPtr(),
		yD.descriptor, y,
	)).error("(p *PoolingD) ForwardUS")
}

//Backward does the backward pooling operation
func (p *PoolingD) Backward(
	handle *Handle,
	alpha float64,
	yD *TensorD, y cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	xD *TensorD, x cutil.Mem,
	beta float64,
	dxD *TensorD, dx cutil.Mem,
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnPoolingBackward(handle.x,
				p.descriptor,
				a.CPtr(),
				yD.descriptor, y.Ptr(),
				dyD.descriptor, dy.Ptr(),
				xD.descriptor, x.Ptr(),
				b.CPtr(),
				dxD.descriptor, dx.Ptr())).error(" (p *PoolingD) Backward")

		})
	}
	return Status(C.cudnnPoolingBackward(handle.x,
		p.descriptor,
		a.CPtr(),
		yD.descriptor, y.Ptr(),
		dyD.descriptor, dy.Ptr(),
		xD.descriptor, x.Ptr(),
		b.CPtr(),
		dxD.descriptor, dx.Ptr())).error(" (p *PoolingD) Backward")
}

//BackwardUS is like Backward but uses unsafe.Pointer instead of cutil.Mem
func (p *PoolingD) BackwardUS(
	handle *Handle,
	alpha float64,
	yD *TensorD, y unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	xD *TensorD, x unsafe.Pointer,
	beta float64,
	dxD *TensorD, dx unsafe.Pointer,
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnPoolingBackward(handle.x,
				p.descriptor,
				a.CPtr(),
				yD.descriptor, y,
				dyD.descriptor, dy,
				xD.descriptor, x,
				b.CPtr(),
				dxD.descriptor, dx)).error("p *PoolingD) BackwardUS")
		})
	}
	return Status(C.cudnnPoolingBackward(handle.x,
		p.descriptor,
		a.CPtr(),
		yD.descriptor, y,
		dyD.descriptor, dy,
		xD.descriptor, x,
		b.CPtr(),
		dxD.descriptor, dx)).error("p *PoolingD) BackwardUS")
}

/*
 *  pooling mode
 */

//PoolingMode is used for flags in pooling
type PoolingMode C.cudnnPoolingMode_t

//Max returns PoolingMode(C.CUDNN_POOLING_MAX) flag
//
//The maximum value inside the pooling window is used.
func (p *PoolingMode) Max() PoolingMode { *p = PoolingMode(C.CUDNN_POOLING_MAX); return *p }

//AverageCountIncludePadding returns PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) flag
//
//Values inside the pooling window are averaged.
//The number of elements used to calculate the average
//includes spatial locations falling in the padding region.
func (p *PoolingMode) AverageCountIncludePadding() PoolingMode {
	*p = PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
	return *p
}

//AverageCountExcludePadding returns PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) flag
//
//Values inside the pooling window are averaged.
//The number of elements used to calculate the average
//excludes spatial locations falling in the padding region.
func (p *PoolingMode) AverageCountExcludePadding() PoolingMode {
	*p = PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
	return *p
}

//MaxDeterministic returns PoolingMode(C.CUDNN_POOLING_MAX_DETERMINISTIC) flag
//
//The maximum value inside the pooling window is used.
//The algorithm used is deterministic.
func (p *PoolingMode) MaxDeterministic() PoolingMode {
	*p = PoolingMode(C.CUDNN_POOLING_MAX_DETERMINISTIC)
	return *p
}

func (p PoolingMode) c() C.cudnnPoolingMode_t      { return C.cudnnPoolingMode_t(p) }
func (p *PoolingMode) cptr() *C.cudnnPoolingMode_t { return (*C.cudnnPoolingMode_t)(p) }

func (p PoolingMode) String() string {
	var x string
	f := p
	switch p {
	case f.AverageCountExcludePadding():
		x = "AverageCountExcludePadding"
	case f.AverageCountIncludePadding():
		x = "AverageCountIncludePadding"
	case f.Max():
		x = "Max"
	case f.MaxDeterministic():
		x = "MaxDeterministic"
	default:
		x = "Unsupported Flag"
	}
	return "PoolingMode" + x
}
