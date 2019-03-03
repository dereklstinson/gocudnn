package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//Spatial hods the funcs and flags of Spatial Stuff
type Spatial struct {
	Funcs SpatialFuncs
	Flgs  SamplerTypeFlag
}

//SpatialFuncs is a struct used to call Spatial functions as methods
type SpatialFuncs struct {
}

//SpatialTfGridGeneratorForward This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
func (st *SpatialTransformerD) SpatialTfGridGeneratorForward(
	handle *Handle,
	theta gocu.Mem, //Input. Affine transformation matrix. It should be of size n*2*3 for a 2d transformation, n is the number of images.
	grid gocu.Mem, /*Output. A grid of coordinates. It is of size n*h*w*2 for a 2d transformation, where n,
	h, w is specified in stDesc . In the 4th dimension, the first coordinate is x, and the
	second coordinate is y*/

) error {
	if setkeepalive {
		keepsalivebuffer(st, handle, grid, theta)
	}
	return Status(C.cudnnSpatialTfGridGeneratorForward(
		handle.x,
		st.descriptor,
		theta.Ptr(),
		grid.Ptr(),
	)).error("SpatialTfGridGeneratorForward")
}

//SpatialTfGridGeneratorBackward - This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
func (st *SpatialTransformerD) SpatialTfGridGeneratorBackward(
	handle *Handle,
	grid gocu.Mem,
	theta gocu.Mem,
) error {
	if setkeepalive {
		keepsalivebuffer(st, handle, grid, theta)
	}

	return Status(C.cudnnSpatialTfGridGeneratorBackward(
		handle.x,
		st.descriptor,
		grid.Ptr(),
		theta.Ptr(),
	)).error("SpatialTfGridGeneratorBackward")
}

//SpatialTfSamplerForward performs the spatialtfsampleforward
func (st *SpatialTransformerD) SpatialTfSamplerForward(
	handle *Handle,
	alpha float64,
	xD *TensorD,
	x gocu.Mem,
	grid gocu.Mem,
	beta float64,
	yD *TensorD,
	y gocu.Mem,
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)

	return Status(C.cudnnSpatialTfSamplerForward(
		handle.x,
		st.descriptor,
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		grid.Ptr(),
		b.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("SpatialTfSamplerForward")
}

//SpatialTfSamplerBackward does the spatial Tranform Sample Backward
func (st *SpatialTransformerD) SpatialTfSamplerBackward(
	handle *Handle,
	alpha float64,
	xD *TensorD,
	x gocu.Mem,
	beta float64,
	dxD *TensorD,
	dx gocu.Mem,
	alphaDgrid float64,
	dyD *TensorD,
	dy gocu.Mem,
	grid gocu.Mem,
	betaDgrid float64,
	dGrid gocu.Mem,

) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dxD.dtype, beta)
	ad := cscalarbydatatype(xD.dtype, alphaDgrid)
	bd := cscalarbydatatype(dxD.dtype, betaDgrid)
	return Status(C.cudnnSpatialTfSamplerBackward(
		handle.x,
		st.descriptor,
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		b.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
		ad.CPtr(),
		dyD.descriptor,
		dy.Ptr(),
		grid.Ptr(),
		bd.CPtr(),
		dGrid.Ptr(),
	)).error("SpatialTfSamplerBackward")
}

/* APIs for spatial transformer network*/

//SamplerTypeFlag is used to pass the Bilinear flag as a method
type SamplerTypeFlag struct {
}

//Bilinear returns SamplerType(C.CUDNN_SAMPLER_BILINEAR)
func (s SamplerTypeFlag) Bilinear() SamplerType { return SamplerType(C.CUDNN_SAMPLER_BILINEAR) }

//SamplerType is used for flags
type SamplerType C.cudnnSamplerType_t

func (s SamplerType) c() C.cudnnSamplerType_t { return C.cudnnSamplerType_t(s) }

//SpatialTransformerD holdes the spatial descriptor
type SpatialTransformerD struct {
	descriptor C.cudnnSpatialTransformerDescriptor_t
	dims       C.int
}

//NewSpatialTransformerNdDescriptor creates and sets SpatialTransformerD
func (sp Spatial) NewSpatialTransformerNdDescriptor(
	sampler SamplerType,
	data DataType,
	dimA []int32,
) (descriptor *SpatialTransformerD, err error) {
	var desc C.cudnnSpatialTransformerDescriptor_t
	err = Status(C.cudnnCreateSpatialTransformerDescriptor(&desc)).error("NewSpatialTransformerNdDescriptor-create")
	if err != nil {
		return nil, err
	}
	dims := C.int(len(dimA))
	cdimA := int32Tocint(dimA)
	err = Status(C.cudnnSetSpatialTransformerNdDescriptor(
		desc,
		sampler.c(),
		data.c(),
		dims,
		&cdimA[0],
	)).error("NewSpatialTransformerNdDescriptor-Set")
	if err != nil {
		return nil, err
	}
	descriptor = &SpatialTransformerD{
		descriptor: desc,
		dims:       dims,
	}
	if setfinalizer {
		runtime.SetFinalizer(descriptor, destroyspatialtransformdescriptor)
	}

	return descriptor, nil
}
func (st *SpatialTransformerD) keepsalive() {
	runtime.KeepAlive(st)
}

//DestroyDescriptor destroys the spatial Transformer Desctiptor
func (st *SpatialTransformerD) DestroyDescriptor() error {
	return destroyspatialtransformdescriptor(st)
}
func destroyspatialtransformdescriptor(st *SpatialTransformerD) error {
	return Status(C.cudnnDestroySpatialTransformerDescriptor(st.descriptor)).error("DestroyDescriptor")
}
