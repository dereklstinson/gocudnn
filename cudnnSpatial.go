package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//SpatialTransformerD holdes the spatial descriptor
type SpatialTransformerD struct {
	descriptor C.cudnnSpatialTransformerDescriptor_t
	dims       C.int
	gogc       bool
}

//GridGeneratorForward This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
func (s *SpatialTransformerD) GridGeneratorForward(
	handle *Handle,
	theta gocu.Mem, //Input. Affine transformation matrix. It should be of size n*2*3 for a 2d transformation, n is the number of images.
	grid gocu.Mem, /*Output. A grid of coordinates. It is of size n*h*w*2 for a 2d transformation, where n,
	h, w is specified in stDesc . In the 4th dimension, the first coordinate is x, and the
	second coordinate is y*/

) error {

	return Status(C.cudnnSpatialTfGridGeneratorForward(
		handle.x,
		s.descriptor,
		theta.Ptr(),
		grid.Ptr(),
	)).error("SpatialTfGridGeneratorForward")
}

//GridGeneratorBackward - This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
func (s *SpatialTransformerD) GridGeneratorBackward(
	handle *Handle,
	grid gocu.Mem,
	theta gocu.Mem,
) error {

	return Status(C.cudnnSpatialTfGridGeneratorBackward(
		handle.x,
		s.descriptor,
		grid.Ptr(),
		theta.Ptr(),
	)).error("SpatialTfGridGeneratorBackward")
}

//SamplerForward performs the spatialtfsampleforward
func (s *SpatialTransformerD) SamplerForward(
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
		s.descriptor,
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		grid.Ptr(),
		b.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("SpatialTfSamplerForward")
}

//SamplerBackward does the spatial Tranform Sample Backward
func (s *SpatialTransformerD) SamplerBackward(
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
		s.descriptor,
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

//SamplerType is used for flags
type SamplerType C.cudnnSamplerType_t

//Bilinear sets s to  SamplerType(C.CUDNN_SAMPLER_BILINEAR) and returns new value of s
func (s *SamplerType) Bilinear() SamplerType { *s = SamplerType(C.CUDNN_SAMPLER_BILINEAR); return *s }

func (s SamplerType) c() C.cudnnSamplerType_t { return C.cudnnSamplerType_t(s) }

//CreateSpatialTransformerDescriptor creates the spacial tesnor
func CreateSpatialTransformerDescriptor() (*SpatialTransformerD, error) {
	x := new(SpatialTransformerD)
	err := Status(C.cudnnCreateSpatialTransformerDescriptor(&x.descriptor)).error("NewSpatialTransformerNdDescriptor-create")
	if setfinalizer {
		runtime.SetFinalizer(x, cudnnDestroySpatialTransformerDescriptor)
	}
	return x, err
}

//SetND sets spacial to nd descriptor.
func (s *SpatialTransformerD) SetND(sampler SamplerType, data DataType, dimA []int32) error {
	dims := C.int(len(dimA))
	cdimA := int32Tocint(dimA)
	return Status(C.cudnnSetSpatialTransformerNdDescriptor(
		s.descriptor,
		sampler.c(),
		data.c(),
		dims,
		&cdimA[0],
	)).error("NewSpatialTransformerNdDescriptor-Set")
}

//Destroy destroys the spatial Transformer Desctiptor.  If GC is enable this function won't delete transformer. It will only return nil
//Since gc is automatically enabled this function is not functional.
func (s *SpatialTransformerD) Destroy() error {
	if s.gogc || setfinalizer {
		return nil
	}
	return cudnnDestroySpatialTransformerDescriptor(s)
}
func cudnnDestroySpatialTransformerDescriptor(s *SpatialTransformerD) error {
	return Status(C.cudnnDestroySpatialTransformerDescriptor(s.descriptor)).error("DestroyDescriptor")
}
