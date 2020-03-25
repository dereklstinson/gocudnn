package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/cutil"
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
	theta cutil.Mem, //Input. Affine transformation matrix. It should be of size n*2*3 for a 2d transformation, n is the number of images.
	grid cutil.Mem, /*Output. A grid of coordinates. It is of size n*h*w*2 for a 2d transformation, where n,
	h, w is specified in stDesc . In the 4th dimension, the first coordinate is x, and the
	second coordinate is y*/

) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnSpatialTfGridGeneratorForward(
				handle.x,
				s.descriptor,
				theta.Ptr(),
				grid.Ptr(),
			)).error("(s *SpatialTransformerD) GridGeneratorForward")
		})
	}
	return Status(C.cudnnSpatialTfGridGeneratorForward(
		handle.x,
		s.descriptor,
		theta.Ptr(),
		grid.Ptr(),
	)).error("(s *SpatialTransformerD) GridGeneratorForward")
}

//GridGeneratorForwardUS is like GridGeneratorForward but uses unsafe.Pointer instead of cutil.Mem
func (s *SpatialTransformerD) GridGeneratorForwardUS(
	handle *Handle,
	theta unsafe.Pointer, //Input. Affine transformation matrix. It should be of size n*2*3 for a 2d transformation, n is the number of images.
	grid unsafe.Pointer, /*Output. A grid of coordinates. It is of size n*h*w*2 for a 2d transformation, where n,
	h, w is specified in stDesc . In the 4th dimension, the first coordinate is x, and the
	second coordinate is y*/

) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnSpatialTfGridGeneratorForward(
				handle.x,
				s.descriptor,
				theta,
				grid,
			)).error("(s *SpatialTransformerD) GridGeneratorForwardUS")

		})
	}
	return Status(C.cudnnSpatialTfGridGeneratorForward(
		handle.x,
		s.descriptor,
		theta,
		grid,
	)).error("(s *SpatialTransformerD) GridGeneratorForwardUS")
}

//GridGeneratorBackward - This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
func (s *SpatialTransformerD) GridGeneratorBackward(
	handle *Handle,
	grid cutil.Mem,
	theta cutil.Mem,
) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnSpatialTfGridGeneratorBackward(
				handle.x,
				s.descriptor,
				grid.Ptr(),
				theta.Ptr(),
			)).error("(s *SpatialTransformerD) GridGeneratorBackward")
		})
	}
	return Status(C.cudnnSpatialTfGridGeneratorBackward(
		handle.x,
		s.descriptor,
		grid.Ptr(),
		theta.Ptr(),
	)).error("(s *SpatialTransformerD) GridGeneratorBackward")
}

//GridGeneratorBackwardUS is like GridGeneratorBackward but uses unsafe.Pointer instead of cutil.Mem
func (s *SpatialTransformerD) GridGeneratorBackwardUS(
	handle *Handle,
	grid unsafe.Pointer,
	theta unsafe.Pointer,
) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnSpatialTfGridGeneratorBackward(
				handle.x,
				s.descriptor,
				grid,
				theta,
			)).error("(s *SpatialTransformerD) GridGeneratorBackwardUS(")
		})
	}
	return Status(C.cudnnSpatialTfGridGeneratorBackward(
		handle.x,
		s.descriptor,
		grid,
		theta,
	)).error("(s *SpatialTransformerD) GridGeneratorBackwardUS(")
}

//SamplerForward performs the spatialtfsampleforward
func (s *SpatialTransformerD) SamplerForward(
	handle *Handle,
	alpha float64,
	xD *TensorD, x cutil.Mem,
	grid cutil.Mem,
	beta float64,
	yD *TensorD, y cutil.Mem,
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
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
			)).error("(s *SpatialTransformerD) SamplerForward")
		})
	}
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
	)).error("(s *SpatialTransformerD) SamplerForward")
}

//SamplerForwardUS is like SamplerForward but uses unsafe.Pointer instead of cutil.Mem
func (s *SpatialTransformerD) SamplerForwardUS(
	handle *Handle,
	alpha float64,
	xD *TensorD, x unsafe.Pointer,
	grid unsafe.Pointer,
	beta float64,
	yD *TensorD, y unsafe.Pointer,
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnSpatialTfSamplerForward(
				handle.x,
				s.descriptor,
				a.CPtr(),
				xD.descriptor, x,
				grid,
				b.CPtr(),
				yD.descriptor, y,
			)).error("(s *SpatialTransformerD) SamplerForwardUS")
		})
	}
	return Status(C.cudnnSpatialTfSamplerForward(
		handle.x,
		s.descriptor,
		a.CPtr(),
		xD.descriptor, x,
		grid,
		b.CPtr(),
		yD.descriptor, y,
	)).error("(s *SpatialTransformerD) SamplerForwardUS")
}

//SamplerBackward does the spatial Tranform Sample Backward
func (s *SpatialTransformerD) SamplerBackward(
	handle *Handle,
	alpha float64,
	xD *TensorD, x cutil.Mem,
	beta float64,
	dxD *TensorD, dx cutil.Mem,
	alphaDgrid float64,
	dyD *TensorD, dy cutil.Mem,
	grid cutil.Mem,
	betaDgrid float64,
	dGrid cutil.Mem,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dxD.dtype, beta)
	ad := cscalarbydatatype(xD.dtype, alphaDgrid)
	bd := cscalarbydatatype(dxD.dtype, betaDgrid)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnSpatialTfSamplerBackward(
				handle.x,
				s.descriptor,
				a.CPtr(),
				xD.descriptor, x.Ptr(),
				b.CPtr(),
				dxD.descriptor, dx.Ptr(),
				ad.CPtr(),
				dyD.descriptor, dy.Ptr(),
				grid.Ptr(),
				bd.CPtr(),
				dGrid.Ptr(),
			)).error("(s *SpatialTransformerD) SamplerBackward")
		})
	}
	return Status(C.cudnnSpatialTfSamplerBackward(
		handle.x,
		s.descriptor,
		a.CPtr(),
		xD.descriptor, x.Ptr(),
		b.CPtr(),
		dxD.descriptor, dx.Ptr(),
		ad.CPtr(),
		dyD.descriptor, dy.Ptr(),
		grid.Ptr(),
		bd.CPtr(),
		dGrid.Ptr(),
	)).error("(s *SpatialTransformerD) SamplerBackward")
}

//SamplerBackwardUS is like SamplerBackward but uses unsafe.Pointer instead of cutil.Mem
func (s *SpatialTransformerD) SamplerBackwardUS(
	handle *Handle,
	alpha float64,
	xD *TensorD, x unsafe.Pointer,
	beta float64,
	dxD *TensorD, dx unsafe.Pointer,
	alphaDgrid float64,
	dyD *TensorD, dy unsafe.Pointer,
	grid unsafe.Pointer,
	betaDgrid float64,
	dGrid unsafe.Pointer,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dxD.dtype, beta)
	ad := cscalarbydatatype(xD.dtype, alphaDgrid)
	bd := cscalarbydatatype(dxD.dtype, betaDgrid)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnSpatialTfSamplerBackward(
				handle.x,
				s.descriptor,
				a.CPtr(),
				xD.descriptor, x,
				b.CPtr(),
				dxD.descriptor, dx,
				ad.CPtr(),
				dyD.descriptor, dy,
				grid,
				bd.CPtr(),
				dGrid,
			)).error("(s *SpatialTransformerD) SamplerBackwardUS")
		})
	}
	return Status(C.cudnnSpatialTfSamplerBackward(
		handle.x,
		s.descriptor,
		a.CPtr(),
		xD.descriptor, x,
		b.CPtr(),
		dxD.descriptor, dx,
		ad.CPtr(),
		dyD.descriptor, dy,
		grid,
		bd.CPtr(),
		dGrid,
	)).error("(s *SpatialTransformerD) SamplerBackwardUS")
}

/* APIs for spatial transformer network*/

//SamplerType is used for flags
type SamplerType C.cudnnSamplerType_t

//Bilinear sets s to  SamplerType(C.CUDNN_SAMPLER_BILINEAR) and returns new value of s
func (s *SamplerType) Bilinear() SamplerType { *s = SamplerType(C.CUDNN_SAMPLER_BILINEAR); return *s }
func (s SamplerType) String() string {
	f := s
	var st string
	switch s {
	case f.Bilinear():
		st = "Bilinear"
	default:
		st = "Unssuported Type"
	}
	return "SamplerType" + st
}
func (s SamplerType) c() C.cudnnSamplerType_t { return C.cudnnSamplerType_t(s) }

//CreateSpatialTransformerDescriptor creates the spacial tesnor
func CreateSpatialTransformerDescriptor() (*SpatialTransformerD, error) {
	x := new(SpatialTransformerD)
	err := Status(C.cudnnCreateSpatialTransformerDescriptor(&x.descriptor)).error("CreateSpatialTransformerDescriptor()")
	if setfinalizer {
		runtime.SetFinalizer(x, cudnnDestroySpatialTransformerDescriptor)
	}
	return x, err
}

//Set sets spacial to nd descriptor.
func (s *SpatialTransformerD) Set(sampler SamplerType, data DataType, dimA []int32) error {
	dims := C.int(len(dimA))
	cdimA := int32Tocint(dimA)
	return Status(C.cudnnSetSpatialTransformerNdDescriptor(
		s.descriptor,
		sampler.c(),
		data.c(),
		dims,
		&cdimA[0],
	)).error("(s *SpatialTransformerD) Set")
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
