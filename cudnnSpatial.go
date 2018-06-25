package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//Spatial hods the funcs and flags of Spatial Stuff
type Spatial struct {
	Funcs SpatialFuncs
	Flgs  SamplerTypeFlag
}

//SpatialFuncs is a struct used to call Spatial functions as methods
type SpatialFuncs struct {
}

//SpatialTfGridGeneratorForward This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
func (space SpatialFuncs) SpatialTfGridGeneratorForward(
	handle *Handle,
	st *SpatialTransformerD,
	theta Memer, //Input. Affine transformation matrix. It should be of size n*2*3 for a 2d transformation, n is the number of images.
	grid Memer, /*Output. A grid of coordinates. It is of size n*h*w*2 for a 2d transformation, where n,
	h, w is specified in stDesc . In the 4th dimension, the first coordinate is x, and the
	second coordinate is y*/

) error {
	return Status(C.cudnnSpatialTfGridGeneratorForward(
		handle.x,
		st.descriptor,
		theta.Ptr(),
		grid.Ptr(),
	)).error("SpatialTfGridGeneratorForward")
}

//SpatialTfGridGeneratorBackward - This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
func (space SpatialFuncs) SpatialTfGridGeneratorBackward(
	handle *Handle,
	st *SpatialTransformerD,
	grid Memer,
	theta Memer,

) error {
	return Status(C.cudnnSpatialTfGridGeneratorBackward(
		handle.x,
		st.descriptor,
		grid.Ptr(),
		theta.Ptr(),
	)).error("SpatialTfGridGeneratorBackward")
}

//SpatialTfSamplerForward performs the spatialtfsampleforward
func (space SpatialFuncs) SpatialTfSamplerForward(
	handle *Handle,
	st *SpatialTransformerD,
	alpha CScalar,
	xD *TensorD,
	x Memer,
	grid Memer,
	beta CScalar,
	yD *TensorD,
	y Memer,
) error {
	return Status(C.cudnnSpatialTfSamplerForward(
		handle.x,
		st.descriptor,
		alpha.CPtr(),
		xD.descriptor,
		x.Ptr(),
		grid.Ptr(),
		beta.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("SpatialTfSamplerForward")
}

//SpatialTfSamplerBackward does the spatial Tranform Sample Backward
func (space SpatialFuncs) SpatialTfSamplerBackward(
	handle *Handle,
	st *SpatialTransformerD,
	alpha CScalar,
	xD *TensorD,
	x Memer,
	beta CScalar,
	dxD *TensorD,
	dx Memer,
	alphaDgrid CScalar,
	dyD *TensorD,
	dy Memer,
	grid Memer,
	betaDgrid CScalar,
	dGrid Memer,

) error {
	return Status(C.cudnnSpatialTfSamplerBackward(
		handle.x,
		st.descriptor,
		alpha.CPtr(),
		xD.descriptor,
		x.Ptr(),
		beta.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
		alphaDgrid.CPtr(),
		dyD.descriptor,
		dy.Ptr(),
		grid.Ptr(),
		betaDgrid.CPtr(),
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
) (*SpatialTransformerD, error) {
	var desc C.cudnnSpatialTransformerDescriptor_t
	err := Status(C.cudnnCreateSpatialTransformerDescriptor(&desc)).error("NewSpatialTransformerNdDescriptor-create")
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
	return &SpatialTransformerD{
		descriptor: desc,
		dims:       dims,
	}, nil
}

//DestroyDescriptor destroys the spatial Transformer Desctiptor
func (sp *SpatialTransformerD) DestroyDescriptor() error {
	return Status(C.cudnnDestroySpatialTransformerDescriptor(sp.descriptor)).error("DestroyDescriptor")
}
