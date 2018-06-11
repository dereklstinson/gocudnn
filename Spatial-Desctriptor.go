package gocudnn

/*
#include <cudnn.h>
*/
import "C"

/* APIs for spatial transformer network*/

//SamplerType is used for flags
type SamplerType C.cudnnSamplerType_t

//SamplerBilinear is the flag for SamplerType
const SamplerBilinear SamplerType = C.CUDNN_SAMPLER_BILINEAR

func (s SamplerType) c() C.cudnnSamplerType_t { return C.cudnnSamplerType_t(s) }

//SpatialTransformerD holdes the spatial descriptor
type SpatialTransformerD struct {
	descriptor C.cudnnSpatialTransformerDescriptor_t
	dims       C.int
}

//NewSpatialTransformerNdDescriptor creates and sets SpatialTransformerD
func NewSpatialTransformerNdDescriptor(
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
