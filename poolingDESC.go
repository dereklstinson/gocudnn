package gocudnn

/*

#include <cudnn.h>
*/
import "C"
import "errors"

/*
 *  pooling mode
 */
//PoolingMode is used for flags in pooling
type PoolingMode C.cudnnPoolingMode_t

//Flags for Pooling Mode
const (
	PoolingMax                        PoolingMode = C.CUDNN_POOLING_MAX
	PoolingAverageCountIncludePadding PoolingMode = C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING /* count for average includes padded values */
	PoolingAverageCountExcludePadding PoolingMode = C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING /* count for average does not include padded values */
	PoolingMaxDeterministic           PoolingMode = C.CUDNN_POOLING_MAX_DETERMINISTIC
)

func (p PoolingMode) c() C.cudnnPoolingMode_t { return C.cudnnPoolingMode_t(p) }

type PoolingD struct {
	descriptor C.cudnnPoolingDescriptor_t
	dims       C.int
}

func NewPooling2dDescriptor(
	mode PoolingMode,
	nan PropagationNAN,
	window []int32, // height and wideth
	padding []int32, //height and width
	stride []int32, //height and width
) (*PoolingD, error) {
	if len(window) != len(padding) || len(window) != len(stride) || len(window) != 2 {
		return nil, errors.New("Window Padding and Stride array lengths need to be 2")
	}
	var desc C.cudnnPoolingDescriptor_t
	err := Status(C.cudnnCreatePoolingDescriptor(&desc)).error("NewPooling2dDescriptor-create")
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
	return &PoolingD{descriptor: desc, dims: C.int(2)}, nil
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
	return PoolingMode(mode), PropagationNAN(nan), cintToint32(window), cintToint32(padding), cintToint32(stride), err
}
