package gocudnn

/*
#include <cudnn.h>
*/
import "C"

import (
	"fmt"
	"runtime"

	"github.com/dereklstinson/cutil"
)

//FilterD is the struct holding discriptor information for cudnnFilterDescriptor_t
type FilterD struct {
	descriptor C.cudnnFilterDescriptor_t
	gogc       bool
	dims       C.int
}

const filterdescriptorallndtest = true

func (f *FilterD) String() string {
	dtype, frmt, dims, err := f.Get()
	if err != nil {
		return fmt.Sprintf("FilterDescriptor{error in pulling values")
	}

	return fmt.Sprintf("FilterDescriptor{\n%v,\n%v,\nShape : %v,\n}\n", frmt, dtype, dims)

}

/*
//TensorD returns the tensor descripter of FilterD.  //Kind of a hack
func (f *FilterD) TensorD() *TensorD {

	return f.tensorD
}
*/

//CreateFilterDescriptor creates a filter distriptor
func CreateFilterDescriptor() (*FilterD, error) {
	flt := new(FilterD)
	err := Status(C.cudnnCreateFilterDescriptor(&flt.descriptor)).error("NewFilter4dDescriptor-create-Filter")
	if setfinalizer || flt.gogc {
		runtime.SetFinalizer(flt, destroyfilterdescriptor)
	}
	return flt, err
}

//Set sets the filter descriptor
//
//	Basic 4D filter
//
// The Basic NCHW shape is shape[0] = # of output feature maps
//				     	   shape[1] = # of input feature maps
//					       shape[2] = height of each filter
//					       shape[3] = width of each input filter
//
// The Basic NHWC shape is shape[0] = # of output feature maps
//						   shape[1] = height of each filter
//						   shape[2] = width of each input filter
//				     	   shape[3] = # of input feature maps
//
//  Basic ND filter
//
// The Basic NCHW shape is shape[0]   = # of output feature maps
//				     	   shape[1]   = # of input feature maps
//					       shape[.]   = feature dims
//					       shape[N-1] = feature dims
//
// The Basic NHWC shape is shape[0]   = # of output feature maps
//						   shape[.]   = feature dims
//				     	   shape[N-1] = # of input feature maps
func (f *FilterD) Set(dtype DataType, format TensorFormat, shape []int32) error {
	cshape := int32Tocint(shape)
	f.dims = (C.int)(len(shape))

	return Status(C.cudnnSetFilterNdDescriptor(f.descriptor, dtype.c(), format.c(), f.dims, &cshape[0])).error("(*FilterD)SetND")
}

//Get returns a copy of the ConvolutionD
func (f *FilterD) Get() (dtype DataType, frmt TensorFormat, shape []int32, err error) {
	if f.dims == 0 {
		f.dims = C.CUDNN_DIM_MAX
		shape = make([]int32, f.dims)
		var actual C.int
		err = Status(C.cudnnGetFilterNdDescriptor(f.descriptor, f.dims, dtype.cptr(), frmt.cptr(), &actual, (*C.int)(&shape[0]))).error("cudnnGetFilterNdDescriptor")
		f.dims = actual
		return dtype, frmt, shape[:f.dims], err
	}
	var holder C.int
	shape = make([]int32, f.dims)
	err = Status(C.cudnnGetFilterNdDescriptor(f.descriptor, f.dims, dtype.cptr(), frmt.cptr(), &holder, (*C.int)(&shape[0]))).error("cudnnGetFilterNdDescriptor")
	return dtype, frmt, shape, err
}

//GetSizeInBytes returns the size in bytes for the filter
func (f *FilterD) GetSizeInBytes() (uint, error) {
	dtype, _, shape, err := f.Get()
	if err != nil {
		return 0, err
	}
	return FindSizeTfromVol(shape, dtype), nil
}

//Destroy Destroys Filter Descriptor if GC is not set. if GC is set then it won't do anything
func (f *FilterD) Destroy() error {
	if f.gogc || setfinalizer {
		return nil
	}
	return destroyfilterdescriptor(f)
}

func destroyfilterdescriptor(f *FilterD) error {

	err := Status(C.cudnnDestroyFilterDescriptor(f.descriptor)).error("DestroyDescriptor-filt")
	if err != nil {
		return err
	}
	f = nil

	return nil

}

//ReorderFilterBias -reorders the filter and bias values. It can be used to enhance
//the inference time by separating the reordering operation from convolution.
//
//For example, convolutions in a neural network of multiple layers can require
//reordering of kernels at every layer, which can take up a significant fraction
//of the total inference time. Using this function, the reordering can be done one
//time on the filter and bias data followed by the convolution operations at the multiple
//layers, thereby enhancing the inference time.
func (f *FilterD) ReorderFilterBias(h *Handle,
	r Reorder,
	filtersrc, reorderfilterdest cutil.Mem,
	reorderbias bool,
	biassrc, reorderbiasdest cutil.Mem) error {
	var robias C.int
	if reorderbias {
		robias = 1
	}
	return Status(C.cudnnReorderFilterAndBias(h.x, f.descriptor,
		r.c(), filtersrc.Ptr(), reorderfilterdest.Ptr(),
		robias, biassrc.Ptr(), reorderbiasdest.Ptr())).error("ReorderFilterBias")

}

/*
//Set4DEx sets the filter up like nd but using a 4d
func (f *FilterD) Set4D(dtype DataType, format TensorFormat, shape []int32) error {
	l := len(shape)
	f.flags = t4d
	f.dims = (C.int)(len(shape))
	if l != 4 {
		return errors.New("length of shape needs to be 4")
	}
	cshape := int32Tocint(shape)
	if filterdescriptorallndtest{
		return Status(C.cudnnSetFilterNdDescriptor(f.descriptor, dtype.c(), format.c(), f.dims, &cshape[0])).error("(*FilterD)SetND")
	}
	return Status(C.cudnnSetFilter4dDescriptor(f.descriptor, dtype.c(), format.c(), cshape[0], cshape[1], cshape[2], cshape[3])).error("(*FilterD)Set4DEx")
}
*/
/*
//Set4D sets the filter like normal
func (f *FilterD) Set4D(dtype DataType, format TensorFormat, k, c, h, w int32) error {
	f.flags = t4d
	f.dims = (C.int)(2)
	return Status(C.cudnnSetFilter4dDescriptor(f.descriptor, dtype.c(), format.c(), (C.int)(k), (C.int)(c), (C.int)(h), (C.int)(w))).error("(*FilterD)Set4DEx")
}
*/
