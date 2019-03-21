package gocudnn

/*
#include <cudnn.h>
*/
import "C"

import (
	"errors"
	"runtime"
)

//FilterD is the struct holding discriptor information for cudnnFilterDescriptor_t
type FilterD struct {
	descriptor C.cudnnFilterDescriptor_t
	gogc       bool
	dims       C.int
	flags      descflag
}

/*
//TensorD returns the tensor descripter of FilterD.  //Kind of a hack
func (f *FilterD) TensorD() *TensorD {

	return f.tensorD
}
*/

//CreateFilterDescriptor creates a filter distriptor
/*
 The Basic 4d shape is shape[0] = # of output feature maps
					   shape[1] = # of input feature maps
					   shape[2] = height of each filter
					   shape[3] = width of each input filter
*/
func CreateFilterDescriptor() (*FilterD, error) {
	flt := new(FilterD)
	err := Status(C.cudnnCreateFilterDescriptor(&flt.descriptor)).error("NewFilter4dDescriptor-create-Filter")
	if setfinalizer || flt.gogc {
		runtime.SetFinalizer(flt, destroyfilterdescriptor)
	}
	return flt, err
}

//SetND sets the filter to ND
func (f *FilterD) SetND(dtype DataType, format TensorFormat, shape []int32) error {
	cshape := int32Tocint(shape)
	f.flags = tnd
	f.dims = (C.int)(len(shape))
	return Status(C.cudnnSetFilterNdDescriptor(f.descriptor, dtype.c(), format.c(), f.dims, &cshape[0])).error("(*FilterD)SetND")
}

//Set4DEx sets the filter up like nd but using a 4d
func (f *FilterD) Set4DEx(dtype DataType, format TensorFormat, shape []int32) error {
	l := len(shape)
	f.flags = t4d
	f.dims = (C.int)(len(shape))
	if l != 4 {
		return errors.New("length of shape needs to be 4")
	}
	cshape := int32Tocint(shape)
	return Status(C.cudnnSetFilter4dDescriptor(f.descriptor, dtype.c(), format.c(), cshape[0], cshape[1], cshape[2], cshape[3])).error("(*FilterD)Set4DEx")
}

//Set4D sets the filter like normal
func (f *FilterD) Set4D(dtype DataType, format TensorFormat, k, c, h, w int32) error {
	f.flags = t4d
	f.dims = (C.int)(2)
	return Status(C.cudnnSetFilter4dDescriptor(f.descriptor, dtype.c(), format.c(), (C.int)(k), (C.int)(c), (C.int)(h), (C.int)(w))).error("(*FilterD)Set4DEx")
}

//Get returns a copy of the ConvolutionD
func (f *FilterD) Get() (dtype DataType, frmt TensorFormat, shape []int32, err error) {

	shape = make([]int32, f.dims)
	if f.flags == t4d {
		err = Status(C.cudnnGetFilter4dDescriptor(f.descriptor, dtype.cptr(), frmt.cptr(), (*C.int)(&shape[0]), (*C.int)(&shape[1]), (*C.int)(&shape[2]), (*C.int)(&shape[3]))).error("GetFilter4dDescriptor")
	} else if f.flags == tnd {
		var holder C.int
		err = Status(C.cudnnGetFilterNdDescriptor(f.descriptor, f.dims, dtype.cptr(), frmt.cptr(), &holder, (*C.int)(&shape[0]))).error("GetFilterNdDescriptor")
	} else {
		err = errors.New("Unsupported flag for descriptor")
	}

	return dtype, frmt, shape, err
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
