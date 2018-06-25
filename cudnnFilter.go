package gocudnn

/*
#include <cudnn.h>
*/
import "C"

import (
	"errors"
)

//Filter is a struct that is used to call Flgs for filter descriptor, and also to create a filter descriptor
type Filter struct {
	Flgs TensorFlags
}

//FilterD is the struct holding discriptor information for cudnnFilterDescriptor_t
type FilterD struct {
	descriptor C.cudnnFilterDescriptor_t
	tensorD    *TensorD
	dims       C.int
	flags      descflag
}

//TensorD returns the tensor descripter of FilterD.  //Kind of a hack
func (f *FilterD) TensorD() *TensorD {
	return f.tensorD
}

//CreateFilterDescriptor creates a filter distriptor
/*
 The Basic 4d shape is shape[0] = # of output feature maps
					   shape[1] = # of input feature maps
					   shape[2] = height of each filter
					   shape[3] = width of each input filter
*/

//NewFilter4dDescriptor Creates and sets an Filter 4d Descpripter
func (f Filter) NewFilter4dDescriptor(data DataType, format TensorFormat, shape []int32) (*FilterD, error) {
	if len(shape) != 4 {
		return nil, errors.New("length of shape != 4")
	}
	var descriptor C.cudnnFilterDescriptor_t
	var tensordescriptor C.cudnnTensorDescriptor_t
	err := Status(C.cudnnCreateFilterDescriptor(&descriptor)).error("NewFilter4dDescriptor-create-Filter")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnCreateTensorDescriptor(&tensordescriptor)).error("NewFilter4dDescriptor-create-Tensor")
	if err != nil {
		return nil, err
	}
	cshape := int32Tocint(shape)
	err = Status(C.cudnnSetFilter4dDescriptor(descriptor, C.cudnnDataType_t(data), C.cudnnTensorFormat_t(format), cshape[0], cshape[1], cshape[2], cshape[3])).error("NewFilter4dDescriptor-set-Filter")
	if err != nil {
		return nil, err
	}

	err = Status(C.cudnnSetTensor4dDescriptor(tensordescriptor, C.cudnnTensorFormat_t(format), C.cudnnDataType_t(data), cshape[0], cshape[1], cshape[2], cshape[3])).error("NewFilter4dDescriptor-set-Tensor")
	if err != nil {
		return nil, err
	}

	return &FilterD{
		descriptor: descriptor,
		tensorD: &TensorD{
			descriptor: tensordescriptor,
			dims:       C.int(4),
			flag:       t4d,
		},
		dims:  C.int(4),
		flags: t4d}, nil
}

//NewFilterNdDescriptor creates and sets an FilterNDDescriptor
func (f Filter) NewFilterNdDescriptor(data DataType, format TensorFormat, shape []int32) (*FilterD, error) {
	if len(shape) < 4 {
		return nil, errors.New("length of shape >= 4")
	}
	var descriptor C.cudnnFilterDescriptor_t
	var tensordescriptor C.cudnnTensorDescriptor_t
	err := Status(C.cudnnCreateFilterDescriptor(&descriptor)).error("NewFilter4dDescriptor-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnCreateTensorDescriptor(&tensordescriptor)).error("NewFilter4dDescriptor-create-Tensor")
	if err != nil {
		return nil, err
	}
	cshape := int32Tocint(shape)
	dims := C.int(len(shape))
	err = Status(C.cudnnSetFilterNdDescriptor(descriptor, C.cudnnDataType_t(data), C.cudnnTensorFormat_t(format), dims, &cshape[0])).error("NewFilterNdDescriptor-set-filter")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetTensorNdDescriptorEx(tensordescriptor, C.cudnnTensorFormat_t(format), C.cudnnDataType_t(data), dims, &cshape[0])).error("NewFilterNdDescriptor-set-tensor")
	if err != nil {
		return nil, err
	}

	return &FilterD{
		descriptor: descriptor,
		tensorD: &TensorD{
			descriptor: tensordescriptor,
			dims:       dims,
			flag:       tnd,
		},
		dims:  dims,
		flags: tnd}, nil
}

//GetDescriptor returns a copy of the ConvolutionD
func (f *FilterD) GetDescriptor() (DataType, TensorFormat, []int32, error) {
	var data C.cudnnDataType_t
	var format C.cudnnTensorFormat_t
	var err error
	shape := make([]C.int, f.dims)
	if f.flags == t4d {
		err = Status(C.cudnnGetFilter4dDescriptor(f.descriptor, &data, &format, &shape[0], &shape[1], &shape[2], &shape[3])).error("GetFilter4dDescriptor")
	} else if f.flags == tnd {
		var holder C.int
		err = Status(C.cudnnGetFilterNdDescriptor(f.descriptor, f.dims, &data, &format, &holder, &shape[0])).error("GetFilterNdDescriptor")
	} else {
		err = errors.New("Unsupported flag for descriptor")
	}

	return DataType(data), TensorFormat(format), cintToint32(shape), err
}

//DestroyDescriptor Destroys Filter Descriptor
func (f *FilterD) DestroyDescriptor() error {
	var flagger1 bool
	var flagger2 bool

	err1 := f.tensorD.DestroyDescriptor()
	if err1 != nil {
		flagger1 = true
	}

	err2 := Status(C.cudnnDestroyFilterDescriptor(f.descriptor)).error("DestroyDescriptor-filt")
	if err2 != nil {
		flagger2 = true
	}
	if flagger1 == flagger2 {
		if flagger1 == true {
			return nil
		}
		return errors.New(err1.Error() + " , " + err2.Error())
	}
	if flagger1 == true {
		return err1
	}
	if flagger2 == true {
		return err2
	}
	return nil

}
