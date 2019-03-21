package gocudnn

/*
#include <cudnn.h>
#include <cuda.h>


*/
import "C"
import (
	"errors"

	"github.com/dereklstinson/half"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//DimMax is the max dims for tensors
const DimMax = int32(8)

//CScalarByDataType takes the DataType flag and puts num into a CScalar interface. The value of num will be bound by what is passed for DataType.
//If a DataType isn't supported by the function it will return nil.
func cscalarbydatatype(dtype DataType, num float64) gocu.CScalar {
	var x DataType //CUDNN_DATATYPE_FLOAT
	switch dtype {
	case x.Double():
		return gocu.CDouble(num)
	case x.Float():
		return gocu.CFloat(num)
	case x.Int32():
		y := float32(num)
		return gocu.CFloat(y)
	case x.Int8():
		y := float32(num)
		return gocu.CFloat(y)
	case x.UInt8():
		y := float32(num)
		return gocu.CFloat(y)
	case x.Half():
		y := float32(num)
		return gocu.CFloat(y)
	default:
		return nil
	}

}

//CScalarByDataType takes the DataType flag and puts num into a CScalar interface. The value of num will be bound by what is passed for DataType.
//If a DataType isn't supported by the function it will return nil.
func cscalarbydatatypeforsettensor(dtype DataType, num float64) gocu.CScalar {
	var x DataType //CUDNN_DATATYPE_FLOAT
	switch dtype {
	case x.Double():
		return gocu.CDouble(num)
	case x.Float():
		return gocu.CFloat(num)
	case x.Int32():
		return gocu.CInt(num)
	case x.Int8():
		return gocu.CInt8(num)
	case x.UInt8():
		return gocu.CUInt8(num)
	case x.Half():
		y := float32(num)
		return gocu.CHalf(half.NewFloat16(y))

	default:
		return nil
	}

}

//RuntimeTag is a type that cudnn uses that I am not sure of yet
type RuntimeTag C.cudnnRuntimeTag_t

// ErrQueryMode are basically flags that are used for different modes
type ErrQueryMode C.cudnnErrQueryMode_t

//ErrQueryModeFlag returns the default flag of ErrQueryMode(C.CUDNN_ERRQUERY_RAWCODE) can be changed with methods
func ErrQueryModeFlag() ErrQueryMode {
	return ErrQueryMode(C.CUDNN_ERRQUERY_RAWCODE)
}

//RawCode return  ErrQueryMode(C.CUDNN_ERRQUERY_RAWCODE)
func (e ErrQueryMode) RawCode() ErrQueryMode {
	return ErrQueryMode(C.CUDNN_ERRQUERY_RAWCODE)
}

//NonBlocking return  ErrQueryMode(C.CUDNN_ERRQUERY_NONBLOCKING)
func (e ErrQueryMode) NonBlocking() ErrQueryMode {
	return ErrQueryMode(C.CUDNN_ERRQUERY_NONBLOCKING)
}

//Blocking 	return  ErrQueryMode(C.CUDNN_ERRQUERY_BLOCKING)
func (e ErrQueryMode) Blocking() ErrQueryMode {
	return ErrQueryMode(C.CUDNN_ERRQUERY_BLOCKING)
}

func (e ErrQueryMode) c() C.cudnnErrQueryMode_t { return C.cudnnErrQueryMode_t(e) }

//GetVersion returns the version
func GetVersion() uint {
	return uint(C.cudnnGetVersion())
}

//GetCudaartVersion prints cuda run time version
func GetCudaartVersion() uint {
	return uint(C.cudnnGetCudartVersion())
}

//QueryRuntimeError check cudnnQueryRuntimeError in DEEP Learning SDK Documentation
func (handle *Handle) QueryRuntimeError(mode ErrQueryMode, tag *RuntimeTag) (Status, error) {
	var rstatus C.cudnnStatus_t

	if tag == nil {
		err := Status(C.cudnnQueryRuntimeError(handle.x, &rstatus, C.cudnnErrQueryMode_t(mode), nil)).error("QueryRuntimeError")

		return Status(rstatus), err
	}
	if setkeepalive {
		keepsalivebuffer(handle)
	}
	return Status(rstatus), errors.New("Tag flags not supported")

}
