package gocudnn

/*
#include <cudnn.h>
#include <cuda.h>
*/
import "C"
import (
	"errors"

	"github.com/dereklstinson/cutil"
	"github.com/dereklstinson/half"
)

var cudnndebugmode bool

//DebugMode is for debugging code soley for these bindings.
func DebugMode() {
	cudnndebugmode = true
}

//DimMax is the max dims for tensors
const DimMax = int32(C.CUDNN_DIM_MAX)

//BnMinEpsilon is the min epsilon for batchnorm
//It used to be 1e-5, but it is now 0
const BnMinEpsilon = (float64)(C.CUDNN_BN_MIN_EPSILON)

//CScalarByDataType takes the DataType flag and puts num into a CScalar interface. The value of num will be bound by what is passed for DataType.
//If a DataType isn't supported by the function it will return nil.
func cscalarbydatatype(dtype DataType, num float64) cutil.CScalar {
	var x DataType //CUDNN_DATATYPE_FLOAT
	switch dtype {
	case x.Double():
		return cutil.CDouble(num)
	case x.Float():
		return cutil.CFloat(num)
	case x.Int32():
		y := float32(num)
		return cutil.CFloat(y)
	case x.Int8():
		y := float32(num)
		return cutil.CFloat(y)
	case x.UInt8():
		y := float32(num)
		return cutil.CFloat(y)
	case x.Half():
		y := float32(num)
		return cutil.CFloat(y)
	default:
		return nil
	}

}

//CScalarByDataType takes the DataType flag and puts num into a CScalar interface. The value of num will be bound by what is passed for DataType.
//If a DataType isn't supported by the function it will return nil.
func cscalarbydatatypeforsettensor(dtype DataType, num float64) cutil.CScalar {
	var x DataType //CUDNN_DATATYPE_FLOAT
	switch dtype {
	case x.Double():
		return cutil.CDouble(num)
	case x.Float():
		return cutil.CFloat(num)
	case x.Int32():
		return cutil.CInt(num)
	case x.Int8():
		return cutil.CChar(num)
	case x.UInt8():
		return cutil.CUChar(num)
	case x.Half():
		y := float32(num)
		return cutil.CHalf(half.NewFloat16(y))

	default:
		return nil
	}

}

//RuntimeTag is a type that cudnn looks to check or kernels to see if they are working correctly.
//Should be used with batchnormialization
type RuntimeTag C.cudnnRuntimeTag_t

// ErrQueryMode are basically flags that are used for different modes that are exposed through the
//types methods
type ErrQueryMode C.cudnnErrQueryMode_t

//RawCode sets e to and returns ErrQueryMode(C.CUDNN_ERRQUERY_RAWCODE)
func (e *ErrQueryMode) RawCode() ErrQueryMode { *e = ErrQueryMode(C.CUDNN_ERRQUERY_RAWCODE); return *e }

//NonBlocking sets e to and returns ErrQueryMode(C.CUDNN_ERRQUERY_NONBLOCKING)
func (e *ErrQueryMode) NonBlocking() ErrQueryMode {
	*e = ErrQueryMode(C.CUDNN_ERRQUERY_NONBLOCKING)
	return *e
}

//Blocking sets e to and returns ErrQueryMode(C.CUDNN_ERRQUERY_BLOCKING)
func (e *ErrQueryMode) Blocking() ErrQueryMode {
	*e = ErrQueryMode(C.CUDNN_ERRQUERY_BLOCKING)
	return *e
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
//tag should be nil
func (handle *Handle) QueryRuntimeError(mode ErrQueryMode, tag *RuntimeTag) (Status, error) {
	var rstatus C.cudnnStatus_t

	if tag == nil {
		err := Status(C.cudnnQueryRuntimeError(handle.x, &rstatus, C.cudnnErrQueryMode_t(mode), nil)).error("QueryRuntimeError")

		return Status(rstatus), err
	}

	return Status(rstatus), errors.New("Tag flags not supported")

}
