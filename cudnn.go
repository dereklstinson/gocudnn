package gocudnn

/*
#include <cudnn.h>
#include <cuda.h>


*/
import "C"
import (
	"errors"
	"unsafe"

	"github.com/dereklstinson/half"
)

//DimMax is the max dims for tensors
const DimMax = int32(8)

/*
//SizeT is a type used by cudnn
type SizeT C.size_t

func (s SizeT) c() C.size_t { return C.size_t(s) }
*/

//CScalar is used for scalar multiplications with cudnn.  They have to be Ctypes. It could have easily been called voider
type CScalar interface {
	CPtr() unsafe.Pointer
	Bytes() int
	SizeT() uint
}

//CScalarByDataType takes the DataType flag and puts num into a CScalar interface. The value of num will be bound by what is passed for DataType.
//If a DataType isn't supported by the function it will return nil.
func CScalarByDataType(dtype DataType, num float64) CScalar {
	var x DataTypeFlag //CUDNN_DATATYPE_FLOAT
	switch dtype {
	case x.Double():
		return CDouble(num)
	case x.Float():
		return CFloat(num)
	case x.Int32():
		return CInt(num)
	case x.Int8():
		return CInt8(num)
	case x.UInt8():
		return CUInt8(num)

	default:
		return nil
	}

}

//CScalarConversion takes a go type and converts it to a CScalar interface. golang type int and int32 will both be converted to a CInt type.
//If a go type is not supported then it will return a nil.
//Current support is float64,float32,int, int32, int8,uint32, uint, uint8 ( I think byte should work because when I put it in the switch with uint8 it says duplicate type).
func CScalarConversion(gotype interface{}) CScalar {
	switch x := gotype.(type) {
	case float64:
		return CDouble(x)
	case float32:
		return CFloat(x)
	case int:
		return CInt(x)
	case int32:
		return CInt(x)
	case int8:
		return CInt8(x)
	case uint8:
		return CUInt8(x)
	case uint32:
		return CUInt(x)
	case uint:
		return CUInt(x)
	case half.Float16:
		return CHalf(x)
	case bool:

		if x == true {
			return CInt(255)
		}
		return CInt(0)

	case CScalar:
		return x

	default:
		return nil
	}
}

//CHalf is a half precision
type CHalf C.ushort

func (f CHalf) c() C.ushort { return C.ushort(f) }

//CPtr returns an unsafe pointer of the half
func (f CHalf) CPtr() unsafe.Pointer { return unsafe.Pointer(&f) }

//Bytes returns the number of bytes the CScalar has as an int
func (f CHalf) Bytes() int { return 4 }

//SizeT returns the number of bytes the CScalar has as an sizeT
func (f CHalf) SizeT() uint { return (4) }

//CFloat is a float in C
type CFloat C.float

func (f CFloat) c() C.float { return C.float(f) }

//CPtr returns an unsafe pointer of the float
func (f CFloat) CPtr() unsafe.Pointer { return unsafe.Pointer(&f) }

//Bytes returns the number of bytes the CScalar has
func (f CFloat) Bytes() int { return 4 }

//SizeT returns the number of bytes the CScalar has
func (f CFloat) SizeT() uint { return (4) }

//CDouble is a double in C
type CDouble C.double

func (d CDouble) c() C.double { return C.double(d) }

//CPtr returns an unsafe pointer of the double
func (d CDouble) CPtr() unsafe.Pointer { return unsafe.Pointer(&d) }

//Bytes returns the number of bytes the CScalar has
func (d CDouble) Bytes() int { return 8 }

//SizeT returns the number of bytes the CScalar has
func (d CDouble) SizeT() uint { return (8) }

//CInt is a int in C
type CInt C.int

func (i CInt) c() C.int { return C.int(i) }

//CPtr returns an unsafe pointer of the int
func (i CInt) CPtr() unsafe.Pointer { return unsafe.Pointer(&i) }

//Bytes returns the number of bytes the CScalar has
func (i CInt) Bytes() int { return 4 }

//SizeT returns the number of bytes the CScalar has
func (i CInt) SizeT() uint { return (4) }

//CUInt is an unsigned int in C
type CUInt C.uint

//CPtr returns an unsafe pointer of the Unsigned Int
func (i CUInt) CPtr() unsafe.Pointer { return unsafe.Pointer(&i) }

//Bytes returns the number of bytes the CScalar has
func (i CUInt) Bytes() int { return 4 }
func (i CUInt) c() C.uint  { return C.uint(i) }

//SizeT returns the number of bytes the CScalar has
func (i CUInt) SizeT() uint { return (4) }

//CInt8 is a signed char
type CInt8 C.char

func (c CInt8) c() C.char { return C.char(c) }

//CPtr retunrs an unsafe pointer for CInt8
func (c CInt8) CPtr() unsafe.Pointer { return unsafe.Pointer(&c) }

//Bytes returns the number of bytes the CScalar has
func (c CInt8) Bytes() int { return 1 }

//SizeT returns the number of bytes the CScalar has
func (c CInt8) SizeT() uint { return (1) }

//CUInt8 is a C.uchar
type CUInt8 C.uchar

func (c CUInt8) c() C.uchar { return C.uchar(c) }

//Bytes returns the number of bytes the CScalar has
func (c CUInt8) Bytes() int { return 1 }

//CPtr retunrs an unsafe pointer for CUInt8
func (c CUInt8) CPtr() unsafe.Pointer { return unsafe.Pointer(&c) }

//SizeT returns the number of bytes the CScalar has
func (c CUInt8) SizeT() uint { return (1) }

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
