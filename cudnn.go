package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

//DimMax is the max dims for tensors
const DimMax = int32(8)

//SizeT is a type used by cudnn
type SizeT C.size_t

func (s SizeT) c() C.size_t { return C.size_t(s) }

//Memer is an interface for memory
type Memer interface {
	Ptr() unsafe.Pointer
	ByteSize() SizeT
}

//CScalar is used for scalar multiplications with cudnn.  They have to be Ctypes. It could have easily been called voider
type CScalar interface {
	CPtr() unsafe.Pointer
}

//CFloat is a float in C
type CFloat C.float

func (f *CFloat) c() C.float { return C.float(*f) }

//CPtr returns an unsafe pointer of the float
func (f *CFloat) CPtr() unsafe.Pointer { return unsafe.Pointer(f) }

//CDouble is a double in C
type CDouble C.double

func (d *CDouble) c() C.double { return C.double(*d) }

//CPtr returns an unsafe pointer of the double
func (d *CDouble) CPtr() unsafe.Pointer { return unsafe.Pointer(d) }

//CInt is a int in C
type CInt C.int

func (i *CInt) c() C.int { return C.int(*i) }

//CPtr returns an unsafe pointer of the int
func (i *CInt) CPtr() unsafe.Pointer { return unsafe.Pointer(i) }

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
func GetVersion() SizeT {
	return SizeT(C.cudnnGetVersion())
}

//GetCudaartVersion prints cuda run time version
func GetCudaartVersion() SizeT {
	return SizeT(C.cudnnGetCudartVersion())
}

//QueryRuntimeError check cudnnQueryRuntimeError in DEEP Learning SDK Documentation
func (handle *Handle) QueryRuntimeError(mode ErrQueryMode, tag *RuntimeTag) (Status, error) {
	var rstatus C.cudnnStatus_t

	if tag == nil {
		err := Status(C.cudnnQueryRuntimeError(handle.x, &rstatus, C.cudnnErrQueryMode_t(mode), nil)).error("QueryRuntimeError")

		return Status(rstatus), err
	}

	return Status(rstatus), errors.New("Tag flags not supported")

}
