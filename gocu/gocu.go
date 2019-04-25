//Package gocu contains common interfaces to allow the different cuda packages/libraries to intermix with each other and with go.
package gocu

/*
#include <cuda.h>
*/
import "C"
import (
	"unsafe"
)

//Streamer allows streams made from cuda or
type Streamer interface {
	Ptr() unsafe.Pointer
	Sync() error
}

//Mem is mem that is shared betweek cuda packages
type Mem interface {
	Ptr() unsafe.Pointer
	DPtr() *unsafe.Pointer
}

//Pointer interface returns an unsafe.Pointer
type Pointer interface {
	Ptr() unsafe.Pointer
}

//DPointer interface returns an *unsafe.Pointer
type DPointer interface {
	DPtr() *unsafe.Pointer
}

//Offset returns a gocu.Mem with the unsafe.Pointer stored in it at the offset bybytes
func Offset(p Pointer, bybytes uint) Pointer {
	return WrapUnsafe(unsafe.Pointer(uintptr(p.Ptr()) + uintptr(bybytes)))
}

func init() {
	x := C.cuInit(0)
	if x != C.CUDA_SUCCESS {
		panic(x)
	}

}
