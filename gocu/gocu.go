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
	DPtr() *unsafe.Pointer
	Ptr() unsafe.Pointer
}

//Offset returns a gocu.Mem with the unsafe.Pointer stored in it at the offset bybytes
func Offset(mem Mem, bybytes uint) Mem {
	return WrapUnsafe(unsafe.Pointer(uintptr(mem.Ptr()) + uintptr(bybytes)))
}

func init() {
	x := C.cuInit(0)
	if x != C.CUDA_SUCCESS {
		panic(x)
	}

}

//Filler fills a slice passed as an empty interface from the mem passed.
//The interface{} needs to be some sort of pointer type.  Either
type Filler interface {
	Fill(interface{}, Mem) error
}
