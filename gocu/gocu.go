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

func init() {
	x := C.cuInit(0)
	if x != C.CUDA_SUCCESS {
		panic(x)
	}

}
