//Package gocu contains common interfaces to allow the different cuda packages/libraries to intermix with each other and with go.
package gocu

//#include <cuda.h>
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

func init() {
	err := newErrorDriver("intit", C.cuInit(0))

	if err != nil {
		panic(err)
	}

}

//Filler fills a slice passed as an empty interface from the mem passed.
//The interface{} needs to be some sort of pointer type.  Either
type Filler interface {
	Fill(interface{}, Mem) error
}
