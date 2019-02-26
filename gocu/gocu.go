//Package gocu contains common interfaces to allow the different cuda packages/libraries to intermix with each other and with go.
package gocu

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
	OffSet(units uint) Mem
}
