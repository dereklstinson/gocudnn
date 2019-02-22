//Package gocu contains common interfaces to allow the different cuda packages/libraries to intermix
package gocu

import "unsafe"

//Streamer allows streams made from cuda or
type Streamer interface {
	Ptr() unsafe.Pointer
	Sync() error
}
