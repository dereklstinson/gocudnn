package gocu

/*
#cgo LDFLAGS:-L/usr/local/cuda/lib64 -lcuda
#cgo CFLAGS: -I/usr/local/cuda/include/

#include <cuda.h>
*/
import "C"

func init() {
	err := newErrorDriver("intit", C.cuInit(0))

	if err != nil {
		panic(err)
	}

}
