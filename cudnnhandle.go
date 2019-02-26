package gocudnn

/*

#include <cudnn.h>
#include <cuda_runtime_api.h>

*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
)

//Handle is a struct containing a cudnnHandle_t which is basically a Pointer to a CUContext
type Handle struct {
	x C.cudnnHandle_t
}

//Pointer is a pointer to the handle
func (handle *Handle) Pointer() unsafe.Pointer {
	if setkeepalive {
		handle.keepsalive()
	}
	return unsafe.Pointer(handle.x)
}

//NewHandle creates a handle its basically a Context
func NewHandle() *Handle {

	var handle C.cudnnHandle_t

	err := Status(C.cudnnCreate(&handle)).error("NewHandle")
	if err != nil {
		panic(err)
	}
	var handler = &Handle{x: handle}
	if setfinalizer {
		runtime.SetFinalizer(handler, destroycudnnhandle)
	}

	return handler
}

func (handle *Handle) keepsalive() {
	runtime.KeepAlive(handle)
}

//Destroy destroys the handle
func (handle *Handle) Destroy() error {
	return destroycudnnhandle(handle)
}
func destroycudnnhandle(handle *Handle) error {
	return Status(C.cudnnDestroy(handle.x)).error("(*Handle).Destroy")
}

//SetStream passes a stream to sent in the cuda handle
func (handle *Handle) SetStream(s gocu.Streamer) error {

	y := C.cudnnSetStream(handle.x, C.cudaStream_t(s.Ptr()))
	if setkeepalive {
		keepsalivebuffer(handle, s)
	}
	return Status(y).error("(*Handle).SetStream")
}

//GetStream will return a stream that the handle is using
func (handle *Handle) GetStream() (gocu.Streamer, error) {

	var s C.cudaStream_t
	//var some *C.cudaStream_t
	//x := C.cudnnHandle_t(handle.Pointer())

	err := Status(C.cudnnGetStream(handle.x, &s)).error("*Handle).GetStream")
	//	s.stream = *some

	return cudart.ExternalWrapper(unsafe.Pointer(s)), err
}
