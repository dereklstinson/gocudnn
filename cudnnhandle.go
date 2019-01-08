package gocudnn

/*

#include <cudnn.h>
#include <cuda_runtime_api.h>

*/
import "C"
import (
	"runtime"
	"unsafe"
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

/*
//Create creates the handle
func (handle *Handle) create() error {

	y := C.cudnnCreate(&handle.x)

	return Status(y).error("(*Handle).Create")
}
*/

//Destroy destroys the handle
func (handle *Handle) Destroy() error {
	return destroycudnnhandle(handle)
}
func destroycudnnhandle(handle *Handle) error {
	return Status(C.cudnnDestroy(handle.x)).error("(*Handle).Destroy")
}

//typedefed CUstream_st *CUstream
//typedefed CUstream_st *cudastream_t

//SetStream passes a stream to sent in the cuda handle
func (handle *Handle) SetStream(s *Stream) error {

	y := C.cudnnSetStream(handle.x, s.stream)
	if setkeepalive {
		keepsalivebuffer(handle, s)
	}
	return Status(y).error("(*Handle).SetStream")
}

//GetStream will return a stream that the handle is using
func (handle *Handle) GetStream() (*Stream, error) {

	var s C.cudaStream_t
	//var some *C.cudaStream_t
	//x := C.cudnnHandle_t(handle.Pointer())

	y := C.cudnnGetStream(handle.x, &s)
	//	s.stream = *some

	newstream := &Stream{
		stream: s,
	}
	if setkeepalive {
		keepsalivebuffer(handle, newstream)
	}

	return newstream, Status(y).error("(*Handle).GetStream")
}
