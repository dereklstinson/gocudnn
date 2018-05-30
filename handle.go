package cudnn

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
	return unsafe.Pointer(handle.x)
}

//CreateHandle creates a handle its basically a Context
func NewHandle() *Handle {
	var handle Handle
	//	var handle C.cudnnHandle_t
	err := Status(C.cudnnCreate(&handle.x)).error("NewHandle")
	if err != nil {
		panic(err)
	}
	runtime.SetFinalizer(&handle, destroyhandle)
	return &handle
}
func destroyhandle(handle *Handle) {
	C.cudnnDestroy(handle.x)
}

//Create creates the handle
func (handle *Handle) create() error {

	y := C.cudnnCreate(&handle.x)

	return Status(y).error("(*Handle).Create")
}

//Destroy destroys the handle
func (handle *Handle) Destroy() error {

	y := C.cudnnDestroy(handle.x)

	return Status(y).error("(*Handle).Destroy")
}

//typedefed CUstream_st *CUstream
//typedefed CUstream_st *cudastream_t

//SetStream passes a stream to sent in the cuda handle
func (handle *Handle) SetStream(s *Stream) error {

	y := C.cudnnSetStream(handle.x, s.stream)

	return Status(y).error("(*Handle).SetStream")
}

//GetStream will return a stream that the handle is using
func (handle *Handle) GetStream() (Stream, error) {
	var s Stream
	var some *C.cudaStream_t
	x := C.cudnnHandle_t(handle.Pointer())

	y := C.cudnnGetStream(x, some)
	s.stream = *some
	return s, Status(y).error("(*Handle).GetStream")
}
