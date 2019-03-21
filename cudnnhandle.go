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
	x    C.cudnnHandle_t
	gogc bool
}

//Pointer is a pointer to the handle
func (handle *Handle) Pointer() unsafe.Pointer {

	return unsafe.Pointer(handle.x)
}

//NewHandle creates a handle its basically a Context
//usegogc is for future use.  Right now it is always on the gc.
func NewHandle(usegogc bool) *Handle {

	handle := new(Handle)
	err := Status(C.cudnnCreate(&handle.x)).error("NewHandle")
	if err != nil {
		panic(err)
	}

	if setfinalizer {
		handle.gogc = true
		runtime.SetFinalizer(handle, destroycudnnhandle)
	} else {
		if usegogc {
			handle.gogc = true
			runtime.SetFinalizer(handle, destroycudnnhandle)
		}
	}

	return handle
}

//Destroy destroys the handle if GC is being use it won't do anything.
func (handle *Handle) Destroy() error {
	if setfinalizer || handle.gogc {
		return nil
	}
	return destroycudnnhandle(handle)
}
func destroycudnnhandle(handle *Handle) error {
	return Status(C.cudnnDestroy(handle.x)).error("(*Handle).Destroy")
}

//SetStream passes a stream to sent in the cuda handle
func (handle *Handle) SetStream(s gocu.Streamer) error {

	y := C.cudnnSetStream(handle.x, C.cudaStream_t(s.Ptr()))

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
