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

//CreateHandle creates a handle its basically a Context
//usegogc is for future use.  Right now it is always on the gc.
//
//This function initializes the cuDNN library and creates a handle to an opaque structure holding the cuDNN library context.
//It allocates hardware resources on the host and device and must be called prior to making any other cuDNN library calls.
//
//The cuDNN library handle is tied to the current CUDA device (context).
//To use the library on multiple devices, one cuDNN handle needs to be created for each device.
//
//For a given device, multiple cuDNN handles with different configurations (e.g., different current CUDA streams) may be created.
//Because cudnnCreate allocates some internal resources, the release of those resources by calling cudnnDestroy will implicitly call cudaDeviceSynchronize;
//therefore, the recommended best practice is to call cudnnCreate/cudnnDestroy outside of performance-critical code paths.
//
//For multithreaded applications that use the same device from different threads, the recommended programming model is to create one
//(or a few, as is convenient) cuDNN handle(s) per thread and use that cuDNN handle for the entire life of the thread.
func CreateHandle(usegogc bool) *Handle {

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
