package nvjpeg

/*
#include <nvjpeg.h>
#include <cuda_runtime_api.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//Handle - Opaque library handle identifier.
type Handle struct {
	h                 C.nvjpegHandle_t
	expinnedallocator *C.nvjpegPinnedAllocator_t
	exdevallocator    *C.nvjpegDevAllocator_t
}

func nvjpegDestroy(h *Handle) error {
	err := status(C.nvjpegDestroy(h.h)).error()
	if err != nil {
		return err
	}
	h = nil
	return nil
}

/*CreateSimple creates an nvjpeg handle with default backend and default memory allocators.

Returns    handle        : Codec instance, use for other calls
*/
func CreateSimple() (*Handle, error) {
	h := new(Handle)
	err := status(C.nvjpegCreateSimple(&h.h)).error()
	if err != nil {
		return nil, err
	}
	runtime.SetFinalizer(h, nvjpegDestroy)
	return h, nil
}

//SetDeviceMemoryPadding -Use the provided padding for all device memory allocations with specified library handle.
// A large number will help to amortize the need for device memory reallocations when needed.
func (h *Handle) SetDeviceMemoryPadding(padding uint) error {
	return status(C.nvjpegSetDeviceMemoryPadding(C.size_t(padding), h.h)).error()
}

//GetDeviceMemoryPadding - Retrieve the device memory padding that is currently used for the specified library handle.
func (h *Handle) GetDeviceMemoryPadding() (uint, error) {
	var size C.size_t
	err := status(C.nvjpegGetDeviceMemoryPadding(&size, h.h)).error()
	return (uint)(size), err
}

//SetPinnedMemoryPadding -Use the provided padding for all pinned host memory allocations with specified library handle.
//A large number will help to amortize the need for pinned host memory reallocations when needed.
func (h *Handle) SetPinnedMemoryPadding(padding uint) error {
	return status(C.nvjpegSetPinnedMemoryPadding(C.size_t(padding), h.h)).error()
}

//GetPinnedMemoryPadding -Retrieve the pinned host memory padding that is currently used for specified library handle.
func (h *Handle) GetPinnedMemoryPadding() (uint, error) {
	var size C.size_t
	err := status(C.nvjpegGetPinnedMemoryPadding(&size, h.h)).error()
	return (uint)(size), err
}

func stream(s gocu.Streamer) C.cudaStream_t {
	return C.cudaStream_t(s.Ptr())
}
