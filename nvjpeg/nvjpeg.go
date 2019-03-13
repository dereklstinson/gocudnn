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

//Image is of type C.nvjpegImage_t.
type Image C.nvjpegImage_t

//Get gets the underlying values of image
func (im *Image) Get() (channel [C.NVJPEG_MAX_COMPONENT]*byte, pitch [C.NVJPEG_MAX_COMPONENT]uint32) {
	for i := 0; i < int(C.NVJPEG_MAX_COMPONENT); i++ {
		channel[i] = (*byte)(im.channel[i])
		pitch[i] = (uint32)(im.pitch[i])
	}
	return channel, pitch
}
func (im *Image) cptr() *C.nvjpegImage_t {
	return (*C.nvjpegImage_t)(im)
}
func (im *Image) c() C.nvjpegImage_t {
	return (C.nvjpegImage_t)(*im)
}

//Handle - Opaque library handle identifier.
type Handle struct {
	h C.nvjpegHandle_t
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
	return h, err
}

//SetDeviceMemoryPadding -Use the provided padding for all device memory allocations with specified library handle.
// A large number will help to amortize the need for device memory reallocations when needed.
func (h *Handle) SetDeviceMemoryPadding(padding uint) error {
	return status(C.nvjpegSetDeviceMemoryPadding(C.size_t(padding), h.h))
}

//GetDeviceMemoryPadding - Retrieve the device memory padding that is currently used for the specified library handle.
func (h *Handle) GetDeviceMemoryPadding() (uint, error) {
	var size C.size_t
	err := status(C.nvjpegGetDeviceMemoryPadding(&size, h.h))
	return (uint)(size), err
}

//SetPinnedMemoryPadding -Use the provided padding for all pinned host memory allocations with specified library handle.
//A large number will help to amortize the need for pinned host memory reallocations when needed.
func (h *Handle) SetPinnedMemoryPadding(padding uint) error {
	return status(C.nvjpegSetPinnedMemoryPadding(C.size_t(padding), h.h))
}

//GetPinnedMemoryPadding -Retrieve the pinned host memory padding that is currently used for specified library handle.
func (h *Handle) GetPinnedMemoryPadding() (uint, error) {
	var size C.size_t
	err := status(C.nvjpegGetPinnedMemoryPadding(&size, h.h))
	return (uint)(size), err
}

// GetImageInfo gets the image info
// Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
// If less than NVJPEG_MAX_COMPONENT channels are encoded, then zeros would be set to absent channels information
// If the image is 3-channel, all three groups are valid.
// This function is thread safe.
// IN         handle      : Library handle
// IN         data        : Pointer to the buffer containing the jpeg stream data to be decoded.
// IN         length      : Length of the jpeg image buffer.
// Return     nComponent  : Number of componenets of the image, currently only supports 1-channel (grayscale) or 3-channel.
// Return     subsampling : Chroma subsampling used in this JPEG, see nvjpegChromaSubsampling_t
// Return     widths      : pointer to NVJPEG_MAX_COMPONENT of ints, returns width of each channel. 0 if channel is not encoded
// Return     heights     : pointer to NVJPEG_MAX_COMPONENT of ints, returns height of each channel. 0 if channel is not encoded
func GetImageInfo(handle *Handle, data *byte, length uint) (nComponents int32, subsampling ChromaSubsampling, width int32, height int32, err error) {
	var ncomp C.int
	var sub C.nvjpegChromaSubsampling_t
	var w C.int
	var h C.int
	err = status(C.nvjpegGetImageInfo(
		handle.h,
		(*C.uchar)(data),
		C.size_t(length),
		&ncomp,
		&sub,
		&w,
		&h)).error()
	subsampling = ChromaSubsampling(sub)
	width = int32(w)
	height = int32(h)
	nComponents = int32(ncomp)
	return nComponents, subsampling, width, height, err
}

func stream(s gocu.Streamer) C.cudaStream_t {
	return C.cudaStream_t(s.Ptr())
}
