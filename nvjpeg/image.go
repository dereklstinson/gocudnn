package nvjpeg

/*
#include <nvjpeg.h>
#include <cuda_runtime_api.h>
*/
import "C"
import (
	"errors"
	"unsafe"

	"github.com/dereklstinson/gocudnn/gocu"
)

//Image is of type C.nvjpegImage_t.
type Image C.nvjpegImage_t

//GetImageInfo gets the image info
//Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
//If less than NVJPEG_MAX_COMPONENT channels are encoded, then zeros would be set to absent channels information
//If the image is 3-channel, all three groups are valid.
//This function is thread safe.
//
//-IN         handle      : Library handle
//
//-IN         data        : Pointer to the buffer containing the jpeg stream data to be decoded
//
//-Return     nComponent  : Number of componenets of the image, currently only supports 1-channel (grayscale) or 3-channel
//
//-Return     subsampling : Chroma subsampling used in this JPEG
//
//-Return     w    		: pointer to NVJPEG_MAX_COMPONENT of ints, returns width of each channel
//
//-Return     h  		   : pointer to NVJPEG_MAX_COMPONENT of ints, returns height of each channel
//
func GetImageInfo(handle *Handle, data []byte) (subsampling ChromaSubsampling, w []int32, h []int32, err error) {
	length := len(data)
	var ncomp C.int
	var sub C.nvjpegChromaSubsampling_t
	w = make([]int32, C.NVJPEG_MAX_COMPONENT)
	h = make([]int32, C.NVJPEG_MAX_COMPONENT)
	if len(data) < 1 || data == nil {
		return 0, nil, nil, errors.New("data len is 0 or nil")
	}
	err = status(C.nvjpegGetImageInfo(
		handle.h,
		(*C.uchar)(&data[0]),
		C.size_t(length),
		&ncomp,
		&sub,
		(*C.int)(&w[0]),
		(*C.int)(&h[0]))).error()
	subsampling = ChromaSubsampling(sub)

	return subsampling, w[:ncomp], h[:ncomp], err
}

//Get gets the underlying values of image
//
//Get will return the channels that were decoded.
func (im *Image) Get() (channel []*gocu.CudaPtr, pitch []uint32) {

	for i := 0; i < int(C.NVJPEG_MAX_COMPONENT); i++ {
		if im.channel[i] != nil || im.pitch[i] != 0 {
			channel = append(channel, gocu.WrapUnsafe((unsafe.Pointer)(im.channel[i])))
			pitch = append(pitch, (uint32)(im.pitch[i]))

		}
	}
	return channel, pitch
}
func (im *Image) cptr() *C.nvjpegImage_t {
	return (*C.nvjpegImage_t)(im)
}
func (im *Image) c() C.nvjpegImage_t {
	return (C.nvjpegImage_t)(*im)
}

//ChannelDimHelper will return the dims for each channel depending on the ouputformat
func ChannelDimHelper(o OutputFormat, pitch, height []int32) (w, h []int32) {
	var flg OutputFormat
	switch o {
	case flg.Y():
		h := make([]int32, 1)
		w = make([]int32, 1)
		w[0] = pitch[0]
		h[0] = height[0]
		return w, h

	case flg.YUV():
		h := make([]int32, 3)
		w = make([]int32, 3)
		for i := 0; i < 3; i++ {

			w[i] = pitch[i]
			h[i] = height[i]

		}
		return w, h
	case flg.RGB():

		h := make([]int32, 3)
		w = make([]int32, 3)
		for i := 0; i < 3; i++ {

			w[i] = pitch[0]
			h[i] = height[0]

		}
		return w, h
	case flg.BGR():

		h := make([]int32, 3)
		w = make([]int32, 3)
		for i := 0; i < 3; i++ {

			w[i] = pitch[0]
			h[i] = height[0]

		}
		return w, h
	case flg.RGBI():
		h := make([]int32, 1)
		w = make([]int32, 1)
		w[0] = pitch[0] * 3
		h[0] = height[0]
		return w, h

	case flg.BGRI():
		h := make([]int32, 1)
		w = make([]int32, 1)
		w[0] = pitch[0] * 3
		h[0] = height[0]
		return w, h

	case flg.Unchanged():

		cmp := (int)(len(pitch))
		h := make([]int32, cmp)
		w = make([]int32, cmp)
		for i := 0; i < cmp; i++ {
			h[i] = height[i]
			w[i] = pitch[i]
		}
		return h, w
	}
	return nil, nil
}

//CreateImageDest allocates cuda memory of an empty Image for Decode methods.
//
//The size of pitch needs to be at least the size of the width.
//
//BGRI and RGBI are not offically supported because didn't give a c I marked them in the table with a question mark
//
//This is not an official nvjpeg function.  I made this just for you. Maybe me too, but I didn't have to put it here with the bindings. So, I guess I could say I put it in here for you.
//
//	-FORMAT				|SIZE OF PITCH							|SIZE OF CHANNEL
//	-Y				|width[0] for c = 0       					|pitch[0]*height[0] for c = 0
//	-YUV				|width[c] for c = 0, 1, 2					|pitch[c]*height[c] for c = 0, 1, 2
//	-RGB				|width[0] for c = 0, 1, 2					|pitch[0]*height[0] for c = 0, 1, 2
//	-BGR				|width[0] for c = 0, 1, 2					|pitch[0]*height[0] for c = 0, 1, 2
//	-RGBI				|width[0]*3	  c=?         					|pitch[0]*height[0] c=?
//	-BGRI  				|width[0]*3	  c=?         					|pitch[0]*height[0] c=?
//	-Unchanged			|width[c] for c = [ 0, nComponents - 1 ]			|pitch[c]*height[c] for c = [ 0, nComponents - 1]
//
func CreateImageDest(o OutputFormat, pitch, height []int32, allocator gocu.Allocator) (*Image, error) {
	nComponents := int32(len(pitch))
	var flg OutputFormat
	img := new(Image)
	switch o {
	case flg.Y():
		size := pitch[0] * height[0]
		ptr, err := allocator.Malloc((uint)(size))
		if err != nil {
			return nil, err
		}
		img.pitch[0] = (C.uint)(pitch[0])
		img.channel[0] = (*C.uchar)(ptr.Ptr())
		return img, nil

	case flg.YUV():
		var size int32
		for i := 0; i < 3; i++ {
			size = pitch[i] * height[i]
			ptr, err := allocator.Malloc((uint)(size))
			if err != nil {
				return nil, err
			}
			img.pitch[i] = (C.uint)(pitch[i])
			img.channel[i] = (*C.uchar)(ptr.Ptr())
		}
		return img, nil
	case flg.RGB():
		var size int32
		for i := 0; i < 3; i++ {
			size = pitch[0] * height[0]
			ptr, err := allocator.Malloc((uint)(size))
			if err != nil {
				return nil, err
			}
			img.pitch[i] = (C.uint)(pitch[0])
			img.channel[i] = (*C.uchar)(ptr.Ptr())
		}
		return img, nil
	case flg.BGR():
		var size int32
		for i := 0; i < 3; i++ {
			size = pitch[0] * height[0]
			ptr, err := allocator.Malloc((uint)(size))
			if err != nil {
				return nil, err
			}
			img.pitch[i] = (C.uint)(pitch[0])
			img.channel[i] = (*C.uchar)(ptr.Ptr())
		}
		return img, nil
	case flg.RGBI():

		size := pitch[0] * height[0] * 3
		ptr, err := allocator.Malloc((uint)(size))
		if err != nil {
			return nil, err
		}
		img.pitch[0] = (C.uint)(pitch[0] * 3)
		img.channel[0] = (*C.uchar)(ptr.Ptr())
		return img, nil

	case flg.BGRI():
		size := pitch[0] * height[0] * 3
		ptr, err := allocator.Malloc((uint)(size))
		if err != nil {
			return nil, err
		}
		img.pitch[0] = (C.uint)(pitch[0] * 3)
		img.channel[0] = (*C.uchar)(ptr.Ptr())
		return img, nil

	case flg.Unchanged():
		var size int32
		cmp := (int)(nComponents)
		for i := 0; i < cmp; i++ {
			size = pitch[i] * height[i]
			ptr, err := allocator.Malloc((uint)(size))
			if err != nil {
				return nil, err
			}
			img.pitch[i] = (C.uint)(pitch[i])
			img.channel[i] = (*C.uchar)(ptr.Ptr())
		}
		return img, nil
	}
	return nil, errors.New("Not supported output format")
}
