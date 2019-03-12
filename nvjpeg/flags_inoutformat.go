package nvjpeg

/*
#include <nvjpeg.h>

*/
import "C"

//OutputFormat specifies what type of output user wants for image decoding
type OutputFormat C.nvjpegOutputFormat_t

func (o OutputFormat) c() C.nvjpegOutputFormat_t {
	return C.nvjpegOutputFormat_t(o)
}

//Unchanged returns decoded image as it is - write planar output
//Method changes underlying value of the type and returns it.
func (o *OutputFormat) Unchanged() OutputFormat {
	*o = OutputFormat(C.NVJPEG_OUTPUT_UNCHANGED)
	return *o
}

//YUV returns planar luma and chroma
//Method changes underlying value of the type and returns it.
func (o *OutputFormat) YUV() OutputFormat {
	*o = OutputFormat(C.NVJPEG_OUTPUT_YUV)
	return *o
}

//Y return luma component only, write to 1-st channel of nvjpegImage_t
//Method changes underlying value of the type and returns it.
func (o *OutputFormat) Y() OutputFormat {
	*o = OutputFormat(C.NVJPEG_OUTPUT_Y)
	return *o
}

//RGB convert to planar RGB
//Method changes underlying value of the type and returns it.
func (o *OutputFormat) RGB() OutputFormat {
	*o = OutputFormat(C.NVJPEG_OUTPUT_RGB)
	return *o
}

//BGR convert to planar BGR
//Method changes underlying value of the type and returns it.
func (o *OutputFormat) BGR() OutputFormat {
	*o = OutputFormat(C.NVJPEG_OUTPUT_BGR)
	return *o
}

//RGBI convert to interleaved RGB and write to 1-st channel of nvjpegImage_t
//Method changes underlying value of the type and returns it.
func (o *OutputFormat) RGBI() OutputFormat {
	*o = OutputFormat(C.NVJPEG_OUTPUT_RGBI)
	return *o
}

//BGRI convert to interleaved BGR and write to 1-st channel of nvjpegImage_t
//Method changes underlying value of the type and returns it.
func (o *OutputFormat) BGRI() OutputFormat {
	*o = OutputFormat(C.NVJPEG_OUTPUT_BGRI)
	return *o
}

//InputFormat are input format flags for nvjpeg
type InputFormat C.nvjpegInputFormat_t

func (fmt InputFormat) c() C.nvjpegInputFormat_t {
	return C.nvjpegInputFormat_t(fmt)
}

//RGB - Input is RGB - will be converted to YCbCr before encoding
//Method changes the underlying value of the flag and returns it
func (fmt *InputFormat) RGB() InputFormat {
	*fmt = InputFormat(C.NVJPEG_INPUT_RGB)
	return *fmt
}

//BGR -Input is BGR - will be converted to YCbCr before encoding
//Method changes the underlying value of the flag and returns it
func (fmt *InputFormat) BGR() InputFormat {
	*fmt = InputFormat(C.NVJPEG_INPUT_BGR)
	return *fmt
}

//RGBI - Input is interleaved RGB - will be converted to YCbCr before encoding
//Method changes the underlying value of the flag and returns it
func (fmt *InputFormat) RGBI() InputFormat {
	*fmt = InputFormat(C.NVJPEG_INPUT_RGBI)
	return *fmt
}

//BGRI - Input is interleaved BGR - will be converted to YCbCr before encoding
//Method changes the underlying value of the flag and returns it
func (fmt *InputFormat) BGRI() InputFormat {
	*fmt = InputFormat(C.NVJPEG_INPUT_BGRI)
	return *fmt
}
