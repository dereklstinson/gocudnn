package nvjpeg

/*
#include <nvjpeg.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//DecodeParams - decode parameters structure. Used to set decode-related tweaks
type DecodeParams struct {
	d C.nvjpegDecodeParams_t
}

//CreateDecodeParams creates a DecodeParam
func CreateDecodeParams(h *Handle) (d *DecodeParams, err error) {
	d = new(DecodeParams)
	err = status(C.nvjpegDecodeParamsCreate(h.h, &d.d)).error()
	return d, err
}

//SetOutputFormat sets the format for the output
func (d *DecodeParams) SetOutputFormat(format OutputFormat) error {
	return status(C.nvjpegDecodeParamsSetOutputFormat(d.d, format.c())).error()
}

//SetROI - set to desired ROI.  Pass (0, 0, -1, -1) to disable ROI decode (decode whole image)
func (d *DecodeParams) SetROI(offsetX, offsetY, roiW, roiH int32) error {
	return status(C.nvjpegDecodeParamsSetROI(d.d, (C.int)(offsetX), (C.int)(offsetY), (C.int)(roiW), (C.int)(roiH))).error()
}

//SetCMYK - set to true to allow conversion from CMYK to RGB or YUV that follows simple subtractive scheme
func (d *DecodeParams) SetCMYK(set bool) error {
	if set {
		return status(C.nvjpegDecodeParamsSetAllowCMYK(d.d, 1)).error()
	}
	return status(C.nvjpegDecodeParamsSetAllowCMYK(d.d, 0)).error()

}

//JpegDecoder is used for decoding images
type JpegDecoder struct {
	d C.nvjpegJpegDecoder_t
}

//CreateDecoder - creates decoder implementation
func CreateDecoder(h *Handle, implementation Backend) (decoder *JpegDecoder, err error) {
	decoder = new(JpegDecoder)
	err = status(C.nvjpegDecoderCreate(h.h, implementation.c(), &decoder.d)).error()
	runtime.SetFinalizer(decoder, nvjpegDecoderDestroy)
	return decoder, err
}

func nvjpegDecoderDestroy(d *JpegDecoder) error {
	return status(C.nvjpegDecoderDestroy(d.d)).error()
}

//JpegSupported if supported == true then decoder is capable to handle jpeg sream with the specified params
func (d *JpegDecoder) JpegSupported(s *JpegStream, param *DecodeParams) (supported bool, err error) {
	var is C.int
	err = status(C.nvjpegDecoderJpegSupported(d.d, s.s, param.d, &is)).error()
	if is == 0 {
		supported = true
	}
	return supported, err
}

func CreateDecoderState(h *Handle, decoder *JpegDecoder) (state *JpegState, err error) {
	state = new(JpegState)
	err = status(C.nvjpegDecoderStateCreate(h.h, decoder.d, &state.j)).error()
	return state, err

}

//DecodeHost - starts decoding on host and save decode parameters to the state
func (j *JpegDecoder) DecodeHost(h *Handle, state *JpegState, param *DecodeParams, stream *JpegStream) (err error) {
	return status(C.nvjpegDecodeJpegHost(h.h, j.d, state.j, param.d, stream.s)).error()
}

//TransferToDevice - hybrid stage of decoding image,  involves device async calls
//note that jpeg stream is a parameter here - because we still might need copy
//parts of bytestream to device
func (j *JpegDecoder) TransferToDevice(h *Handle, state *JpegState, jpgstream *JpegStream, s gocu.Streamer) error {
	return status(C.nvjpegDecodeJpegTransferToDevice(h.h, j.d, state.j, jpgstream.s, (C.cudaStream_t)(s.Ptr()))).error()
}
func (j *JpegDecoder) DecodeDevice(h *Handle, state *JpegState, dest *Image, s gocu.Streamer) (err error) {
	return status(C.nvjpegDecodeJpegDevice(h.h, j.d, state.j, dest.cptr(), (C.cudaStream_t)(s.Ptr()))).error()
}
