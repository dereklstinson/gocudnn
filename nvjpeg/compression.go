package nvjpeg

/*
#include<nvjpeg.h>
#include<cuda_runtime_api.h>
*/
import "C"
import (
	"fmt"
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//EncoderParams is a struct that contains paramerters for the encoder
type EncoderParams struct {
	ep C.nvjpegEncoderParams_t
}

//CreateEncoderParams creates an EncoderParams
func CreateEncoderParams(h *Handle, s gocu.Streamer) (*EncoderParams, error) {
	ep := new(EncoderParams)
	err := status(C.nvjpegEncoderParamsCreate(h.h, &ep.ep, stream(s)))
	runtime.SetFinalizer(ep, nvjpegEncoderParamsDestroy)
	return ep, err
}

func nvjpegEncoderParamsDestroy(ep *EncoderParams) error {
	err := status(C.nvjpegEncoderParamsDestroy(ep.ep)).error()
	if err != nil {
		return err
	}
	ep = nil
	return nil
}

//SetQuality sets the quality of the encoder params
//Quality should be a number between 1 and 100. The default is set to 70
func (ep *EncoderParams) SetQuality(quality int32, s gocu.Streamer) error {
	return status(C.nvjpegEncoderParamsSetQuality(ep.ep, C.int(quality), stream(s)))
}

//SetOptimizedHuffman - Sets whether or not to use optimized Huffman.
//Using optimized Huffman produces smaller JPEG bitstream sizes with
//the same quality, but with slower performance.
//Default is false
func (ep *EncoderParams) SetOptimizedHuffman(optimized bool, s gocu.Streamer) error {
	var opt C.int
	if optimized {
		opt = 1
	}
	return status(C.nvjpegEncoderParamsSetOptimizedHuffman(ep.ep, opt, stream(s)))
}

//SetSamplingFactors -Sets which chroma subsampling will be used for JPEG compression.
//ssfactor that will be used for JPEG compression.
//If the input is in YUV color model and ssfactor is different from the subsampling factors
//of source image, then the NVJPEG library will convert subsampling to the value of chroma_subsampling.
// Default value is 4:4:4.
func (ep *EncoderParams) SetSamplingFactors(ssfactor ChromaSubsampling, s gocu.Streamer) error {
	return status(C.nvjpegEncoderParamsSetSamplingFactors(ep.ep, ssfactor.c(), stream(s)))
}

//GetBufferSize - Returns the maximum possible buffer size that is needed to store the
//compressed JPEG stream, for the given input parameters.
func (ep *EncoderParams) GetBufferSize(h *Handle, width, height int32) (maxStreamLength uint, err error) {
	var msl C.size_t
	err = status(C.nvjpegEncodeGetBufferSize(h.h, ep.ep, (C.int)(width), (C.int)(height), &msl))
	maxStreamLength = (uint)(msl)
	return maxStreamLength, err
}

//EncoderState is used for encoding functions in nvjpeg
type EncoderState struct {
	es C.nvjpegEncoderState_t
}

//CreateEncoderState creates an EncoderState
func CreateEncoderState(h *Handle, s gocu.Streamer) (*EncoderState, error) {
	es := new(EncoderState)
	err := status(C.nvjpegEncoderStateCreate(h.h, &es.es, stream(s)))
	runtime.SetFinalizer(es, nvjpegEncoderStateDestroy)
	return es, err
}

func nvjpegEncoderStateDestroy(es *EncoderState) error {
	err := status(C.nvjpegEncoderStateDestroy(es.es)).error()
	if err != nil {
		return err
	}
	es = nil
	return nil
}

//EncodeYUV -Compresses the image in YUV colorspace to JPEG stream using the provided parameters,
// and stores it in the state structure.
func (es *EncoderState) EncodeYUV(
	h *Handle,
	ep *EncoderParams,
	src *Image,
	srcsampling ChromaSubsampling,
	swidth int32,
	sheight int32,
	s gocu.Streamer) error {
	return status(C.nvjpegEncodeYUV(
		h.h,
		es.es,
		ep.ep,
		src.cptr(),
		srcsampling.c(),
		(C.int)(swidth),
		(C.int)(sheight),
		stream(s),
	))
}

//EncodeImage - Compresses the image in the provided format to the JPEG stream
//using the provided parameters, and stores it in the state structure
func (es *EncoderState) EncodeImage(
	h *Handle,
	ep *EncoderParams,
	src *Image,
	fmt InputFormat,
	swidth int32,
	sheight int32,
	s gocu.Streamer) error {
	return status(C.nvjpegEncodeImage(
		h.h,
		es.es,
		ep.ep,
		src.cptr(),
		fmt.c(),
		(C.int)(swidth),
		(C.int)(sheight),
		stream(s),
	))
}

//
// nvjpegEncodeRetrieveBitstream does a ton of things and I am going to break it up into
// seperate functions I think will be handy in a go environment.
//

//RetrieveBitStream does GetCompressedBufferSize makes a slice of bytes from that, and
//runs ReadBitStream and returns the bytes filled.
func (es *EncoderState) RetrieveBitStream(h *Handle, s gocu.Streamer) ([]byte, error) {
	size, err := es.GetCompressedBufferSize(h, s)
	if err != nil {
		return nil, err
	}
	data := make([]byte, size)
	_, err = es.ReadBitStream(h, data, s)
	if err != nil {
		return nil, err
	}
	return data, nil
}

//GetCompressedBufferSize returns the length of the compressed stream
func (es *EncoderState) GetCompressedBufferSize(h *Handle, s gocu.Streamer) (uint, error) {
	var length C.size_t
	err := status(C.nvjpegEncodeRetrieveBitstream(h.h, es.es, nil, &length, stream(s))).error()
	return (uint)(length), err
}

//ReadBitStream reads the compressed bit stream and puts it into p.
//Method returns the n the number of bytes written into p. if len(p) is less than the bit stream.
//nothing will be written and an error will be returned.
func (es *EncoderState) ReadBitStream(h *Handle, p []byte, s gocu.Streamer) (n int, err error) {
	length := C.size_t(len(p))
	err = status(C.nvjpegEncodeRetrieveBitstream(h.h, es.es, (*C.uchar)(&p[0]), &length, stream(s)))
	if err != nil {
		msg := err.Error()
		lenact, err2 := es.GetCompressedBufferSize(h, s)
		if err2 != nil {
			suberr := err2.Error()
			return 0, fmt.Errorf("Double Error one in Reading int p (%s), and next finding length (%s)", msg, suberr)
		}
		return 0, fmt.Errorf("len of p passed %d, it should be %d. From Cuda(%s)", length, lenact, msg)
	}
	return int(length), nil
}
