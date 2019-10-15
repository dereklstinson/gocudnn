package nvjpeg

/*
#include <nvjpeg.h>
*/
import "C"
import "runtime"

//JpegStream - handle that stores stream information metadata, encoded image parameters, encoded stream parameters
//stores everything on CPU side. This allows us parse header separately from implementation
//and retrieve more information on the stream. Also can be used for transcoding and transfering
//metadata to encoder
type JpegStream struct {
	s C.nvjpegJpegStream_t
}

//CreateJpegStream creates a Jpeg Stream
func CreateJpegStream(h *Handle) (stream *JpegStream, err error) {
	stream = new(JpegStream)
	err = status(C.nvjpegJpegStreamCreate(h.h, &stream.s)).error()
	runtime.SetFinalizer(stream, nvjpegJpegStreamDestroy)
	return stream, err
}
func nvjpegJpegStreamDestroy(s *JpegStream) error {
	return status(C.nvjpegJpegStreamDestroy(s.s)).error()
}
func (s *JpegStream) Parse(h *Handle, data []byte, saveMetaData, saveStream bool) error {
	var smd C.int
	var sst C.int
	if saveMetaData {
		smd = 1
	}
	if saveStream {
		sst = 1
	}
	length := (C.size_t)(len(data))
	return status(C.nvjpegJpegStreamParse(h.h, (*C.uchar)(&data[0]), length, smd, sst, s.s)).error()
}
func (s *JpegStream) GetFrameDimensions() (width, height uint32, err error) {
	err = status(C.nvjpegJpegStreamGetFrameDimensions(s.s, (*C.uint)(&width), (*C.uint)(&height))).error()
	return width, height, err
}
func (s *JpegStream) GetComponentsNum() (num uint32, err error) {
	err = status(C.nvjpegJpegStreamGetComponentsNum(s.s, (*C.uint)(&num))).error()
	return num, err
}
func (s *JpegStream) GetComponentDims(componenet uint32) (width, height uint32, err error) {
	err = status(C.nvjpegJpegStreamGetComponentDimensions(s.s, (C.uint)(componenet), (*C.uint)(&width), (*C.uint)(&height))).error()
	return width, height, err
}
func (s *JpegStream) GetChromaSubsampling() (subsamp ChromaSubsampling, err error) {
	err = status(C.nvjpegJpegStreamGetChromaSubsampling(s.s, subsamp.cptr())).error()
	return subsamp, err
}
