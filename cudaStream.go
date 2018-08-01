package gocudnn

/*
#include <cudnn.h>
#include <cuda_runtime_api.h>
*/
import "C"

//Stream holds a C.cudaStream_t
type Stream struct {
	stream C.cudaStream_t
}

//CreateStream creats a stream for the user
func CreateStream() (*Stream, error) {
	var s Stream
	err := s.create()
	return &s, err
}

//Sync Syncronizes the stream
func (s *Stream) Sync() error {
	return newErrorRuntime("Sync", C.cudaStreamSynchronize(s.stream))
}

//Create creates a Stream
func (s *Stream) create() error {
	x := C.cudaStreamCreate(&s.stream)
	return newErrorRuntime("cudaStreamCreate", x)

}
func destroystream(s *Stream) error {
	return newErrorRuntime("Destroy", C.cudaStreamDestroy(s.stream))
}

//Destroy destroys the stream
func (s *Stream) Destroy() error {
	return destroystream(s)
}
