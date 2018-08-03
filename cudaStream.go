package gocudnn

/*
#include <cudnn.h>
#include <cuda_runtime_api.h>
*/
import "C"
import "errors"

//Stream holds a C.cudaStream_t
type Stream struct {
	stream C.cudaStream_t
}

//CreateBlockingStream creats an asyncronus stream stream for the user
func CreateBlockingStream() (*Stream, error) {
	var s Stream
	err := s.create(false, false, 0)
	return &s, err
}

//CreateNonBlockingStream creates a blocking stream
func CreateNonBlockingStream() (*Stream, error) {
	var s Stream
	err := s.create(true, false, 0)
	return &s, err

}
func CreateNonBlockingPriorityStream(priority int32) (*Stream, error) {
	var s Stream
	err := s.create(true, true, priority)
	return &s, err

}
func CreateBlockingPriorityStream(priority int32) (*Stream, error) {
	var s Stream
	err := s.create(true, true, priority)
	return &s, err

}

//Sync Syncronizes the stream
func (s *Stream) Sync() error {
	return newErrorRuntime("Sync", C.cudaStreamSynchronize(s.stream))
}

//Create creates a Stream
func (s *Stream) create(blocking, priority bool, rank int32) error {
	if blocking == true && priority == false {
		x := C.cudaStreamCreateWithFlags(&s.stream, C.cudaStreamDefault)
		return newErrorRuntime("cudaStreamCreate", x)
	}
	if blocking == false && priority == false {
		x := C.cudaStreamCreateWithFlags(&s.stream, C.cudaStreamNonBlocking)
		return newErrorRuntime("cudaStreamCreate", x)
	}
	if blocking == true && priority == true {
		x := C.cudaStreamCreateWithPriority(&s.stream, C.cudaStreamDefault, C.int(rank))
		return newErrorRuntime("cudaStreamCreate", x)
	}
	if blocking == false && priority == true {
		x := C.cudaStreamCreateWithPriority(&s.stream, C.cudaStreamNonBlocking, C.int(rank))
		return newErrorRuntime("cudaStreamCreate", x)
	}
	return errors.New("CreateStream: Unreachable: How did this Happen")
}
func destroystream(s *Stream) error {
	return newErrorRuntime("Destroy", C.cudaStreamDestroy(s.stream))
}

//Destroy destroys the stream
func (s *Stream) Destroy() error {
	return destroystream(s)
}
