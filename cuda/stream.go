package cuda

//#include <cuda.h>
import "C"
import (
	"runtime"
)

//Stream is a queue for gpu execution
type Stream struct {
	s C.CUstream
}

//CreateStream creates a stream using the default stream creation flag.
func CreateStream() (s *Stream, err error) {
	s = new(Stream)
	err = newErrorDriver("CreateStream()",
		C.cuStreamCreate(&s.s, C.CU_STREAM_DEFAULT))
	if err != nil {
		return nil, err
	}
	runtime.SetFinalizer(s, cuStreamDestroy)
	return s, err
}

//CreateNonBlockingStream - Creates a steam that work running on this stream may run concurrently
//with work in stream 0 (the NULL stream), and that the created stream should perform no implicit
//synchronization with stream 0.
func CreateNonBlockingStream() (s *Stream, err error) {
	s = new(Stream)
	err = newErrorDriver("CreateNonBlockingStream()",
		C.cuStreamCreate(&s.s, C.CU_STREAM_NON_BLOCKING))
	if err != nil {
		return nil, err
	}
	runtime.SetFinalizer(s, cuStreamDestroy)
	return s, err
}

//CreateStreamWithPriority - Using the default settings stream creation flag.  This API alters the scheduler priority of work in the stream.
//Work in a higher priority stream may preempt work already executing in a low priority stream.
//The lower the number the higher the priority.
func CreateStreamWithPriority(priority int32) (s *Stream, err error) {
	s = new(Stream)
	err = newErrorDriver("CreateStreamWithPriority()",
		C.cuStreamCreateWithPriority(&s.s, C.CU_STREAM_DEFAULT,
			(C.int)(priority)))
	if err != nil {
		return nil, err
	}
	runtime.SetFinalizer(s, cuStreamDestroy)
	return s, err
}

//CreateNonBlockingStreamWithPriority - Using the Nonblocking stream creation flag.  This API alters the scheduler priority of work in the stream.
//Work in a higher priority stream may preempt work already executing in a low priority stream.
//The lower the number the higher the priority.
func CreateNonBlockingStreamWithPriority(priority int32) (s *Stream, err error) {
	s = new(Stream)
	err = newErrorDriver("CreateNonBlockingStreamWithPriority()",
		C.cuStreamCreateWithPriority(&s.s, C.CU_STREAM_NON_BLOCKING,
			(C.int)(priority)))
	if err != nil {
		return nil, err
	}
	runtime.SetFinalizer(s, cuStreamDestroy)
	return s, err
}
func cuStreamDestroy(s *Stream) error {
	return newErrorDriver("cuStreamDestroy ()", C.cuStreamDestroy(s.s))

}

//GetPriority gets the stream priority
func (s *Stream) GetPriority() (priority int32, err error) {
	err = newErrorDriver("(s *Stream) GetPriority()",
		C.cuStreamGetPriority(s.s, (*C.int)(&priority)))
	return priority, err
}

//Sync - Wait until a stream's tasks are completed.
func (s *Stream) Sync() (err error) {
	return newErrorDriver("(s *Stream)Sync() ()",
		C.cuStreamSynchronize(s.s))
}
