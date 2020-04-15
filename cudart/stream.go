package cudart

/*

#include <cuda_runtime_api.h>


const cudaStream_t gocunullstream = NULL;
*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//Stream holds a C.cudaStream_t
type Stream struct {
	stream C.cudaStream_t
}

//Ptr returns an unsafe pointer to the hidden stream.
//This allows stream to be used with other cuda libraries in other go packages
//so if a C function calls for a Pointer then you can type case the unsafe pointer
//into a (C.cudaStream_t)(unsafe.Pointer)
func (s *Stream) Ptr() unsafe.Pointer {
	if s == nil {
		return unsafe.Pointer(C.cudaStream_t(C.NULL))
	}
	return unsafe.Pointer(s.stream)
}
func (s *Stream) c() C.cudaStream_t {
	if s == nil {
		return C.cudaStream_t(C.NULL)
	}
	return s.stream
}

//AttachMemAsync - Enqueues an operation in stream to specify stream association of length bytes of memory starting from devPtr. This function is a stream-ordered operation, meaning that it is dependent on, and will only take effect when, previous work in stream has completed. Any previous association is automatically replaced.
//
//From Cuda documentation:
//
//devPtr must point to an one of the following types of memories:
//
//managed memory declared using the __managed__ keyword or allocated with cudaMallocManaged.
//
//a valid host-accessible region of system-allocated pageable memory. This type of memory may only be specified if the device associated with the stream reports a non-zero value for the device attribute cudaDevAttrPageableMemoryAccess.
//
//For managed allocations, length must be either zero or the entire allocation's size. Both indicate that the entire allocation's stream association is being changed. Currently, it is not possible to change stream association for a portion of a managed allocation.
//
//For pageable allocations, length must be non-zero.
//
//The stream association is specified using flags which must be one of cudaMemAttachGlobal, cudaMemAttachHost or cudaMemAttachSingle. The default value for flags is cudaMemAttachSingle If the cudaMemAttachGlobal flag is specified, the memory can be accessed by any stream on any device. If the cudaMemAttachHost flag is specified, the program makes a guarantee that it won't access the memory on the device from any stream on a device that has a zero value for the device attribute cudaDevAttrConcurrentManagedAccess. If the cudaMemAttachSingle flag is specified and stream is associated with a device that has a zero value for the device attribute cudaDevAttrConcurrentManagedAccess, the program makes a guarantee that it will only access the memory on the device from stream. It is illegal to attach singly to the NULL stream, because the NULL stream is a virtual global stream and not a specific stream. An error will be returned in this case.
//
//When memory is associated with a single stream, the Unified Memory system will allow CPU access to this memory region so long as all operations in stream have completed, regardless of whether other streams are active. In effect, this constrains exclusive ownership of the managed memory region by an active GPU to per-stream activity instead of whole-GPU activity.
//
//Accessing memory on the device from streams that are not associated with it will produce undefined results. No error checking is performed by the Unified Memory system to ensure that kernels launched into other streams do not access this region.
//
//It is a program's responsibility to order calls to cudaStreamAttachMemAsync via events, synchronization or other means to ensure legal access to memory at all times. Data visibility and coherency will be changed appropriately for all kernels which follow a stream-association change.
//
//If stream is destroyed while data is associated with it, the association is removed and the association reverts to the default visibility of the allocation as specified at cudaMallocManaged. For __managed__ variables, the default association is always cudaMemAttachGlobal. Note that destroying a stream is an asynchronous operation, and as a result, the change to default association won't happen until all work in the stream has completed.
//
func (s *Stream) AttachMemAsync(mem cutil.Pointer, size uint, attachmode MemAttach) error {
	sizet := C.size_t(size)
	if s == nil {
		return newErrorRuntime(" (s *Stream) AttachMemAsync(): ", C.cudaStreamAttachMemAsync(C.gocunullstream, mem.Ptr(), sizet, attachmode.c()))
	}
	return newErrorRuntime(" (s *Stream) AttachMemAsync(): ", C.cudaStreamAttachMemAsync(s.stream, mem.Ptr(), sizet, attachmode.c()))
}

/*
//BeginCapture - notes from cuda documentation
//
//Begin graph capture on stream. When a stream is in capture mode, all operations pushed into the stream will not be executed, but will instead be captured into a graph, which will be returned via cudaStreamEndCapture. Capture may not be initiated if stream is cudaStreamLegacy. Capture must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not already in capture mode. The capture mode may be queried via cudaStreamIsCapturing. A unique id representing the capture sequence may be queried via cudaStreamGetCaptureInfo.
//
//If mode is not cudaStreamCaptureModeRelaxed, cudaStreamEndCapture must be called on this stream from the same thread.
func (s *Stream) BeginCapture(mode StreamCaptureMode) error {

	if s == nil {
		return newErrorRuntime(" (s *Stream) BeginCapture(): ", C.cudaStreamBeginCapture(C.gocunullstream, mode.c()))
	}
	return newErrorRuntime(" (s *Stream) BeginCapture(): ", C.cudaStreamBeginCapture(s.stream, mode.c()))
}
func (s *Stream) EndCapture(mode StreamCaptureMode, g *Graph) error {

	if s == nil {
		return newErrorRuntime(" (s *Stream) BeginCapture(): ", C.cudaStreamEndCapture(C.gocunullstream, &g.c))
	}
	return newErrorRuntime(" (s *Stream) BeginCapture(): ", C.cudaStreamEndCapture(s.stream, &g.c))
}
//CaptureInfo gets the status of the stream capture
func (s *Stream) CaptureInfo() (uniqueid uint64, status StreamCaptureStatus, err error) {
	//var unid C.ulonglong
	if s == nil {
		err = newErrorRuntime(" (s *Stream) CaptureInfo(): ",
			C.cudaStreamGetCaptureInfo(C.gocunullstream,
				status.cptr(),
				(*C.ulonglong)(&uniqueid)))
		return uniqueid, status, err
	}
	err = newErrorRuntime(" (s *Stream) CaptureInfo(): ",
		C.cudaStreamGetCaptureInfo(s.stream,
			status.cptr(),
			(*C.ulonglong)(&uniqueid)))
	return uniqueid, status, err
}
//IsCapturing -Returns a stream's capture status.
func (s *Stream) IsCapturing() (status StreamCaptureStatus, err error) {
	if s == nil {
		err = newErrorRuntime(" (s *Stream) CaptureInfo(): ",
			C.cudaStreamIsCapturing(C.gocunullstream,
				status.cptr()))
		return status, err
	}
	err = newErrorRuntime(" (s *Stream) CaptureInfo(): ",
		C.cudaStreamIsCapturing(s.stream,
			status.cptr()))
	return status, err

}
*/

//Query - Queries an asynchronous stream for completion status.
//
//returns true if ready, false if not.
//
//if an error occures err will not be nil
func (s *Stream) Query() (b bool, err error) {
	var status C.cudaError_t
	if s == nil {
		status = C.cudaStreamQuery(C.gocunullstream)
	} else {
		status = C.cudaStreamQuery(s.stream)
	}
	switch status {
	case C.cudaSuccess:
		return true, nil
	case C.cudaErrorNotReady:
		return false, nil
	default:
		return false, newErrorRuntime("(s *Stream)Query()", status)
	}
}

//WaitEvent - Make a compute stream wait on an event.
//
//Flags must be zero
func (s *Stream) WaitEvent(event *Event, flags uint32) error {
	if s == nil {
		return newErrorRuntime(" (s *Stream) WaitEvent(): ", C.cudaStreamWaitEvent(C.gocunullstream, event.x, C.uint(flags)))
	}
	return newErrorRuntime(" (s *Stream) WaitEvent(): ", C.cudaStreamWaitEvent(s.stream, event.x, C.uint(flags)))
}

//ExternalWrapper is used for other packages that might return a C.cudaStream_t
func ExternalWrapper(x unsafe.Pointer) *Stream {
	return &Stream{
		stream: C.cudaStream_t(x),
	}
}

//CreateBlockingStream creats an asyncronus stream stream for the user
func CreateBlockingStream() (*Stream, error) {
	s := new(Stream)
	err := s.create(false, false, 0)

	runtime.SetFinalizer(s, destroystream)

	return s, err
}

//CreateNonBlockingStream creates a blocking stream
func CreateNonBlockingStream() (*Stream, error) {
	s := new(Stream)
	err := s.create(true, false, 0)

	runtime.SetFinalizer(s, destroystream)

	return s, err

}

//CreateNonBlockingPriorityStream creates a non blocking Priority Stream
func CreateNonBlockingPriorityStream(priority int32) (*Stream, error) {
	s := new(Stream)
	err := s.create(true, true, priority)

	runtime.SetFinalizer(s, destroystream)

	return s, err

}

//CreateBlockingPriorityStream creates a blocking stream
func CreateBlockingPriorityStream(priority int32) (*Stream, error) {
	s := new(Stream)
	err := s.create(true, true, priority)

	runtime.SetFinalizer(s, destroystream)

	return s, err

}

//SyncNillStream will sync the nill stream
func SyncNillStream() error {
	return newErrorRuntime("SyncNillStream()", C.cudaStreamSynchronize(nil))
}

//Sync Syncronizes the stream
func (s *Stream) Sync() error {
	if s == nil {
		return newErrorRuntime("Sync", C.cudaStreamSynchronize(nil))
	}
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

	err := newErrorRuntime("Destroy", C.cudaStreamDestroy(s.stream))
	s = nil
	return err
}

//StreamCaptureMode - Possible modes for stream capture thread interactions
type StreamCaptureMode C.enum_cudaStreamCaptureMode

func (s StreamCaptureMode) c() C.enum_cudaStreamCaptureMode {
	return (C.enum_cudaStreamCaptureMode)(s)
}
func (s *StreamCaptureMode) cptr() *C.enum_cudaStreamCaptureMode {
	return (*C.enum_cudaStreamCaptureMode)(s)
}

//Global sets s to global and returns s
func (s *StreamCaptureMode) Global() StreamCaptureMode {
	*s = (StreamCaptureMode)(C.cudaStreamCaptureModeGlobal)
	return *s
}

//ThreadLocal sets s to ThreadLocal and returns s
func (s *StreamCaptureMode) ThreadLocal() StreamCaptureMode {
	*s = (StreamCaptureMode)(C.cudaStreamCaptureModeThreadLocal)
	return *s
}

//Relaxed sets s to Relaxed and returns s
func (s *StreamCaptureMode) Relaxed() StreamCaptureMode {
	*s = (StreamCaptureMode)(C.cudaStreamCaptureModeRelaxed)
	return *s
}

func (s StreamCaptureMode) String() string {
	var st string
	sf := s
	switch s {
	case sf.Global():
		st = "Global"
	case sf.Relaxed():
		st = "Relaxed"
	case sf.ThreadLocal():
		st = "ThreadLocal"
	default:
		st = "Unsupported Flag"
	}
	return "StreamCaptureMode: " + st

}

//StreamCaptureStatus - Possible stream capture statuses returned by cudaStreamIsCapturing
//Even though this is for returns.  I think this can still be used for switches.
type StreamCaptureStatus C.enum_cudaStreamCaptureStatus

func (s StreamCaptureStatus) c() C.enum_cudaStreamCaptureStatus {
	return (C.enum_cudaStreamCaptureStatus)(s)
}
func (s *StreamCaptureStatus) cptr() *C.enum_cudaStreamCaptureStatus {
	return (*C.enum_cudaStreamCaptureStatus)(s)
}

//None sets s to None and returns s
func (s *StreamCaptureStatus) None() StreamCaptureStatus {
	*s = (StreamCaptureStatus)(C.cudaStreamCaptureStatusNone)
	return *s
}

//Active sets s to Active and returns s
func (s *StreamCaptureStatus) Active() StreamCaptureStatus {
	*s = (StreamCaptureStatus)(C.cudaStreamCaptureStatusActive)
	return *s
}

//Invalid sets s to Invalid and returns s
func (s *StreamCaptureStatus) Invalid() StreamCaptureStatus {
	*s = (StreamCaptureStatus)(C.cudaStreamCaptureStatusInvalidated)
	return *s
}

func (s StreamCaptureStatus) String() string {
	var st string
	sf := s
	switch s {
	case sf.None():
		st = "None"
	case sf.Active():
		st = "Active"
	case sf.Invalid():
		st = "Invalid"
	default:
		st = "Unsupported Flag"
	}
	return "StreamCaptureMode: " + st

}

//MemAttach - This is a new type derived from a list of the defines for cudart
type MemAttach C.uint

func (m MemAttach) c() C.uint {
	return (C.uint)(m)
}
func (m *MemAttach) cptr() *C.uint {
	return (*C.uint)(m)
}

//Global sets m to Global and returns m - Memory can be accessed by any stream on any device
func (m *MemAttach) Global() MemAttach {
	*m = (MemAttach)(C.cudaMemAttachGlobal)
	return *m
}

//Host sets m to Active and returns m - Memory cannot be accessed by any stream on any device
func (m *MemAttach) Host() MemAttach {
	*m = (MemAttach)(C.cudaMemAttachHost)
	return *m
}

//Single sets m to Single and returns m - Memory can only be accessed by a single stream on the associated device
func (m *MemAttach) Single() MemAttach {
	*m = (MemAttach)(C.cudaMemAttachSingle)
	return *m
}
func (m MemAttach) String() string {
	var st string
	sf := m
	switch m {
	case sf.Global():
		st = "Global"
	case sf.Host():
		st = "Host"
	case sf.Single():
		st = "Single"
	default:
		st = "Unsupported Flag"
	}
	return "StreamCaptureMode: " + st

}
