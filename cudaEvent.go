package gocudnn

/*
#include <cudnn.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
*/
import "C"
import (
	"runtime"
)

//Event is a cuda event
type Event struct {
	x C.cudaEvent_t
}

//CreateEvent will create and return an Event
func (cu Cuda) CreateEvent() (event *Event, err error) {
	var e Event
	err = newErrorRuntime("CreateEvent", C.cudaEventCreate(&e.x))
	if err != nil {
		return nil, err
	}
	event = &e
	if setfinalizer {
		runtime.SetFinalizer(event, destroyevent)
	}
	return event, err
}
func (e *Event) keepsalive() {
	runtime.KeepAlive(e)
}

//Record records an event
func (e *Event) Record(s *Stream) error {
	if setkeepalive {
		e.keepsalive()
	}
	return newErrorRuntime("Record", C.cudaEventRecord(e.x, s.stream))
}

//Status is the function cudaEventQuery. I didn't like the name and how the function was handled.
//error will returned as nil if cudaSuccess and cudaErrorNotReady are returned. It will return a 1 of event is completed.
//It will return a 0 if event is not complete
func (e *Event) Status() (bool, error) {

	x := C.cudaEventQuery(e.x)
	if x == C.cudaSuccess {
		return true, nil
	}
	if x == C.cudaErrorNotReady {
		return false, nil
	}
	if setkeepalive {
		e.keepsalive()
	}
	return false, newErrorRuntime("Status", x)

}

//Sync waits for an event to complete
func (e *Event) Sync() error {
	return newErrorRuntime("Sync", C.cudaEventSynchronize(e.x))
}

//Destroy destroys an event!
func (e *Event) Destroy() error {
	return destroyevent(e)
}

func destroyevent(e *Event) error {
	return newErrorRuntime("destroy", C.cudaEventDestroy(e.x))
}

//ElapsedTime takes the current event and compares it to a previous event and returns the time difference.
//in ms
func (e *Event) ElapsedTime(previous *Event) (float32, error) {
	var time C.float
	err := newErrorRuntime("Elapsed Time", C.cudaEventElapsedTime(&time, previous.x, e.x))
	if setkeepalive {
		e.keepsalive()
	}
	return float32(time), err

}
