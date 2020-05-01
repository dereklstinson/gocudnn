package cudart

/*
#include <cuda_runtime_api.h>
*/
import "C"
import "runtime"
import "github.com/dereklstinson/gocudnn/gocu"

//Event is a cuda event
type Event struct {
	x C.cudaEvent_t
}

//CreateEvent will create and return an Event
func CreateEvent() (event *Event, err error) {
	e := new(Event)
	err = newErrorRuntime("CreateEvent", C.cudaEventCreate(&e.x))
	if err != nil {
		return nil, err
	}

	runtime.SetFinalizer(e, destroyevent)
	return e, err
}

//Record records an event
func (e *Event) Record(s gocu.Streamer) error {

	return newErrorRuntime("Record", C.cudaEventRecord(e.x, C.cudaStream_t(s.Ptr())))
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
	return false, newErrorRuntime("Status", x)
}

//Sync waits for an event to complete
func (e *Event) Sync() error {
	return newErrorRuntime("Sync", C.cudaEventSynchronize(e.x))
}

func destroyevent(e *Event) error {
	return newErrorRuntime("destroy", C.cudaEventDestroy(e.x))
}

//ElapsedTime takes the current event and compares it to a previous event and returns the time difference.
//in ms
func (e *Event) ElapsedTime(previous *Event) (float32, error) {
	var time C.float
	err := newErrorRuntime("Elapsed Time", C.cudaEventElapsedTime(&time, previous.x, e.x))

	return float32(time), err

}
