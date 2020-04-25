package gocu

import (
	"runtime"
)

//Device is a cuda device that can be set on the host thread
//Major is the major compute capability
//Minor is the minor compute capability
type Device interface {
	Set() error
}

//Worker works on functions on a device
type Worker struct {
	w       chan (func() error)
	errChan chan error
	d       Device
	err     error
}

//NewWorker creates a worker that works on a single host thread
func NewWorker(d Device) (w *Worker) {
	w = new(Worker)
	w.d = d
	w.errChan = make(chan error, 2)
	w.w = make(chan (func() error), 2)
	go w.start()
	return w
}

//start locks the host thread and sets
//and dedicates a device to that host thread.
//Functions that use the gpu will use the gpu set by the worker.
//Errors are returned through the error channel.
func (w *Worker) start() {
	runtime.LockOSThread()
	if w.d != nil {
		w.d.Set()
	}
	for x := range w.w {
		w.errChan <- x()
	}
	runtime.UnlockOSThread()
	return
}

//Work takes a call back function, sends it
//through a channel to a locked thread hosting a gpu.
//
//This function will block until work is done.  You don't have to wait though.
//
// If not wanting to wait.  I would recomend wrapping this around a go func(){}() so that you could pick up the error.
func (w *Worker) Work(fn func() error) error {
	w.w <- fn
	return <-w.errChan
}

//Close closes the worker channel
func (w *Worker) Close() {
	//println("ClosedWorker")
	close(w.w)
}
