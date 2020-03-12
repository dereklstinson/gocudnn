package gocu

import (
	"runtime"
)

//Device is a cuda device that can be set on the host thread
type Device interface {
	Set() error
}

//Worker works on functions on a device
type Worker struct {
	w       chan (func() error)
	errChan chan error
}

//NewWorker creates a worker that works on a single host thread
func NewWorker(d Device) (w *Worker) {
	w = new(Worker)

	w.errChan = make(chan error, 1)
	w.w = make(chan (func() error), 1)
	go w.start(d)
	return w
}
func (w *Worker) start(d Device) {
	runtime.LockOSThread()
	if d != nil {
		d.Set()
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
	close(w.w)
}
