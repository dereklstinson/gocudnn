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
	close(w.w)
}
