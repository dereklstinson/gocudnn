package gocudnn

//Host thread is a sample interface with the cudnn library.  Since
import (
	"runtime"
)

//Host is the cpu
type Host struct {
	Threads []HostThread
	threads int
	closed  int
	done    chan int
	quit    chan int
}

type cudachan struct {
	thread  int
	handle  int
	memeroy Memer
}

//HostThread holds handles
type HostThread struct {
	handle Handle
}

/*

func (h *Host)HostController(numberofthreads int,f func(<-chan cudachan, chan<- cudachan, <-chan int, chan<- cudachan) error){
	h.Threads=make([]HostThread, numberofthreads)
	srcchan := make(chan cudachan, numberofthreads)
	destchan :=make( chan cudachan, numberofthreads)
	controllerchan :=make(chan cudachan,numberofthreads)

	for i:=0;i<numberofthreads;i++{
h.Threads.RunHostThread()
	}

}

*/

//RunHostThread is a sample host thread
func (thread *HostThread) RunHostThread(
	src <-chan cudachan,
	controller chan<- cudachan,
	dest chan<- cudachan,
	quit <-chan int,
	f func(<-chan cudachan, chan<- cudachan, <-chan int, chan<- cudachan) error,
) <-chan error {

	threadquit := make(chan int, 1)
	outputchan := make(chan cudachan, 1)
	err := make(chan error, 1)
	go func() {
		runtime.LockOSThread()
		err <- f(src, dest, threadquit, outputchan)

	}()

	for {
		select {
		case x := <-src:
			dest <- x
		case x := <-outputchan:
			controller <- x
		case x := <-quit:
			threadquit <- x

		case x := <-err:
			err <- x
			runtime.UnlockOSThread()
			return err
		}

	}

}

//Closer closes program
func (host *Host) Closer() {
	for {
		select {
		case <-host.done:
			host.closed++
			if host.closed >= host.threads {
				return
			}

		default:
			if host.closed >= host.threads {
				return
			}

		}

	}
}
