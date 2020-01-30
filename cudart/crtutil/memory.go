package crtutil

import (
	"io"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/cutil"
)

//ReadWriter is made to work with the golang io packages
type ReadWriter struct {
	p    unsafe.Pointer
	i    uint
	size uint
	//	asnyc       bool
	//	dev         cudart.Device
	//	w           *gocudnn.Worker
	//	hackbuffer  []byte
	//	hackpointer *cutil.Wrapper
	//	hackflag    bool
	s *cudart.Stream
}

//Allocator alocates memory to the current device
type Allocator struct {
	//dev cudart.Device
	//w   *gocudnn.Worker
	s *cudart.Stream
}

//const usehackbuffer = false

/*
func (a *Allocator) GetWorker() (w *gocudnn.Worker) {
	return a.w
}
*/

//CreateAsyncAllocator creates an allocator whose memory it creates does async mem copies.
func CreateAsyncAllocator(s *cudart.Stream) (a *Allocator) {
	a = new(Allocator)
	//var err error
	//	a.w = gocudnn.NewWorker(dev)
	//	err = a.w.Work(func() error {
	//		a.s, err = cudart.CreateNonBlockingStream()
	//		return err
	//
	//	})
	//
	//	if err != nil {
	//		panic(err)
	//	}

	a.s = s
	return a
}

//CreateDeviceAllocator creates a device allocator whose memory will use the regular mem copy.
func CreateDeviceAllocator() (a *Allocator) {
	a = new(Allocator)
	//a.w = gocudnn.NewWorker(dev)
	//a.dev = dev

	return a
}

//AllocateReadWriterGlobal allocates memory on the current device
func (a *Allocator) AllocateReadWriterGlobal(size uint) (r *ReadWriter, err error) {
	r = new(ReadWriter)

	r.size = size
	//	r.dev = a.dev
	//	err = a.w.Work(
	//		func() error {
	//return cudart.MallocManagedGlobal(r, size)
	cudart.MallocManagedGlobal(r, size)
	//
	//		})
	//	if err != nil {
	//		return nil, err
	//	}
	r.s = a.s
	//	r.w = a.w
	return r, nil
}

/*
func AllocateReadWriterGlobal(size uint) (r *ReadWriter, err error) {
	r = new(ReadWriter)
	err = cudart.MallocManagedGlobal(r, size)
	r.size = size
	return r, err
}
func AllocateReadWriterHost(size uint) (r *ReadWriter, err error) {
	r = new(ReadWriter)
	err = cudart.MallocManagedHost(r, size)
	r.size = size
	return r, err
}
*/

//NewReadWriter returns ReadWriter from already allocated memory passed in p.  It just needs to know the size of the memory.
//It will use the regular mem copy.
func NewReadWriter(p cutil.Pointer, size uint) *ReadWriter {
	return &ReadWriter{
		p:    p.Ptr(),
		size: size,
	}
}

//NewReadWriterAsync returns ReadWriter from already allocated memory passed in p.  It just needs to know the size of the memory.
//It will use the async mem copy.
func NewReadWriterAsync(p cutil.Pointer, size uint, s *cudart.Stream) *ReadWriter {
	return &ReadWriter{
		p:    p.Ptr(),
		size: size,
		s:    s,
	}
}

//Ptr satisfies cutil.Pointer interface.
func (r *ReadWriter) Ptr() unsafe.Pointer {
	return r.p
}

//DPtr satisfies cutil.DPointer interface.
func (r *ReadWriter) DPtr() *unsafe.Pointer {
	return &r.p
}

//Reset resets the index to 0.
func (r *ReadWriter) Reset() {
	r.i = 0
	//	r.hackflag = false
	//	r.hackbuffer = nil
}

//Len returns the remaining bytes that are not read.
func (r *ReadWriter) Len() int {
	if r.i >= r.size {
		return 0
	}
	return (int)(r.size) - (int)(r.i)
}

//Size returns the total size in bytes of the memory the readwriter holds
func (r *ReadWriter) Size() uint {
	return r.size
}

var copyflag cudart.MemcpyKind

//hackread was created because there was trouble getting ioutil.Readall to work with this.
//func (r *ReadWriter) hackread(b []byte) (n int, err error) {
//	if r.i >= r.size {
//		r.Reset()
//		println("hit reset")
//		return 0, io.EOF
//	}
//	if len(b) == 0 {
//		return 0, nil
//	}
//	if r.hackflag == false {
//		r.hackflag = true
//		r.hackbuffer = make([]byte, r.size)
//		r.hackpointer, err = cutil.WrapGoMem(r.hackbuffer)
//		if err != nil {
//			return 0, err
//		}
//		err = cudart.MemCpy(r.hackpointer, r, r.size, copyflag.Default())
//	}
//
//	n = copy(b, r.hackbuffer[r.i:])
//	r.i += uint(n)
//	if len(b) == int(r.size) {
//		r.Reset()
//		return n, io.EOF
//	}
//	return n, nil
//}

func (r *ReadWriter) Read(b []byte) (n int, err error) {
	//if usehackbuffer {
	//	return r.hackread(b)
	//}
	return r.nonhackbuffer(b)

}
func (r *ReadWriter) nonhackbuffer(b []byte) (n int, err error) {
	if r.i >= r.size {
		r.Reset()
		println("hit reset")
		return 0, io.EOF
	}
	if len(b) == 0 {
		return 0, nil
	}
	var size = r.size - r.i
	if uint(len(b)) < size {
		size = uint(len(b))
	}
	bwrap, err := cutil.WrapGoMem(b)
	if err != nil {
		return 0, err
	}
	//	err = r.w.Work(
	//		func() error {
	//			if r.asnyc {
	//				err2 := cudart.MemcpyAsync(bwrap, cutil.Offset(r, r.i), size, copyflag.Default(), r.s)
	//				if err2 != nil {
	//					return err2
	//				}
	//				return nil
	//			} else {
	//				err2 := cudart.MemCpy(bwrap, cutil.Offset(r, r.i), size, copyflag.Default())
	//				if err2 != nil {
	//					return err2
	//				}
	//				return nil
	//			}
	//		})
	if r.s != nil {
		err = cudart.MemcpyAsync(bwrap, cutil.Offset(r, r.i), size, copyflag.Default(), r.s)
	} else {
		err = cudart.MemCpy(bwrap, cutil.Offset(r, r.i), size, copyflag.Default())
	}

	if err != nil {
		return 0, nil
	}

	r.i += size
	n = int(size)
	if len(b) == int(r.size) {
		r.Reset()
		return n, io.EOF
	}
	return n, nil
}

/*
func (r *ReadWriter) ReadAt(b []byte, off int64) (n int, err error) {
	if off < 0 {
		return 0, errors.New("crtutil.ReadWriter.ReadAt: negative offset")
	}
	if off >= int64(r.size) {
		return 0, io.EOF
	}
	cudart.MemCpy()
}
*/
