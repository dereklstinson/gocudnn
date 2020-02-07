//Package crtutil allows cudart to work with Go's io Reader and Writer interfaces.
//
//This package only works with devices that have Compute Capability 6.1 and up.
package crtutil

import (
	"errors"
	"io"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/cutil"
)

//ReadWriter is made to work with the golang io packages
type ReadWriter struct {
	p    unsafe.Pointer
	i    uint
	size uint
	s    gocu.Streamer
}

//Allocator alocates memory to the current device
type Allocator struct {
	s gocu.Streamer
}

//CreateAllocator creates an allocator whose memory it creates does async mem copies.
func CreateAllocator(s gocu.Streamer) (a *Allocator) {
	a = new(Allocator)
	a.s = s
	return a
}

//AllocateMemory allocates memory on the current device
func (a *Allocator) AllocateMemory(size uint) (r *ReadWriter, err error) {
	r = new(ReadWriter)
	r.size = size
	cudart.MallocManagedGlobal(r, size)
	r.s = a.s
	return r, nil
}

//NewReadWriter returns ReadWriter from already allocated memory passed in p.  It just needs to know the size of the memory.
//If s is nil. Then it will do a non async copy.  If it is not nil then it will do a async copy.
func NewReadWriter(p cutil.Pointer, size uint, s gocu.Streamer) *ReadWriter {
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

func (r *ReadWriter) Read(b []byte) (n int, err error) {
	//if usehackbuffer {
	//	return r.hackread(b)
	//}
	return r.nonhackbuffer(b)

}
func (r *ReadWriter) nonhackbuffer(b []byte) (n int, err error) {
	if r.i >= r.size {
		r.Reset()

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
func (r *ReadWriter) Write(b []byte) (n int, err error) {
	if r.i >= r.size {
		r.Reset()
		return 0, errors.New("Write Location Out of Memory")
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
	if r.s != nil {
		err = cudart.MemcpyAsync(cutil.Offset(r, r.i), bwrap, size, copyflag.Default(), r.s)
	} else {
		err = cudart.MemCpy(cutil.Offset(r, r.i), bwrap, size, copyflag.Default())
	}
	r.i += size
	n = int(size)
	return n, err
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
