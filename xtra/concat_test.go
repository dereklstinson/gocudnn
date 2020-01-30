package xtra

import (
	"fmt"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/cutil"
	"runtime"

	"testing"
)

func TestCreateConcatEx(t *testing.T) {
	runtime.LockOSThread()
	handle := gocudnn.CreateHandle(false)
	check := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	s, err := cudart.CreateBlockingStream()
	check(err)
	check(handle.SetStream(s))
	dev, err := cudart.GetDevice()
	check(err)
	check(dev.Set())

	h, err := MakeHandle(dev, true)
	check(err)
	op, err := CreateConcatEx(h)
	check(err)
	offset := 12
	srcds := make([]*gocudnn.TensorD, 5)
	srcmem := make([]cutil.Mem, 5)

	var frmt gocudnn.TensorFormat
	var dtype gocudnn.DataType
	frmtflg := frmt
	frmt.NCHW()
	dtype.Float()
	nbatch, hheight, width := int32(3), int32(12), int32(12)
	memmanger, err := cudart.CreateMemManager(s, dev)

	check(err)
	for i := range srcds {
		var shape []int32
		if frmtflg.NCHW() == frmt {
			shape = []int32{nbatch, int32(i + offset), hheight, width}
		} else {
			shape = []int32{nbatch, hheight, width, int32(i + offset)}
		}

		srcds[i], err = gocudnn.CreateTensorDescriptor()
		check(err)

		check(srcds[i].Set(frmt, dtype, shape, nil))
		sib, err := srcds[i].GetSizeInBytes()
		check(err)
		srcmem[i], err = memmanger.Malloc(sib)
		fmt.Println("Src SIB", sib)
		fmt.Println("Src Dims", shape)
		check(err)
		check(gocudnn.SetTensor(handle, srcds[i], srcmem[i], float64(i+1)))
		/*	var vol = int32(1)
			for i := range shape {
				vol *= shape[i]
			}
			hostmem := make([]float32, vol)
			hostptr, err := cutil.WrapGoMem(hostmem)
			check(memmanger.Copy(hostptr, srcmem[i], sib))
			fmt.Println(hostmem)
		*/
	}

	outputdims, err := op.GetOutputdims(srcds)
	fmt.Println(outputdims)
	check(err)
	destd, err := gocudnn.CreateTensorDescriptor()
	check(err)
	check(destd.Set(frmt, dtype, outputdims, nil))
	sib, err := destd.GetSizeInBytes()
	check(err)
	fmt.Println("Destsib", sib)
	destmem, err := memmanger.Malloc(sib)
	check(err)
	var vol = int32(1)
	for i := range outputdims {
		vol *= outputdims[i]
	}

	hostmem := make([]float32, vol)
	hostptr, err := cutil.WrapGoMem(hostmem)
	fmt.Println("Destsib", sib)
	fmt.Println("hostmemlen", len(hostmem))
	println("copy")
	//var mflg cudart.MemcpyKind
	check(memmanger.Copy(destmem, hostptr, sib))
	println("done copy")

	check(s.Sync())
	println("Doing Forward")
	check(op.Op(h, srcds, srcmem, destd, destmem, true))
	println("Done with Forward")

	check(err)
	println("sync")
	check(s.Sync())
	println("done sync")
	println("copy")
	//var mflg cudart.MemcpyKind
	check(memmanger.Copy(hostptr, destmem, sib))
	println("done copy")
	t.Error(hostmem)
}
