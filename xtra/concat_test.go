package xtra

import (
	"fmt"
	"runtime"

	"github.com/dereklstinson/gocudnn/gocu"

	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocudnn/cudart"
	"github.com/dereklstinson/cutil"

	"testing"
)

func TestCreateConcatEx(t *testing.T) {
	runtime.LockOSThread()
	check := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	dev, err := cudart.GetDevice()
	check(err)
	check(dev.Set())
	worker := gocu.NewWorker(dev)
	handle := gocudnn.CreateHandleEX(worker, false)

	s, err := cudart.CreateBlockingStream()
	check(err)
	check(handle.SetStream(s))
	h, err := MakeHandleEx(worker, true)
	check(err)
	alpha := float64(1)
	beta := float64(0)

	op, err := CreateConcatEx(h)
	check(err)
	offset := 1
	srcs := 3
	srcds := make([]*gocudnn.TensorD, srcs)
	srcmem := make([]cutil.Mem, srcs)
	hostmembytes := make([][]float32, srcs)  //This checks if the forward and backward work
	copybackbytes := make([][]float32, srcs) //This checks if the forward and backward work
	var frmt gocudnn.TensorFormat
	var dtype gocudnn.DataType
	frmtflg := frmt
	frmt.NCHW()
	dtype.Float()
	nbatch, hheight, width := int32(3), int32(3), int32(3)
	memmanger, err := cudart.CreateMemManager(worker)

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
		bytesize, err := srcds[i].GetSizeInBytes()
		check(err)
		hostmembytes[i] = make([]float32, bytesize/4)
		copybackbytes[i] = make([]float32, bytesize/4)
		quckcopy, err := cutil.WrapGoMem(hostmembytes[i])
		check(err)
		memmanger.Copy(quckcopy, srcmem[i], bytesize)
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
	check(op.Op(h, srcds, srcmem, alpha, destd, destmem, beta, true))
	println("Done with Forward")

	check(err)
	println("sync")
	check(s.Sync())
	println("done sync")
	println("copy")
	//var mflg cudart.MemcpyKind
	check(memmanger.Copy(hostptr, destmem, sib))
	println("done copy")
	//	t.Error(hostmem)
	/*for i := range srcds {
		check(gocudnn.SetTensor(handle, srcds[i], srcmem[i], float64(0)))

	}*/
	check(op.Op(h, srcds, srcmem, alpha, destd, destmem, beta, false))
	for i := range srcds {
		sibback, err := srcds[i].GetSizeInBytes()
		check(err)
		quickcopy, err := cutil.WrapGoMem(copybackbytes[i])
		check(err)
		check(memmanger.Copy(quickcopy, srcmem[i], sibback))
	}
	for i := range copybackbytes {
		for j := range copybackbytes[i] {
			if copybackbytes[i][j] != hostmembytes[i][j] {
				t.Error(copybackbytes[i], hostmembytes[i])
				break
			}
		}

	}

}
