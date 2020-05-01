package main

import (
	"fmt"
	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocudnn/cudart"
	"github.com/dereklstinson/gocudnn/gocu"
	"github.com/dereklstinson/gocudnn/xtra"
	"github.com/dereklstinson/cutil"
	"runtime"
	"time"
)

func main() {
	runtime.LockOSThread()
	check := func(err error) {
		if err != nil {
			panic(err)
		}
	}
	dev := cudart.CreateDevice(1)
	//dev, err := cudart.GetDevice()
	//check(err)
	check(dev.Set())
	worker := gocu.NewWorker(dev)

	handle := gocudnn.CreateHandleEX(worker, false)

	s, err := cudart.CreateNonBlockingStream()
	check(err)
	check(handle.SetStream(s))
	h, err := xtra.MakeHandleEx(worker, true)
	check(err)
	check(h.SetStream(s))
	alpha := float64(1)
	beta := float64(0)

	op, err := xtra.CreateConcatEx(h)

	check(err)
	offset := 4
	nsrcs := 300
	srcds := make([]*gocudnn.TensorD, nsrcs)
	srcmem := make([]cutil.Mem, nsrcs)
	hostmembytes := make([][]float32, nsrcs)  //This checks if the forward and backward work
	copybackbytes := make([][]float32, nsrcs) //This checks if the forward and backward work
	var frmt gocudnn.TensorFormat
	var dtype gocudnn.DataType
	frmtflg := frmt
	frmt.NCHW()
	dtype.Float()
	nbatch, hheight, width := int32(20), int32(64), int32(64)
	memmanger, err := cudart.CreateMemManager(worker)

	check(err)
	for i := range srcds {
		var shape []int32
		if frmtflg.NCHW() == frmt {
			shape = []int32{nbatch, int32(offset), hheight, width}
		} else {
			shape = []int32{nbatch, hheight, width, int32(offset)}
		}

		srcds[i], err = gocudnn.CreateTensorDescriptor()
		check(err)

		check(srcds[i].Set(frmt, dtype, shape, nil))
		sib, err := srcds[i].GetSizeInBytes()
		check(err)
		srcmem[i], err = memmanger.Malloc(sib)
		//fmt.Println("Src SIB", sib)
		//fmt.Println("Src Dims", shape)
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
	totalconcats := 20
	for i := 0; i < totalconcats; i++ {

		println("Doing Forward")
		fwdtime := time.Now()
		check(op.Op(h, srcds, srcmem, alpha, destd, destmem, beta, true))
		println("Done with Forward, TIME(ms):", time.Now().Sub(fwdtime).Seconds()*1000.0)

		//println("copy")
		if i > totalconcats-1 {
			check(memmanger.Copy(hostptr, destmem, sib))
		}

		//	println("done copy")
		bwdtime := time.Now()
		check(op.Op(h, srcds, srcmem, alpha, destd, destmem, beta, false))
		println("Done with Backward, TIME(ms):", time.Now().Sub(bwdtime).Seconds()*1000.0)
	}
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
				fmt.Println("Don't Match", copybackbytes[i][j], hostmembytes[i][j])
				panic("DontMatch")

			}
		}

	}
	fmt.Println("Everything chekcs out")

}
