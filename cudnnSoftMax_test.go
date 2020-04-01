package gocudnn

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/dereklstinson/GoCudnn/cudart"

	"github.com/dereklstinson/GoCudnn/gocu"
)

func TestCreateSoftMaxDescriptor(t *testing.T) {
	runtime.LockOSThread()

	var crtmcpykind cudart.MemcpyKind
	crtmcpykind.Default()
	h := CreateHandle(true)
	stream, err := cudart.CreateBlockingStream()
	if err != nil {
		t.Error(err)
	}
	err = h.SetStream(stream)
	if err != nil {
		t.Error(err)
	}
	smd := CreateSoftMaxDescriptor()
	var smmode SoftMaxMode
	var smalgo SoftMaxAlgorithm
	smmode.Channel()
	smalgo.Accurate()
	err = smd.Set(smalgo, smmode)
	if err != nil {
		t.Error(err)
	}
	xyD, err := CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}
	var dtype DataType
	var frmt TensorFormat
	dtype.Float()
	flg := frmt
	frmt.NHWC()
	var dims []int32
	var xvals []float32
	switch frmt {
	case flg.NHWC():
		dims = []int32{1, 1, 1, 3}
		xvals = []float32{
			1, -1, -2, //2, -1, -2,
			//3, -1, -2 , 4, -1, -2,
		} //[1 -1 -2 2 -1 -2 3 -1 -2 4 -1 -2]
	case flg.NCHW():
		dims = []int32{1, 3, 2, 2}
		xvals = []float32{
			//C0
			4, 0,
			0, 0,
			//C1
			0, 4,
			0, 0,
			//C2
			0, 0,
			0, 4}
	}

	yvals := make([]float32, len(xvals))
	xvptr, err := gocu.MakeGoMem(xvals)
	if err != nil {
		t.Error(err)
	}

	yvptr, err := gocu.MakeGoMem(yvals)
	if err != nil {
		t.Error(err)
	}
	err = xyD.Set(frmt, dtype, dims, nil)
	if err != nil {
		t.Error(err)
	}
	x := new(gocu.CudaPtr)
	y := new(gocu.CudaPtr)
	xyDsib, err := xyD.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}

	err = cudart.MallocManagedGlobal(x, xyDsib)
	err = cudart.MallocManagedGlobal(y, xyDsib)
	err = cudart.MemCpy(x, xvptr, xyDsib, crtmcpykind)

	err = smd.Forward(h, 1.0, xyD, x, 0.0, xyD, y)
	if err != nil {
		t.Error(err)
	}
	stream.Sync()
	for i := range xvals {
		xvals[i] = 0
	}
	err = cudart.MemCpy(yvptr, y, xyDsib, crtmcpykind)
	if err != nil {
		t.Error(err)
	}
	err = cudart.MemCpy(xvptr, x, xyDsib, crtmcpykind)
	if err != nil {
		t.Error(err)
	}
	stream.Sync()
	t.Error("Look at output")
	fmt.Println(yvals)
	fmt.Println(xvals)
}
