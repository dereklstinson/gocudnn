package gocudnn

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/dereklstinson/gocudnn/cudart"

	"github.com/dereklstinson/gocudnn/gocu"
)

func TestCreateSoftMaxDescriptor(t *testing.T) {
	runtime.LockOSThread()

	var smmode SoftMaxMode
	var smalgo SoftMaxAlgorithm
	var dtype DataType
	var frmt TensorFormat
	smmode.Channel()
	smalgo.Accurate()
	dtype.Float()
	frmt.NHWC()
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

	err = smd.Set(smalgo, smmode)
	if err != nil {
		t.Error(err)
	}
	xD, err := CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}
	yD, err := CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}

	var dims []int32
	var xvals []float32
	flg := frmt
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
	err = xD.Set(frmt, dtype, dims, nil)
	if err != nil {
		t.Error(err)
	}
	err = yD.Set(frmt, dtype, dims, nil)
	if err != nil {
		t.Error(err)
	}
	x := new(gocu.CudaPtr)
	y := new(gocu.CudaPtr)
	xyDsib, err := xD.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}

	err = cudart.MallocManagedGlobal(x, xyDsib)
	err = cudart.MallocManagedGlobal(y, xyDsib)
	err = cudart.Memcpy(x, xvptr, xyDsib, crtmcpykind)

	err = smd.Forward(h, 1.0, xD, x, 0.0, yD, y)
	if err != nil {
		t.Error(err)
	}
	stream.Sync()
	for i := range xvals {
		xvals[i] = 0
	}
	err = cudart.Memcpy(yvptr, y, xyDsib, crtmcpykind)
	if err != nil {
		t.Error(err)
	}
	err = cudart.Memcpy(xvptr, x, xyDsib, crtmcpykind)
	if err != nil {
		t.Error(err)
	}
	err = stream.Sync()
	if err != nil {
		t.Error(err)
	}

	var yadder float32
	for _, yval := range yvals {
		yadder = yadder + yval
	}

	if yadder > 1 {
		t.Error("yadder greater than 4")
		fmt.Println(yvals)
		fmt.Println(xvals)
	}

}
