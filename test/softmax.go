package main

import (
	"fmt"
	"runtime"

	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocudnn/cudart"
	"github.com/dereklstinson/gocudnn/gocu"
)

//softmax
func main() {
	runtime.LockOSThread()
	var smmode gocudnn.SoftMaxMode
	var smalgo gocudnn.SoftMaxAlgorithm
	var dtype gocudnn.DataType
	var frmt gocudnn.TensorFormat
	smmode.Channel()
	smalgo.Accurate()
	dtype.Float()
	frmt.NHWC()
	var crtmcpykind cudart.MemcpyKind
	crtmcpykind.Default()
	h := gocudnn.CreateHandle(true)
	//	file, err := os.Create("softmaxcallback")
	//	if err != nil {
	//		panic(err)
	//	}
	//	defer file.Close()
	//err = gocudnn.SetCallBack(nil, file)
	//if err != nil {
	//	panic(err)
	//}
	stream, err := cudart.CreateBlockingStream()
	if err != nil {
		panic(err)
	}
	err = h.SetStream(stream)
	if err != nil {
		panic(err)
	}
	smd := gocudnn.CreateSoftMaxDescriptor()

	err = smd.Set(smalgo, smmode)
	if err != nil {
		panic(err)
	}
	xD, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		panic(err)
	}
	yD, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		panic(err)
	}

	var dims []int32
	var xvals []float32
	flg := frmt
	switch frmt {
	case flg.NHWC():
		dims = []int32{1, 1, 1, 3}
		xvals = []float32{1, -1, -2}
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
		panic(err)
	}

	yvptr, err := gocu.MakeGoMem(yvals)
	if err != nil {
		panic(err)
	}
	err = xD.Set(frmt, dtype, dims, nil)
	if err != nil {
		panic(err)
	}
	err = yD.Set(frmt, dtype, dims, nil)
	if err != nil {
		panic(err)
	}
	x := new(gocu.CudaPtr)
	y := new(gocu.CudaPtr)
	xyDsib, err := xD.GetSizeInBytes()
	if err != nil {
		panic(err)
	}

	err = cudart.MallocManagedGlobal(x, xyDsib)
	err = cudart.MallocManagedGlobal(y, xyDsib)
	err = cudart.MemCpy(x, xvptr, xyDsib, crtmcpykind)

	err = smd.Forward(h, 1.0, xD, x, 0.0, yD, y)
	if err != nil {
		panic(err)
	}
	stream.Sync()
	for i := range xvals {
		xvals[i] = 0
	}
	err = cudart.MemCpy(yvptr, y, xyDsib, crtmcpykind)
	if err != nil {
		panic(err)
	}
	err = cudart.MemCpy(xvptr, x, xyDsib, crtmcpykind)
	if err != nil {
		panic(err)
	}
	err = stream.Sync()
	if err != nil {
		panic(err)
	}

	var yadder float32
	for _, yval := range yvals {
		yadder = yadder + yval
	}
	fmt.Println("Added Channels: ", yadder)
	inputstringer, err := gocudnn.GetStringer(xD, x)
	outputstringer, err := gocudnn.GetStringer(yD, y)
	if err != nil {
		panic(err)
	}
	fmt.Println("Input", inputstringer)
	fmt.Println("Output", outputstringer)
}
