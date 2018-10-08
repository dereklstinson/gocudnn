package tests

import (
	"fmt"
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

func TestReshape(t *testing.T) {
	gocudnn.Cuda{}.LockHostThread()
	trainingkernellocation := "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/"
	var cu gocudnn.Cuda
	dev, err := cu.GetDeviceList()

	if err != nil {
		t.Error(err)
	}

	err = dev[0].Set()
	if err != nil {
		t.Error(err)
	}
	//	handle := gocudnn.NewHandle()
	stream, err := cu.CreateBlockingStream()
	if err != nil {
		t.Error(err)
	}
	//	err = handle.SetStream(stream)
	if err != nil {
		t.Error(err)
	}
	xhandle, err := gocudnn.Xtra{}.MakeXHandle(trainingkernellocation, dev[0])
	if err != nil {
		t.Error(err)
	}
	xhandle.SetStream(stream)
	X, Xmem, err := testTensorFloat4dNCHW([]int32{1, 10, 25, 25})
	if err != nil {
		t.Error(err)
	}
	Xhmem, err := goptrtest([]int32{1, 10, 25, 25})
	if err != nil {
		t.Error(err)
	}
	err = gocudnn.CudaMemCopy(Xmem, Xhmem, Xmem.ByteSize(), gocudnn.MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		t.Error(err)
	}
	Y, err := gocudnn.Xtra{}.FindSegmentedOutputTensor(X, 6, 6)
	if err != nil {
		t.Error(err)
	}
	Ysize, err := Y.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}
	Ymem, err := gocudnn.Malloc(Ysize)
	if err != nil {
		t.Error(err)
	}
	err = gocudnn.Xtra{}.SegmentedBatches1CHWtoNCHWForward(xhandle, X, Xmem, Y, Ymem)
	if err != nil {
		t.Error(err)
	}
	fmt.Println(xhandle, X, Y)
}
func goptrtest(dims []int32) (*gocudnn.GoPointer, error) {
	params := parammaker(dims)
	array := make([]float32, params)
	for i := int32(0); i < params; i++ {
		array[i] = float32(i)
	}
	return gocudnn.MakeGoPointer(array)
}
func parammaker(dims []int32) int32 {
	mult := int32(1)
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	return mult
}
