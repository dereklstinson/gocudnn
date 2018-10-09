package tests

import (
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//THIS IS FAILING

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
	originaldims := []int32{1, 32, 32, 32}
	descX, Xmem, err := testTensorFloat4dNCHW(originaldims)
	if err != nil {
		t.Error(err)
	}
	Xhmem, err := goptrtest(originaldims)
	if err != nil {
		t.Error(err)
	}
	err = gocudnn.CudaMemCopy(Xmem, Xhmem, Xmem.ByteSize(), gocudnn.MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		t.Error(err)
	}
	descY, err := gocudnn.Xtra{}.FindSegmentedOutputTensor(descX, 4, 4)
	if err != nil {
		t.Error(err)
	}
	Ysize, err := descY.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}
	Ymem, err := gocudnn.Malloc(Ysize)
	if err != nil {
		t.Error(err)
	}

	err = gocudnn.Xtra{}.SegmentedBatches1CHWtoNCHWForward(xhandle, descX, Xmem, descY, Ymem)
	if err != nil {
		t.Error(err)
	}
	err = stream.Sync()
	if err != nil {
		t.Error(err)
	}
	_, dimsY, _, err := descY.GetDescrptor()
	if err != nil {
		t.Error(err)
	}
	yslice := make([]float32, parammaker(dimsY))
	yptr, err := gocudnn.MakeGoPointer(yslice)
	if err != nil {
		t.Error(err)
	}

	err = gocudnn.CudaMemCopy(yptr, Ymem, Ymem.ByteSize(), gocudnn.MemcpyKindFlag{}.DeviceToHost())
	if err != nil {
		t.Error(err)
	}
	t.Error(yslice)
	//fmt.Println(yslice)
}
func goptrtest(dims []int32) (*gocudnn.GoPointer, error) {
	params := parammaker(dims)
	array := make([]float32, params)
	for i := int32(0); i < dims[0]; i++ {
		for j := int32(0); j < dims[1]; j++ {
			for k := int32(0); j < dims[2]; k++ {
				for l := int32(0); l < dims[3]; l++ {
					array[i] = float32(i)
				}
			}
		}

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
