package gocudnn_test

import (
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

func TestCudaMemory(t *testing.T) {
	somemem, err := gocudnn.Malloc(64 * 4)
	if err != nil {
		t.Error(err)
	}
	float := make([]float32, 32)
	for i := range float {
		float[i] = float32(i)
	}
	gomem, err := gocudnn.MakeGoPointer(float)
	if err != nil {
		t.Error(err)
	}
	err = gocudnn.CudaMemCopy(somemem, gomem, 64*4, gocudnn.MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		t.Error(err)
	}
	readgpumem := make([]float32, 64)
	gomem2, err := gocudnn.MakeGoPointer(readgpumem)
	err = gocudnn.CudaMemCopy(gomem2, somemem, 64*4, gocudnn.MemcpyKindFlag{}.DeviceToHost())
	if err != nil {
		t.Error(err)
	}

}
