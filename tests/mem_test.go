package tests

import (
	"testing"

	"github.com/dereklstinson/GoCudnn"
)

func TestMem(t *testing.T) {
	array := testarray(1024)

	hostmem, err := gocudnn.MakeGoPointer(array)
	if err != nil {
		t.Error(err)
	}
	size := hostmem.ByteSize()
	var memmanager gocudnn.ManagedMemFlag
	managedmem, err := gocudnn.MallocManaged(size, memmanager.Global())
	if err != nil {
		t.Error(err)
	}
	var memflag gocudnn.MemcpyKindFlag

	err = gocudnn.CudaMemCopy(managedmem, hostmem, size, memflag.Default())
	if err != nil {
		t.Error(err)
	}

	cudahost, err := gocudnn.MallocHost(size)
	if err != nil {
		t.Error("MallocHost:", err)
	}
	err = gocudnn.CudaMemCopy(cudahost, managedmem, size, memflag.Default())
	if err != nil {
		t.Error(err)
	}

	err = cudahost.Free()
	if err != nil {
		t.Error("FreeHost", err)
	}
	err = managedmem.Free()
	if err != nil {
		t.Error("FreeHost", err)
	}
	err = hostmem.Free()
	if err != nil {
		t.Error("FreeHost", err)
	}

}

func testarray(size int) []int32 {
	array := make([]int32, size)
	for i := 0; i < size; i++ {
		array[i] = int32(1)
	}
	return array
}
