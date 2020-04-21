package crtutil

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"runtime"
	"testing"

	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/cutil"
)

func TestReadWriter_Read(t *testing.T) {
	runtime.LockOSThread()
	dev, err := cudart.GetDevice()
	if err != nil {
		t.Error(err)
	}
	dev.Set()
	cstream, err := cudart.CreateNonBlockingStream()
	if err != nil {
		t.Error(err)
	}
	arraysize := (uint)(4096 * 4)
	goslice := make([]byte, arraysize)
	for i := range goslice {
		goslice[i] = (byte)((i) % 256)
	}
	gopointer, err := cutil.WrapGoMem(goslice)
	if err != nil {
		t.Error(err)
	}
	devallo := CreateAllocator(cstream, dev)

	cudamem, err := devallo.AllocateMemory(arraysize)
	if err != nil {
		t.Error(err)
	}
	var copyflag cudart.MemcpyKind
	err = cudart.Memcpy(cudamem, gopointer, arraysize, copyflag.Default())
	if err != nil {
		t.Error(err)
	}
	var kindflag cudart.MemcpyKind
	goslice2 := make([]byte, arraysize)
	gopointer2, err := cutil.WrapGoMem(goslice2)
	if err != nil {
		t.Error(err)
	}
	err = cudart.Memcpy(gopointer2, cudamem, arraysize, kindflag.Default())
	if bytes.Compare(goslice, goslice2) != 0 {
		fmt.Println("goslice2", goslice2)
		fmt.Println("goslice", goslice)
		t.Error("Not same")
	}

	goslice3 := make([]byte, arraysize)
	n, err := cudamem.Read(goslice3)
	if err != nil {
		if err != io.EOF {
			t.Fatal(err)
		}

	}
	if n != (int)(arraysize) {
		t.Fatal("bytes not arraysize", n)
	}
	if bytes.Compare(goslice3, goslice) != 0 {
		fmt.Println("goslice3", goslice3)
		fmt.Println("goslice", goslice)
		t.Error("NotSame on Read")
	}
	println("done first read")
	//	cudamem.Reset()
	utilmem, err := ioutil.ReadAll(cudamem)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Compare(utilmem, goslice) != 0 {
		fmt.Println("utilMem", utilmem)
		fmt.Println("goslice", goslice)
		t.Error("Not same on ioutil")
	}
}
