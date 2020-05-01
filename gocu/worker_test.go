package gocu_test

import (
	"bytes"
	"io/ioutil"
	"testing"

	"github.com/dereklstinson/gocudnn/cudart/crtutil"

	"github.com/dereklstinson/gocudnn/cudart"
	"github.com/dereklstinson/gocudnn/gocu"
)

func TestNewWorker(t *testing.T) {
	check := func(e error) {
		if e != nil {
			t.Error(e)
		}
	}
	var (
		a = 1
		b = 2
		c = 0
	)
	d, err := cudart.GetDevice()
	check(err)
	worker := gocu.NewWorker(d)
	err = worker.Work(func() error {
		c = a + b
		return nil
	})
	if c != a+b {
		t.Error("c != a+b")
	}
	allocator := crtutil.CreateAllocator(nil, d)
	mem, err := allocator.AllocateMemory(256)
	check(err)
	writegomem := make([]byte, 256)

	for i := range writegomem {
		writegomem[i] = byte(i)
	}

	_, err = mem.Write(writegomem)
	check(err)
	mem.Reset()
	backtohost, err := ioutil.ReadAll(mem)
	check(err)
	if bytes.Compare(writegomem, backtohost) != 0 {
		t.Error("Don't Match")
	}

}
