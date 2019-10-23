package xtrakerns

import (
	"github.com/dereklstinson/GoCudnn/cuda"
	"testing"
)

func TestCreateModule(t *testing.T) {
	devs, err := cuda.GetDeviceList()
	if err != nil {
		t.Error(err)
	}

	cuda.CtxCreate(-1, devs[0])
	got, err := CreateModule(MSELoss(), devs[0])
	if err != nil {
		t.Error(err)
	}
	print(got)
}
