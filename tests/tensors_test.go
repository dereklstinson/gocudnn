package tests

import (
	"fmt"
	"testing"

	"github.com/dereklstinson/GoCudnn"
)

func TestTensors(t *testing.T) {
	var datatypeflag gocudnn.DataTypeFlag
	var tffunctionflag gocudnn.TensorFormatFlag

	dtf := datatypeflag.Double()
	tff := tffunctionflag.NCHW()

	var Tensor gocudnn.Tensor
	shape := Tensor.Shape
	TensorD, err := Tensor.NewTensor4dDescriptor(dtf, tff, shape(20, 20, 20, 20))

	if err != nil {
		t.Error(err)
	}
	t.Log(TensorD)
	fmt.Println(TensorD)
	/*

		c0 := shape[1] * shape[2] * slide[2]
		if c0 != int(c[0]) {
			t.Error("returned slide not matching slide formula", c0, c[0])
		}
		c1 := shape[2] * slide[2]
		if c1 != int(c[1]) {
			t.Error("returned slide not matching slide formula", c1, c[1])
		}
		c2 := shape[3]
		if c2 != int(c[2]) {
			t.Error("returned slide not matching slide formula", c2, c[2])
		}
		c3 := 1 //it should always be one
		if c3 != int(c[3]) {
			t.Error("returned slide not matching slide formula", c3, c[3])
		}
	*/
}
