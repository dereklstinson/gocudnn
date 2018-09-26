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

func maketestfilterW() (*gocudnn.FilterD, gocudnn.Memer, error) {
	dtf := gocudnn.DataTypeFlag{}.Float()
	tff := gocudnn.TensorFormatFlag{}.NCHW()

	shape := gocudnn.Tensor{}.Shape //shape function
	FilterD, err := gocudnn.Filter{}.NewFilter4dDescriptor(dtf, tff, shape(20, 20, 20, 20))
	size, err := FilterD.TensorD().GetSizeInBytes()
	if err != nil {
		return nil, nil, err
	}
	cudamem, err := gocudnn.MallocManaged(size, gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		return nil, nil, err
	}
	goslice := floatslicew()
	gomem, err := gocudnn.MakeGoPointer(goslice)
	if err != nil {
		return nil, nil, err
	}
	err = gocudnn.CudaMemCopy(cudamem, gomem, size, gocudnn.MemcpyKindFlag{}.Default())
	if err != nil {
		return nil, nil, err
	}
	return FilterD, cudamem, nil

}

func maketestfilterDW() (*gocudnn.FilterD, gocudnn.Memer, error) {
	dtf := gocudnn.DataTypeFlag{}.Float()
	tff := gocudnn.TensorFormatFlag{}.NCHW()

	shape := gocudnn.Tensor{}.Shape //shape function
	FilterD, err := gocudnn.Filter{}.NewFilter4dDescriptor(dtf, tff, shape(20, 20, 20, 20))
	size, err := FilterD.TensorD().GetSizeInBytes()
	if err != nil {
		return nil, nil, err
	}
	cudamem, err := gocudnn.MallocManaged(size, gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		return nil, nil, err
	}
	goslice := floatslicedw()
	gomem, err := gocudnn.MakeGoPointer(goslice)
	if err != nil {
		return nil, nil, err
	}
	err = gocudnn.CudaMemCopy(cudamem, gomem, size, gocudnn.MemcpyKindFlag{}.Default())
	if err != nil {
		return nil, nil, err
	}
	return FilterD, cudamem, nil

}

func floatslicedw(dims ...int) []float32 {
	mult := 1
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	array := make([]float32, mult)
	for i := 0; i < mult; i++ {
		array[i] = float32(1.0) / float32((i%6)-3) //just some patterns
	}
	return array
}

func floatslicew(dims ...int) []float32 {
	mult := 1
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	array := make([]float32, mult)
	for i := 0; i < mult; i++ {
		array[i] = float32((i % 6) - 3) //just some patterns
	}
	return array
}
