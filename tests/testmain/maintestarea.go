package main

/*
#include <cudnn.h>
*/
import "C"

import (
	"github.com/dereklstinson/GoCudnn"
)

func main() {

}

func cherr(err error) {
	if err != nil {
		panic(err)
	}
}

func testarray(size int) []int32 {
	array := make([]int32, size)
	for i := 0; i < size; i++ {
		array[i] = int32(i)
	}
	return array
}

func testTensorFloat4dNCHW(input []int32) (*gocudnn.TensorD, *gocudnn.Malloced, error) {
	var Dtype gocudnn.DataTypeFlag
	var format gocudnn.TensorFormatFlag
	var Tensor gocudnn.Tensor
	xD, err := Tensor.NewTensor4dDescriptor(Dtype.Float(), format.NCHW(), input)
	if err != nil {
		return nil, nil, err
	}
	xDsize, err := xD.GetSizeInBytes()
	if err != nil {
		return nil, nil, err
	}
	xmem, err := gocudnn.UnifiedMangedGlobal(xDsize)
	if err != nil {
		return nil, nil, err
	}
	return xD, xmem, nil

}
func goarraytest(dims []int32) []float32 {
	array := make([]float32, parammaker(dims))
	for i := 0; i < len(array); i++ {
		array[i] = float32(i)
	}

	return array
}
func parammaker(dims []int32) int32 {
	mult := int32(1)
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	return mult
}
