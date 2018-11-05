package gocudnn_test

import (
	"fmt"
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

func TestTensor(t *testing.T) {
	gocudnn.Cuda{}.LockHostThread()
	var tensor gocudnn.Tensor
	handle := gocudnn.NewHandle()
	float := gocudnn.DataTypeFlag{}.Float()
	var err error
	n, c, h, w := int32(1), int32(3), int32(4), int32(2)
	sharedims := []int32{n, c, h, w}
	//tensor dims a 1,4,4,2... slide is 32,8,2,1
	chw := c * h * w
	hw := h * w
	ostride := []int32{chw, hw, w, 1}
	xDesc, err := tensor.NewTensor4dDescriptorEx(float, sharedims, ostride)
	if err != nil {
		t.Error(err)
	}

	x, y, z := int32(1), int32(4), int32(4)
	xyz := x * y * z
	yz := y * z
	stride := []int32{ostride[0] * xyz, ostride[1] * xyz, ostride[2] * yz, ostride[3] * z}
	outputdims := []int32{(stride[0] * sharedims[0]) / (chw * xyz), (sharedims[1] * stride[1]) / (yz * hw), (sharedims[2] * stride[2]) / (w * z), sharedims[3] * stride[3]}
	//tensor dims a 1,4,4,2...
	yDesc, err := tensor.NewTensor4dDescriptorEx(float, sharedims, stride)
	if err != nil {
		t.Error(err)
	}
	xsize, _ := xDesc.GetSizeInBytes()
	dx, err := gocudnn.Malloc(xsize)
	if err != nil {
		t.Error(err)
	}
	xslice := regular4darray(sharedims)
	xcpuptr, err := gocudnn.MakeGoPointer(xslice)
	if err != nil {
		t.Error(err)
	}
	err = gocudnn.CudaMemCopy(dx, xcpuptr, xsize, gocudnn.MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		t.Error(err)
	}
	printnchw(sharedims, xslice)
	if err != nil {
		t.Error(err)
	}
	ysize, _ := yDesc.GetSizeInBytes()
	//t.Error(xsize)
	//t.Error(ysize)
	dy, err := gocudnn.Malloc(ysize)
	if err != nil {
		t.Error(err)
	}
	err = dy.Set(0)
	if err != nil {
		t.Error(err)
	}
	err = tensor.TransformTensor(handle, gocudnn.CFloat(1), xDesc, dx, gocudnn.CFloat(0), yDesc, dy)
	if err != nil {
		t.Error(err)
	}

	yslice := make([]float32, ysize/4)

	ycpuptr, err := gocudnn.MakeGoPointer(yslice)
	if err != nil {
		t.Error(err)
	}

	gocudnn.CudaMemCopy(ycpuptr, dy, ysize, gocudnn.MemcpyKindFlag{}.DeviceToHost())

	//fmt.Println(yslice)

	fmt.Println("Strided")
	printnchwwithstride(sharedims, stride, yslice)
	fmt.Println("WithZeros")
	if len(yslice) != vol(outputdims) {
		t.Error(len(yslice), "!=", vol(outputdims))
	}
	printnchw(outputdims, yslice)
	testfillzeros(yslice)
	gocudnn.CudaMemCopy(dy, ycpuptr, ysize, gocudnn.MemcpyKindFlag{}.HostToDevice())

	err = tensor.TransformTensor(handle, gocudnn.CFloat(1), yDesc, dy, gocudnn.CFloat(0), xDesc, dx)
	err = gocudnn.CudaMemCopy(xcpuptr, dx, xsize, gocudnn.MemcpyKindFlag{}.DeviceToHost())
	//t.Error(xslice)
	xDesc.DestroyDescriptor()
	yDesc.DestroyDescriptor()
	dx.Free()
	dy.Free()
}

/*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
 */
func testfillzeros(slice []float32) {
	for i := range slice {
		if slice[i] == 0 {
			slice[i] = 100
		}

	}
}

func makeemptyslice(dims []int32) (slice []float32) {
	vol := 1
	for i := range dims {
		vol *= int(dims[i])
	}
	slice = make([]float32, vol)
	return slice
}

func vol(dims []int32) int {
	vol := 1
	for i := range dims {
		vol *= int(dims[i])
	}
	return vol
}
func regular4darray(dims []int32) (slice []float32) {
	vol := 1
	size := len(dims)
	s := make([]int32, size)

	for i := range dims {

		vol *= int(dims[i])
	}
	mult := int32(1)
	for i := size - 1; i >= 0; i-- {
		s[i] = mult
		mult *= (dims[i])
	}
	z := int32(0)

	slice = make([]float32, vol)
	for n := z; n < dims[0]; n++ {
		for c := z; c < dims[1]; c++ {
			for h := z; h < dims[2]; h++ {
				for w := z; w < dims[3]; w++ {
					slice[(n*s[0])+(c*s[1])+(h*s[2])+(w*s[3])] = float32((n+1)*10) + float32((c+1)*1) + float32(h+1)*.1 + float32(w+1)*.01
				}
			}
		}
	}
	return slice
}

func printnchwwithstride(dims, stride []int32, slice []float32) {
	//	size := len(dims)
	s := stride //make([]int32, size)
	//	mult := int32(1)

	/*for i := size - 1; i >= 0; i-- {
		s[i] = mult
		mult *= (dims[i])
	}*/
	z := int32(0)
	for n := z; n < dims[0]; n++ {
		for c := z; c < dims[1]; c++ {
			for h := z; h < dims[2]; h++ {
				fmt.Printf("| ")
				for w := z; w < dims[3]; w++ {
					fmt.Printf("%2.2f ", slice[(n*s[0])+(c*s[1])+(h*s[2])+(w*s[3])])
				}
				fmt.Printf("|\n")
			}
			fmt.Printf("\n")
		}

	}
}
func printnchw(dims []int32, slice []float32) {
	size := len(dims)
	s := make([]int32, size)
	mult := int32(1)
	for i := size - 1; i >= 0; i-- {
		s[i] = mult
		mult *= (dims[i])
	}
	z := int32(0)
	fmt.Println("Dims", dims)
	fmt.Println("Stride", s)
	for n := z; n < dims[0]; n++ {
		for c := z; c < dims[1]; c++ {
			for h := z; h < dims[2]; h++ {
				fmt.Printf("| ")
				for w := z; w < dims[3]; w++ {
					fmt.Printf(" %2.2f ", slice[(n*s[0])+(c*s[1])+(h*s[2])+w])
				}
				fmt.Printf("|\n")
			}
			fmt.Printf("\n")
		}

	}
}

func makesliceofones(dims []int32) (slice []float32) {
	vol := 1
	for i := range dims {
		vol *= int(dims[i])
	}
	slice = make([]float32, vol)
	for i := 0; i < vol; i++ {
		slice[i] = 1
	}
	return slice
}
