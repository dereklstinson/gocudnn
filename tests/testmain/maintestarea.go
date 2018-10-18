package main

/*
#include <cudnn.h>
*/
import "C"

import (
	"fmt"

	"github.com/dereklstinson/GoCudnn"
)

func main() {

	gocudnn.Cuda{}.LockHostThread()
	trainingkernellocation := "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/"
	var cu gocudnn.Cuda
	dev, err := cu.GetDeviceList()
	err = dev[0].Set()
	handle := gocudnn.NewHandle()
	xhandle, err := gocudnn.Xtra{}.MakeXHandle(trainingkernellocation, dev[0])
	cherr(err)
	stream, err := cu.CreateBlockingStream()
	cherr(err)
	handle.SetStream(stream)
	xhandle.SetStream(stream)
	originaldims := []int32{2, 2, 5, 5}
	descX, Xmem, err := testTensorFloat4dNCHW(originaldims)
	cherr(err)

	goarray := goarraytest(originaldims)
	//fmt.Println(goarray)
	Xhmem, err := gocudnn.MakeGoPointer(goarray)
	cherr(err)

	err = gocudnn.CudaMemCopy(Xmem, Xhmem, Xmem.ByteSize(), gocudnn.MemcpyKindFlag{}.Default())
	cherr(err)

	DescTrans, err := gocudnn.Xtra{}.CreateTransposeDesc(xhandle)

	transdesc, perms, err := DescTrans.GetChannelTransposeOutputDescAndPerm4d(descX)
	fmt.Println(perms)
	cherr(err)
	tsib, err := transdesc.GetSizeInBytes()
	cherr(err)
	transmem, err := gocudnn.UnifiedMangedGlobal(tsib)
	cherr(err)
	err = DescTrans.Transpose(xhandle, perms, descX, Xmem, transdesc, transmem)
	cherr(err)
	flt := gocudnn.DataTypeFlag{}.Float()
	length := gocudnn.FindLength(tsib, flt)
	slice := make([]float32, length)
	transmem.FillSlice(slice)
	//fmt.Println(slice)
	transback, permsback, err := DescTrans.GetChannelTransposeOutputDescAndPerm4d(transdesc)
	fmt.Println(permsback)
	transmemback, err := gocudnn.UnifiedMangedGlobal(tsib)
	err = DescTrans.Transpose(xhandle, permsback, transdesc, transmem, transback, transmemback)
	stream.Sync()

	transmemback.FillSlice(slice)
	//	fmt.Println(slice)
	//	fmt.Println(goarray)
	flag := false
	for i := range slice {
		if slice[i] != goarray[i] {
			flag = true
		}
	}
	if flag == true {
		panic("SliceNoMatchGoarray")
	}
	stbd, err := gocudnn.Xtra{}.CreateShapetoBatchDesc(xhandle)
	cherr(err)
	tobatch, err := stbd.FindShapetoBatchoutputTensor(transdesc, 3, 3)
	cherr(err)
	batchbytes, err := tobatch.GetSizeInBytes()
	cherr(err)
	tobatchmem, err := gocudnn.UnifiedMangedGlobal(batchbytes)
	cherr(err)
	//cherr(tobatchmem.Set(0))
	transmem.FillSlice(slice)
	fmt.Println(slice)

	stream.Sync()
	cherr(stbd.ShapeToBatch4d(xhandle, transdesc, transmem, tobatch, tobatchmem, true))
	_, batchdims, _, err := tobatch.GetDescrptor()
	if err != nil {
		cherr(err)
	}
	blockslice := make([]float32, parammaker(batchdims))
	err = (tobatchmem.FillSlice(blockslice))
	fmt.Println(blockslice)
	cherr(err)
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
