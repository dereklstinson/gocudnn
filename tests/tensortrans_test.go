package tests

import (
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

func TestTensoTrans(t *testing.T) {
	handle := gocudnn.NewHandle()
	float := gocudnn.DataTypeFlag{}.Float()
	//	frmt := gocudnn.TensorFormatFlag{}.NCHW()
	//frmt2 := gocudnn.TensorFormatFlag{}.NHWC()
	var err error
	xDesc, err := gocudnn.Tensor{}.NewTensor4dDescriptorEx(float, []int32{1, 4, 4, 2}, []int32{4 * 4 * 2, 4 * 2, 2, 1})
	if err != nil {
		t.Error(err)
	}
	yDesc, err := gocudnn.Tensor{}.NewTensor4dDescriptorEx(float, []int32{1, 4, 4, 2}, []int32{4 * 4 * 2, 4 * 2 * 2, 2 * 2, 1})
	if err != nil {
		t.Error(err)
	}
	xsize, _ := xDesc.GetSizeInBytes()
	x, err := gocudnn.Malloc(xsize)
	if err != nil {
		t.Error(err)
	}

	ysize, _ := yDesc.GetSizeInBytes()
	//t.Error(xsize)
	//t.Error(ysize)
	y, err := gocudnn.Malloc(ysize)

	err = gocudnn.Tensor{}.TransformTensor(handle, gocudnn.CFloat(1), xDesc, x, gocudnn.CFloat(0), yDesc, y)
	if err != nil {
		t.Error(err)
	}
	xDesc.DestroyDescriptor()
	yDesc.DestroyDescriptor()
	x.Free()
	y.Free()
}
