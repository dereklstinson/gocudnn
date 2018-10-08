package tests

import gocudnn "github.com/dereklstinson/GoCudnn"

func testarray(size int) []int32 {
	array := make([]int32, size)
	for i := 0; i < size; i++ {
		array[i] = int32(1)
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
	xmem, err := gocudnn.Malloc(xDsize)
	if err != nil {
		return nil, nil, err
	}
	return xD, xmem, nil

}

func testTensorFloat4dNHWC(input []int32) (*gocudnn.TensorD, *gocudnn.Malloced, error) {
	var Dtype gocudnn.DataTypeFlag
	var format gocudnn.TensorFormatFlag
	var Tensor gocudnn.Tensor
	xD, err := Tensor.NewTensor4dDescriptor(Dtype.Float(), format.NHWC(), input)
	if err != nil {
		return nil, nil, err
	}
	xDsize, err := xD.GetSizeInBytes()
	if err != nil {
		return nil, nil, err
	}
	xmem, err := gocudnn.Malloc(xDsize)
	if err != nil {
		return nil, nil, err
	}
	return xD, xmem, nil

}

func testFilterFloatNHWC(input []int32) (*gocudnn.FilterD, *gocudnn.Malloced, error) {
	var Tensor gocudnn.Tensor
	NHWC := Tensor.Flgs.Format.NHWC()
	DataFloat := Tensor.Flgs.Data.Float()
	var Filter gocudnn.Filter
	xD, err := Filter.NewFilter4dDescriptor(DataFloat, NHWC, input)
	if err != nil {
		return nil, nil, err
	}
	bytesize := filterbytesize(input, int32(4))

	xmem, err := gocudnn.Malloc(gocudnn.SizeT(bytesize))
	if err != nil {
		return nil, nil, err
	}
	return xD, xmem, nil
}
func filterbytesize(size []int32, typebytes int32) int32 {
	multiplier := int32(1)
	for i := 0; i < len(size); i++ {
		multiplier *= size[i]
	}
	return multiplier
}
func testConvolutionFloat2d(pads, strides, dialation []int32) (*gocudnn.ConvolutionD, error) {
	var Conv gocudnn.Convolution
	var Tensor gocudnn.Tensor
	Float := Tensor.Flgs.Data.Float()
	Cross := Conv.Flgs.Mode.CrossCorrelation()
	Conv.Flgs.Mode.Convolution()
	convd, err := Conv.NewConvolution2dDescriptor(Cross, Float,
		pads, strides, dialation)
	if err != nil {
		return nil, err
	}
	return convd, nil
}
