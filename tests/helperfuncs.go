package tests

import gocudnn "github.com/dereklstinson/GoCudnn"

func testarray(size int) []int32 {
	array := make([]int32, size)
	for i := 0; i < size; i++ {
		array[i] = int32(1)
	}
	return array
}

func testTensorFloat4dNHWC(input []int32) (*gocudnn.TensorD, *gocudnn.Malloced, error) {
	var Dtype gocudnn.DataTypeFlag
	var format gocudnn.TensorFormatFlag

	xD, err := gocudnn.NewTensor4dDescriptor(Dtype.Float(), format.NHWC(), input)
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
	var Dtype gocudnn.DataTypeFlag
	var format gocudnn.TensorFormatFlag

	xD, err := gocudnn.NewFilter4dDescriptor(Dtype.Float(), format.NHWC(), input)
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
	var dtype gocudnn.DataTypeFlag
	var convmode gocudnn.ConvolutionModeFlag

	convd, err := gocudnn.NewConvolution2dDescriptor(convmode.CrossCorrelation(), dtype.Float(),
		pads, strides, dialation)
	if err != nil {
		return nil, err
	}
	return convd, nil
}
