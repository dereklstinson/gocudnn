package gocudnn_test

import (
	"fmt"
	"testing"

	"github.com/dereklstinson/GoCudnn/cudart"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

func TestConvolution2(t *testing.T) {
	handle := gocudnn.CreateHandle(true)
	cdesc, err := gocudnn.CreateConvolutionDescriptor()
	if err != nil {
		t.Error(err)
	}
	var cmode gocudnn.ConvolutionMode
	var dtype gocudnn.DataType
	err = cdesc.Set(cmode.CrossCorrelation(), dtype.Float(), []int32{0, 1, 1}, []int32{1, 1, 1}, []int32{1, 1, 1})
	if err != nil {
		t.Error(err)
	}
	input, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}
	var tfmt gocudnn.TensorFormat
	inputdims := []int32{3, 3, 1000, 32, 32}

	err = input.Set(tfmt.NCHW(), dtype, inputdims, nil)
	if err != nil {
		t.Error(err)
	}
	filter, err := gocudnn.CreateFilterDescriptor()
	if err != nil {
		t.Error(err)
	}
	//filter.Set()
	err = filter.Set(dtype, tfmt, []int32{20, 3, 1, 5, 5})
	if err != nil {
		t.Error(err)
	}
	outputdims, err := cdesc.GetOutputDims(input, filter)
	if err != nil {
		t.Error(err)
	}

	fmt.Println(outputdims)
	output, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}
	err = output.Set(tfmt, dtype, outputdims, nil)
	if err != nil {
		t.Error(err)
	}
	stream, err := cudart.CreateBlockingStream()
	if err != nil {
		t.Error(err)
	}
	dev, err := cudart.CreateDevice(0)
	if err != nil {
		t.Error(err)
	}
	allocator, err := cudart.CreateMemManager(dev)
	if err != nil {
		t.Error(err)
	}
	inputSIB, err := input.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}
	filterSIB, err := filter.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}
	outputSIB, err := output.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}
	inputMEM, err := allocator.Malloc(inputSIB)
	if err != nil {
		t.Error(err)
	}
	filterMEM, err := allocator.Malloc(filterSIB)
	if err != nil {
		t.Error(err)
	}
	outputMEM, err := allocator.Malloc(outputSIB)
	if err != nil {
		t.Error(err)
	}
	var forwardpref gocudnn.ConvolutionForwardPref
	convalgo, err := cdesc.GetForwardAlgorithm(handle, input, filter, output, forwardpref.NoWorkSpace(), 0)
	if err != nil {
		t.Error(err)
	}
	err = cdesc.Forward(handle, 1, input, inputMEM, filter, filterMEM, convalgo, nil, 0, 0, output, outputMEM)
	if err != nil {
		t.Error(err)
	}
	err = stream.Sync()
	if err != nil {
		t.Error(err)
	}
}
func TestConvolution(t *testing.T) {
	handle := gocudnn.CreateHandle(true)
	cdesc, err := gocudnn.CreateConvolutionDescriptor()
	if err != nil {
		t.Error(err)
	}
	var cmode gocudnn.ConvolutionMode
	var dtype gocudnn.DataType
	err = cdesc.Set(cmode.CrossCorrelation(), dtype.Float(), []int32{0, 1, 1}, []int32{1, 1, 1}, []int32{1, 1, 1})
	if err != nil {
		t.Error(err)
	}
	input, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}
	var tfmt gocudnn.TensorFormat

	err = input.Set(tfmt.NCHW(), dtype, []int32{3, 3, 1000, 32, 32}, nil)
	if err != nil {
		t.Error(err)
	}
	filter, err := gocudnn.CreateFilterDescriptor()
	if err != nil {
		t.Error(err)
	}
	err = filter.Set(dtype, tfmt, []int32{20, 3, 1, 5, 5})
	if err != nil {
		t.Error(err)
	}
	outputdims, err := cdesc.GetOutputDims(input, filter)
	if err != nil {
		t.Error(err)
	}

	fmt.Println(outputdims)
	output, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}
	err = output.Set(tfmt, dtype, outputdims, nil)
	if err != nil {
		t.Error(err)
	}
	stream, err := cudart.CreateBlockingStream()
	if err != nil {
		t.Error(err)
	}
	dev, err := cudart.CreateDevice(0)
	if err != nil {
		t.Error(err)
	}
	allocator, err := cudart.CreateMemManager(dev)
	if err != nil {
		t.Error(err)
	}
	inputSIB, err := input.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}
	filterSIB, err := filter.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}
	outputSIB, err := output.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}
	inputMEM, err := allocator.Malloc(inputSIB)
	if err != nil {
		t.Error(err)
	}
	filterMEM, err := allocator.Malloc(filterSIB)
	if err != nil {
		t.Error(err)
	}
	outputMEM, err := allocator.Malloc(outputSIB)
	if err != nil {
		t.Error(err)
	}
	var forwardpref gocudnn.ConvolutionForwardPref
	convalgo, err := cdesc.GetForwardAlgorithm(handle, input, filter, output, forwardpref.NoWorkSpace(), 0)
	if err != nil {
		t.Error(err)
	}
	err = cdesc.Forward(handle, 1, input, inputMEM, filter, filterMEM, convalgo, nil, 0, 0, output, outputMEM)
	if err != nil {
		t.Error(err)
	}
	err = stream.Sync()
	if err != nil {
		t.Error(err)
	}
}
