package gocudnn_test

import (
	"fmt"
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cudart"
)

func TestDeconvolution(t *testing.T) {
	handle := gocudnn.CreateHandle(true)
	cdesc, err := gocudnn.CreateDeConvolutionDescriptor()
	if err != nil {
		t.Error(err)
	}
	var cmode gocudnn.ConvolutionMode
	var dtype gocudnn.DataType
	err = cdesc.Set(cmode.CrossCorrelation(), dtype.Float(), []int32{1, 1}, []int32{1, 1}, []int32{1, 1})
	if err != nil {
		t.Error(err)
	}
	input, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}
	var tfmt gocudnn.TensorFormat
	inputdims := []int32{3, 3, 32, 32}

	err = input.Set(tfmt.NCHW(), dtype, inputdims, nil)
	if err != nil {
		t.Error(err)
	}
	filter, err := gocudnn.CreateFilterDescriptor()
	if err != nil {
		t.Error(err)
	}
	filtertensor, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}
	filtershape := []int32{3, 12, 5, 5}
	err = filtertensor.Set(tfmt, dtype, filtershape, nil)
	if err != nil {
		t.Error(err)
	}
	//filter.Set()
	err = filter.Set(dtype, tfmt, filtershape)
	if err != nil {
		t.Error(err)
	}
	bias, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}
	biasdims, err := cdesc.GetBiasDims(filter)
	if err != nil {
		t.Error(err)
	}
	err = bias.Set(tfmt, dtype, biasdims, nil)
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
	biasSIB, err := bias.GetSizeInBytes()
	if err != nil {
		t.Error(err)
	}
	inputMEM, err := allocator.Malloc(inputSIB)
	if err != nil {
		t.Error(err)
	}
	dinputMEM, err := allocator.Malloc(inputSIB)
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

	biasMem, err := allocator.Malloc(biasSIB)
	if err != nil {
		t.Error(err)
	}
	var forwardpref gocudnn.DeConvolutionForwardPref
	convalgo, err := cdesc.GetForwardAlgorithm(handle, input, filter, output, forwardpref.NoWorkSpace(), 0)
	if err != nil {
		t.Error(err)
	}

	err = gocudnn.SetTensor(handle, filtertensor, filterMEM, 2)
	if err != nil {
		t.Error(err)
	}
	err = gocudnn.SetTensor(handle, input, inputMEM, 1)
	if err != nil {
		t.Error(err)
	}
	err = gocudnn.SetTensor(handle, input, dinputMEM, 0)
	if err != nil {
		t.Error(err)
	}
	err = gocudnn.SetTensor(handle, bias, biasMem, .001)
	if err != nil {
		t.Error(err)
	}

	err = cdesc.Forward(handle, 1, input, inputMEM, filter, filterMEM, convalgo, nil, 0, 0, output, outputMEM)
	if err != nil {
		t.Error(err)
	}
	err = gocudnn.AddTensor(handle, 1, bias, biasMem, 1, output, outputMEM)
	if err != nil {
		t.Error(err)
	}

	err = gocudnn.SetTensor(handle, output, outputMEM, .001)
	if err != nil {
		t.Error(err)
	}
	var backwarddatapref gocudnn.DeConvBwdDataPref

	bdconvalgo, err := cdesc.GetBackwardDataAlgorithm(handle, filter, output, input, backwarddatapref.NoWorkSpace(), 0)
	if err != nil {
		t.Error(err)
	}
	err = cdesc.BackwardData(handle, 1, filter, filterMEM, output, outputMEM, bdconvalgo, nil, 0, 0, input, dinputMEM)
	if err != nil {
		t.Error(err)
	}
	//Now the real test
	var backwardfilterpref gocudnn.DeConvBwdFilterPref
	bfconvalgo, err := cdesc.GetBackwardFilterAlgorithm(handle, output, input, filter, backwardfilterpref.NoWorkSpace(), 0)
	if err != nil {
		t.Error(err)
	}
	err = cdesc.BackwardFilter(handle, 1, input, inputMEM, output, outputMEM, bfconvalgo, nil, 0, 0, filter, filterMEM)
	if err != nil {
		t.Error(err)
	}
	err = stream.Sync()
	if err != nil {
		t.Error(err)
	}

}
