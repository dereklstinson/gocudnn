package tests

import (
	"fmt"
	"testing"

	"github.com/dereklstinson/cuda/cudnn"
)

func TestConvolution(t *testing.T) {

	tens, err := cudnn.NewTensor(cudnn.DataTypeFloat, cudnn.TensorFormatNHWC, cudnn.Shape(1, 3, 32, 32), nil)
	if err != nil {
		t.Error(err)
	}
	fmt.Println(tens)
	filts, err := cudnn.CreateFilterDescriptor(cudnn.DataTypeFloat, cudnn.TensorFormatNHWC, cudnn.Shape(3, 3, 3, 3))
	if err != nil {
		t.Error(err)
	}
	fmt.Println(filts)

	convd, err := cudnn.CreateConvolutionDescriptor(cudnn.CrossCorrelation, cudnn.DataTypeFloat,
		cudnn.Pads(1, 1), cudnn.Strides(1, 1), cudnn.Dialation(1, 1))
	if err != nil {
		t.Error(err)
	}
	fmt.Println(convd)
	dims, err := convd.GetConvolution2dForwardOutputDim(&tens, &filts)
	if err != nil {
		t.Error(err)
	}
	fmt.Println(dims)
	tensout, err := cudnn.NewTensor(cudnn.DataTypeFloat, cudnn.TensorFormatNCHW, dims, nil)
	if err != nil {
		t.Error(err)
	}
	handle := cudnn.NewHandle()

	top5performers, err := handle.FindConvolutionForwardAlgorithm(&tens, &filts, &convd, &tensout, 5)
	if err != nil {
		t.Error(err)
	}
	for i := 0; i < len(top5performers); i++ {
		fmt.Println(top5performers[i])
	}

}
