package gocudnn

import (
	"runtime"
	"testing"
)

func TestCreateDeConvolutionDescriptor(t *testing.T) {
	runtime.LockOSThread()
	check := func(e error) {
		if e != nil {
			t.Error(e)
		}
	}
	spacial := func(i, f, p, s, d int32) int32 {
		return (i-1)*s - 2*p + (((f - 1) * d) + 1)
	}

	var cmode ConvolutionMode
	var dtype DataType
	var frmt TensorFormat
	cmode.CrossCorrelation()
	dtype.Float()
	frmt.NCHW()
	deconv, err := CreateDeConvolutionDescriptor()
	check(err)
	tensor, err := CreateTensorDescriptor()
	check(err)
	filter, err := CreateFilterDescriptor()
	check(err)
	check(deconv.Set(cmode.Convolution(), dtype.Float(), []int32{2, 2}, []int32{2, 2}, []int32{2, 2}))
	check(tensor.Set(frmt, dtype, []int32{32, 256, 32, 32}, nil))
	check(filter.Set(dtype, frmt, []int32{256, 5, 5, 5}))

	want := []int32{32, 5, spacial(32, 5, 2, 2, 2), spacial(32, 5, 2, 2, 2)}
	received, err := deconv.GetOutputDims(tensor, filter)
	check(err)
	for i := range want {
		if want[i] != received[i] {
			t.Error("NotSame")
		}
	}

}
