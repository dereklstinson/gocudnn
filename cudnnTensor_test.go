package gocudnn

import (
	"fmt"
	"testing"
)

func TestCreateTensorDescriptor(t *testing.T) {
	//checkarray func
	checkarrays := func(a, b []int32) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}

	var (
		frmt  TensorFormat
		dtype DataType
		cmode ConvolutionMode
	)
	tensor, err := CreateTensorDescriptor()
	if err != nil {
		t.Error(err)
	}
	oshape := []int32{10, 3, 36, 36}
	//NCHW N=Batches, C=Channels/Feature Maps, H=Height,W=Width for a tensor
	err = tensor.Set(frmt.NCHW(), dtype.Float(), oshape, nil) //Since frmt is not set to strided then last option can be nil
	if err != nil {
		t.Error(err)
	}
	ostride := make([]int32, 4)

	strider := int32(1)
	for i := len(ostride) - 1; i >= 0; i-- {
		ostride[i] = strider
		strider *= oshape[i]
	}
	gfrmt, gtype, gshape, gstride, err := tensor.Get()
	if !checkarrays(oshape, gshape) {
		t.Error("Not Matching", oshape, gshape)
	}
	if !checkarrays(ostride, gstride) {
		t.Error("Not Matching", ostride, gstride)
	}
	if gfrmt != frmt {
		t.Error("Not Matching", gfrmt.ToString(), frmt.ToString())
	}
	if gtype != dtype {
		t.Error("Not Matching", gtype.ToString(), dtype.ToString())
	}
	filter, err := CreateFilterDescriptor()
	if err != nil {
		t.Error(err)
	}
	//dtype,frmt don't need their methods recalled since their methods also change their value.
	//N = Number of Output Feature Maps, C= Number of Input Feature Maps (C needs to be 3 because of the previous tensor), H=Height ,W=Width
	ofshape := []int32{20, 3, 5, 5}
	err = filter.Set(dtype, frmt, ofshape)
	if err != nil {
		t.Error(err)
	}
	fdtype, ffrmt, fgshape, err := filter.Get()
	if err != nil {
		t.Error(err)
	}
	if !checkarrays(ofshape, fgshape) {
		if !checkarrays(oshape, gshape) {
			t.Error("Not Matching", ofshape, fgshape)
		}
	}
	if ffrmt != frmt {
		t.Error("Not Matching", ffrmt.ToString(), frmt.ToString())
	}
	if fdtype != dtype {
		t.Error("Not Matching", fdtype.ToString(), dtype.ToString())
	}
	convolution, err := CreateConvolutionDescriptor()
	if err != nil {
		t.Error(err)
	}

	err = convolution.Set(cmode.CrossCorrelation(), dtype, []int32{0, 0}, []int32{3, 3}, []int32{1, 1})
	if err != nil {
		t.Error(err)
	}
	//34-5=29     29/3
	outdims, err := convolution.GetOutputDims(tensor, filter)
	if err != nil {
		t.Error(err)
	}

	fmt.Println(outdims)
}
