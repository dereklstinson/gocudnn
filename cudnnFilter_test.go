package gocudnn

import (
	"testing"
)

func TestCreateFilterDescriptor(t *testing.T) {
	filter, err := CreateFilterDescriptor()
	if err != nil {
		t.Error(err)
	}
	var frmt TensorFormat
	frmt.NHWC()
	var dtype DataType
	dtype.Float()
	shape := []int32{1, 2, 3, 4}
	err = filter.Set(dtype, frmt, shape)

	if err != nil {
		t.Error(err)
	}
	rdtype, rfrmt, rshape, err := filter.Get()
	if err != nil {
		t.Error(err)
	}
	if rdtype != dtype || rfrmt != frmt {
		t.Error("rdtype!=dtype ||rfrmt!=frmt")
	}
	for i := range shape {
		if shape[i] != rshape[i] {
			t.Error("shape[i]!=rshape[i]")
		}
	}
}
