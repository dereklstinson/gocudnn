package gocudnn_test

import (
	"errors"
	"testing"

	"github.com/dereklstinson/GoCudnn"
)

func TestPooling(t *testing.T) {

	var p gocudnn.Pooling
	var ten gocudnn.Tensor
	TD, err := ten.NewTensor4dDescriptor(ten.Flgs.Data.Float(), ten.Flgs.Format.NCHW(), []int32{1, 5, 4, 4})
	cherr(err, t)
	PD, err := p.NewPooling2dDescriptor(p.PFlg.AverageCountIncludePadding(), p.NFlg.PropagateNan(), []int32{2, 2}, []int32{0, 0}, []int32{2, 2})
	cherr(err, t)
	outdims, err := PD.GetPoolingForwardOutputDim(TD)
	cherr(err, t)
	resultdims := []int32{1, 5, 2, 2}
	cherr(comp(outdims, resultdims), t)
	PD.DestroyDescriptor()
	TD.DestroyDescriptor()

}
func cherr(err error, t *testing.T) {
	if err != nil {
		t.Error(err)
	}
}
func comp(dims, dims1 []int32) error {
	if len(dims) != len(dims1) {
		return errors.New("dims not same length")
	}
	for i := range dims {
		if dims[i] != dims1[i] {
			return errors.New("dims don't match")
		}
	}
	return nil
}
