package gocudnn_test

import (
	"fmt"
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

func TestBatchNorm(t *testing.T) {
	var bn gocudnn.BatchNorm
	var dtf gocudnn.DataTypeFlag
	var tff gocudnn.TensorFormatFlag

	xD, err := gocudnn.Tensor{}.NewTensor4dDescriptor(dtf.Float(), tff.NCHW(), []int32{3, 5, 5, 4})
	check(err, t)
	bnd, err := bn.DeriveBNTensorDescriptor(xD, gocudnn.BatchNormModeFlag{}.PerActivation())
	check(err, t)
	fmt.Println(bnd)
}
func check(err error, t *testing.T) {
	if err != nil {
		t.Error(err)
	}
}
