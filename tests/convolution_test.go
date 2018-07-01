package tests

import (
	"testing"

	"github.com/dereklstinson/GoCudnn"
)

func TestConvolution(t *testing.T) {
	handle := gocudnn.NewHandle()
	var Tensor gocudnn.Tensor
	array := Tensor.Shape
	pad := array
	stride := array
	dialation := array
	var Conv gocudnn.Convolution
	xD, x, err := testTensorFloat4dNHWC(array(1, 3, 32, 32))

	if err != nil {
		t.Error(err)
	}

	wD, w, err := testFilterFloatNHWC(array(22, 3, 3, 3))
	if err != nil {
		t.Error(err)
	}

	cD, err := testConvolutionFloat2d(pad(1, 1), stride(1, 1), dialation(1, 1))
	if err != nil {
		t.Error(err)
	}

	dims, err := cD.GetConvolution2dForwardOutputDim(xD, wD)
	if err != nil {
		t.Error(err)
	}
	yD, y, err := testTensorFloat4dNHWC(dims)
	if err != nil {
		t.Error(err)
	}
	var pref gocudnn.ConvolutionFwdPrefFlag

	fwdalgo, err := Conv.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, xD, wD, cD, yD, pref.PreferFastest(), 0)
	if err != nil {
		t.Error(err)
	}
	wssize, err := Conv.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle, xD, wD, cD, yD, fwdalgo)
	if err != nil {
		t.Error(err)
	}
	wspace, err := gocudnn.Malloc(wssize)
	if err != nil {
		t.Error(err)
	}
	alpha := gocudnn.CFloat(1.0)
	beta := gocudnn.CFloat(1.0)
	err = Conv.Funcs.Fwd.ConvolutionForward(handle, alpha, xD, x, wD, w, cD, fwdalgo, wspace, beta, yD, y)
	if err != nil {
		t.Error(err)
	}
	//algobwd, err := handle.GetConvolutionBackwardDataAlgorithm()
	w.Free()
	x.Free()
	y.Free()
	wspace.Free()
	//fmt.Println(algobwd)
}
