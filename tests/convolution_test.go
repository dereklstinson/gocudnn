package tests

import (
	"testing"

	"github.com/dereklstinson/GoCudnn"
)

func TestConvolution(t *testing.T) {
	var datatypeflag gocudnn.DataTypeFlag
	var tffunctionflag gocudnn.TensorFormatFlag
	dtflag := datatypeflag.Double()
	tfflag := tffunctionflag.NHWC()
	tens, err := gocudnn.NewTensor4dDescriptor(dtflag, tfflag, gocudnn.Shape(1, 3, 32, 32))
	if err != nil {
		t.Error(err)
	}
	//fmt.Println(tens)
	filts, err := gocudnn.NewFilter4dDescriptor(dtflag, tfflag, gocudnn.Shape(3, 3, 3, 3))
	if err != nil {
		t.Error(err)
	}
	//fmt.Println(filts)
	var convmode gocudnn.ConvolutionModeFlag
	convd, err := gocudnn.NewConvolution2dDescriptor(convmode.CrossCorrelation(), dtflag,
		gocudnn.Pads(1, 1), gocudnn.Strides(1, 1), gocudnn.Dialation(1, 1))
	if err != nil {
		t.Error(err)
	}
	//fmt.Println(convd)
	dims, err := convd.GetConvolution2dForwardOutputDim(tens, filts)
	if err != nil {
		t.Error(err)
	}
	//fmt.Println(dims)

	tensout, err := gocudnn.NewTensor4dDescriptor(dtflag, tfflag, dims)
	if err != nil {
		t.Error(err)
	}
	handle := gocudnn.NewHandle()

	top5performers, err := handle.FindConvolutionForwardAlgorithm(tens, filts, convd, tensout, 5)
	if err != nil {
		t.Error(err)
	}
	flaggers := make([]bool, len(top5performers))
	for i := 0; i < len(top5performers); i++ {
		//fmt.Println(top5performers[i])
		flaggers[i] = true
		if top5performers[i].Stat != gocudnn.StatusSuccess {
			flaggers[i] = false
			//	t.Error(top5performers[i].Stat.GetErrorString())
		}
	}
	var checkflag bool
	for i := 0; i < len(flaggers); i++ {
		if flaggers[i] == true {
			checkflag = true
		}
	}
	if checkflag != true {
		t.Error("one of these should have been true")
	}
}
