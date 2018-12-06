package gocudnn_test

import (
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

func TestBatchNorm(t *testing.T) {
	gocudnn.Cuda{}.LockHostThread()
	handle := gocudnn.NewHandle()
	var bn gocudnn.BatchNorm
	bnmode := bn.Flg.Spatial()

	alpha := gocudnn.CFloat(1)
	beta := gocudnn.CFloat(0)

	xD, x, xGtr, xGslice, err := helperwithdifferentelements([]int32{1, 2, 3, 4})
	if err != nil {
		t.Error(err)
	}
	yD, y, yGtr, yGslice, err := maketestingmatrial4dnchw(0, []int32{1, 2, 3, 4})
	if err != nil {
		t.Error(err)
	}
	bntD, err := bn.DeriveBNTensorDescriptor(xD, bnmode)
	if err != nil {
		t.Error(err)
	}
	BNormMems, err := bnmems(bntD, 6)
	if err != nil {
		t.Error(err)
	}
	bnbias := []float32{0, 0}  //, 1, 1, 1}
	bnscale := []float32{1, 1} //, 1, 1, 1}
	scaleptr, _ := gocudnn.MakeGoPointer(bnscale)
	biasptr, _ := gocudnn.MakeGoPointer(bnbias)
	gocudnn.UnifiedMemCopy(BNormMems[0], scaleptr)
	gocudnn.UnifiedMemCopy(BNormMems[1], biasptr)
	bn.Funcs.BatchNormalizationForwardTraining(
		handle,
		bnmode,
		alpha,
		beta,
		xD,
		x,
		yD,
		y,
		bntD,
		BNormMems[0], BNormMems[1],
		1.0,
		BNormMems[2], BNormMems[3],
		bn.Funcs.MinEpsilon()*10.0,
		BNormMems[4], BNormMems[5],
	)

	slices := make([][]float32, len(BNormMems))
	gptrs := make([]*gocudnn.GoPointer, len(slices))
	for i := range BNormMems {
		slizesize := BNormMems[i].ByteSize()
		slices[i] = make([]float32, slizesize/4)
		gptrs[i], err = gocudnn.MakeGoPointer(slices[i])
		if err != nil {
			t.Error(err)
		}
		err = gocudnn.UnifiedMemCopy(gptrs[i], BNormMems[i])
		if err != nil {
			t.Error(err)
		}
	}

	err = gocudnn.UnifiedMemCopy(xGtr, x)
	if err != nil {
		t.Error(err)
	}
	err = gocudnn.UnifiedMemCopy(yGtr, y)
	if err != nil {
		t.Error(err)
	}
	//	fmt.Println(xD, x, xGtr, xGslice, yD, y, yGtr, yGslice, BNormMems)
	for i := range slices {
		t.Error(slices[i])
	}
	t.Error(xGslice)
	t.Error(yGslice)
}
func check(err error, t *testing.T) {
	if err != nil {
		t.Error(err)
	}
}

func bnmems(t *gocudnn.TensorD, num int) ([]*gocudnn.Malloced, error) {
	cudamem := make([]*gocudnn.Malloced, num)
	size, err := t.GetSizeInBytes()
	if err != nil {
		return nil, err
	}
	for i := range cudamem {

		cudamem[i], err = gocudnn.Malloc(size)
		if err != nil {
			return nil, err
		}
		err = cudamem[i].Set(0)
		if err != nil {
			return nil, err
		}
	}
	return cudamem, nil
}
