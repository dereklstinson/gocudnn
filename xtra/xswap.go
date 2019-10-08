package xtra

import (
	"errors"

	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cuda"
	"github.com/dereklstinson/GoCudnn/kernels"
	"github.com/dereklstinson/cutil"
)

//Swapper contains swap kernels that are used through methods
type Swapper struct {
	swapeveryother     *cuda.Kernel
	swapeveryotherfp16 *cuda.Kernel
	swapupperlower     *cuda.Kernel
	swapupperlowerfp16 *cuda.Kernel
}

func comparedimsswap(a, b []int32) error {
	if len(a) != len(b) {
		return errors.New("SwapFunction-A,B dims don't match")
	}
	for i := range a {
		if a[i] != b[i] {
			return errors.New("SwapFunction-A,B dims don't match")
		}
	}
	return nil
}

//UpperLower swaps two different tensor batches. Either the upper half of both tensors or the lower half of both tensors
//inverse is a holder variable. It doesn't do anything right now
func (s *Swapper) UpperLower(h *Handle, Adesc *gocudnn.TensorD, A cutil.Mem, Bdesc *gocudnn.TensorD, B cutil.Mem, Aupper, Bupper, inverse bool) error {

	err := comparedimsswap(Adesc.Dims(), Bdesc.Dims())
	if err != nil {
		return err
	}
	if Adesc.DataType() != Bdesc.DataType() {
		return errors.New("Swapper Datatype don't match")
	}
	dims := Adesc.Dims()
	batches := dims[0]
	batchvol := findvol(dims[1:])
	//cfg := h.LaunchConfig(batchvol)
	cfg := h.LaunchConfig2d(batches, batchvol)
	var isAupper int32
	var isBupper int32
	var isinverse int32
	if Aupper {
		isAupper = 255
	}
	if Bupper {
		isBupper = 255
	}
	if inverse {
		isinverse = 255
	}
	var dflg gocudnn.DataType
	switch Adesc.DataType() {
	case dflg.Float():
		err = s.swapupperlower.Launch(cfg.BlockCountx, cfg.BlockCounty, 1, cfg.ThreadPerBlockx, cfg.ThreadPerBlocky, 1, 0, h.s, cfg.Dimx, cfg.Dimy, A, B, isAupper, isBupper, isinverse)
		if err != nil {
			return err
		}
	case dflg.Half():
		err = s.swapupperlowerfp16.Launch(cfg.BlockCountx, cfg.BlockCounty, 1, cfg.ThreadPerBlockx, cfg.ThreadPerBlocky, 1, 0, h.s, cfg.Dimx, cfg.Dimy, A, B, isAupper, isBupper, isinverse)
		if err != nil {
			return err
		}
	default:
		return errors.New("Unsupported Datatype")
	}
	return h.s.Sync()

}

//EveryOther swaps the two tensors by every other batch.  Even does the evens if not even then it does the ood.
func (s *Swapper) EveryOther(h *Handle, Adesc *gocudnn.TensorD, A cutil.Mem, Bdesc *gocudnn.TensorD, B cutil.Mem, start, stride int32) error {
	err := comparedimsswap(Adesc.Dims(), Bdesc.Dims())
	if err != nil {
		return err
	}
	if Adesc.DataType() != Bdesc.DataType() {
		return errors.New("Swapper Datatype don't match")
	}
	dims := Adesc.Dims()
	batches := dims[0]
	batchvol := findvol(dims[1:])
	//cfg := h.LaunchConfig2d(batches, batchvol)
	cfg := h.LaunchConfig(batchvol)
	var dflg gocudnn.DataType
	switch Adesc.DataType() {
	case dflg.Float():
		err = s.swapeveryother.Launch(cfg.BlockCount, 1, 1, cfg.ThreadPerBlock, 1, 1, 0, h.s, cfg.Elements, batches, A, B, start, stride)
		if err != nil {
			return err
		}

	case dflg.Half():
		err = s.swapeveryotherfp16.Launch(cfg.BlockCount, 1, 1, cfg.ThreadPerBlock, 1, 1, 0, h.s, cfg.Elements, batches, A, B, start, stride)
		if err != nil {
			return err
		}
	default:
		return errors.New("Unsupported Datatype")
	}

	return h.s.Sync()

}

//NewBatchSwapper makes a Swapper
func NewBatchSwapper(h *Handle) (*Swapper, error) {

	swapeveryother, err := cuda.MakeKernel(kernels.XtraKerns{}.SwapEveryOther(), h.mod)
	if err != nil {
		return nil, err
	}
	swapeveryotherfp16, err := cuda.MakeKernel(kernels.XtraKerns{}.SwapEveryOtherFP16(), h.mod)
	if err != nil {
		return nil, err
	}

	swapupperlower, err := cuda.MakeKernel(kernels.XtraKerns{}.SwapUpperLower(), h.mod)
	if err != nil {
		return nil, err
	}
	swapupperlowerfp16, err := cuda.MakeKernel(kernels.XtraKerns{}.SwapUpperLowerFP16(), h.mod)
	if err != nil {
		return nil, err
	}
	return &Swapper{
		swapeveryother:     swapeveryother,
		swapeveryotherfp16: swapeveryotherfp16,
		swapupperlower:     swapupperlower,
		swapupperlowerfp16: swapupperlowerfp16,
	}, nil
}
