package gocudnn

import (
	"errors"

	"github.com/dereklstinson/GoCudnn/kernels"
)

//Swapper contains swap kernels that are used through methods
type Swapper struct {
	swapeveryother *Kernel
	//innerswaploweruper *Kernel
	//innerswapbatch     *Kernel
	swapupperlower *Kernel
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
func (s *Swapper) UpperLower(h *XHandle, Adesc *TensorD, A *Malloced, Bdesc *TensorD, B *Malloced, Aupper, Bupper, inverse bool) error {

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
	return s.swapupperlower.Launch(cfg.BlockCountx, cfg.BlockCounty, 1, cfg.ThreadPerBlockx, cfg.ThreadPerBlocky, 1, 0, h.s, cfg.Dimx, cfg.Dimy, A, B, isAupper, isBupper, isinverse)
}

//EveryOther swaps the two tensors by every other batch.  Even does the evens if not even then it does the ood.
func (s *Swapper) EveryOther(h *XHandle, Adesc *TensorD, A *Malloced, Bdesc *TensorD, B *Malloced, even bool) error {
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
	var iseven int32
	if even {
		iseven = 255
	}
	return s.swapeveryother.Launch(cfg.BlockCount, 1, 1, cfg.ThreadPerBlock, 1, 1, 0, h.s, cfg.Elements, batches, A, B, iseven)
}

//NewBatchSwapper makes a Swapper
func (xtra Xtra) NewBatchSwapper(h *XHandle) (*Swapper, error) {
	var cu Cuda
	swapeveryother, err := cu.MakeKernel(kernels.XtraKerns{}.SwapEveryOther(), h.mod)
	if err != nil {
		return nil, err
	}

	swapupperlower, err := cu.MakeKernel(kernels.XtraKerns{}.SwapUpperLower(), h.mod)
	if err != nil {
		return nil, err
	}
	return &Swapper{
		swapeveryother: swapeveryother,

		swapupperlower: swapupperlower,
	}, nil
}
