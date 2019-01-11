package gocudnn

import (
	"errors"

	"github.com/dereklstinson/GoCudnn/kernels"
)

//Need to add functions for the kernels that I made for batch swapping and stuff

/*
//SwapEveryOther allows the user to swap batches between to tensors
//Either the even or the odd tensors.
func (t XtraKerns) SwapEveryOther() string {
	return "SwapEveryOther"
}

//SwapUpperLower takes to tensors and swaps the upper or lower batches between the two tensors
func (t XtraKerns) SwapUpperLower() string {
	return "SwapUpperLower"
}

//InnerSwapBatch allows the user to swap batches between to tensors
//Either the even or the odd tensors.
func (t XtraKerns) InnerSwapBatch() string {
	return "InnerSwapBatch"
}

//InnerSwapLowerUpper this takes a tensor and swaps the batches inside of a tensor
func (t XtraKerns) InnerSwapLowerUpper() string {
	return "InnerSwapLowerUpper"
}

*/

//Swapper contains swap kernels that are used through methods
type Swapper struct {
	swapeveryother     *Kernel
	innerswaploweruper *Kernel
	innerswapbatch     *Kernel
	swapupperlower     *Kernel
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

//SwapUpperLower swaps two different tensor batches. Either the upper half of both tensors or the lower half of both tensors
func (s *Swapper) SwapUpperLower(h *XHandle, Adesc *TensorD, A *Malloced, Bdesc *TensorD, B *Malloced, upper bool) error {

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
	cfg := h.LaunchConfig(batchvol)
	var isupper int32
	if upper {
		isupper = 255
	}
	return s.swapupperlower.Launch(cfg.BlockCount, 1, 1, cfg.ThreadPerBlock, 1, 1, 0, h.s, cfg.Elements, batches, A, B, isupper)
}

//InnerSwapLowerUpper swaps a single tensor top and bottom batches.  Inverse starts with the very top and the very bottom swapping going toward the middle
//Not inverse has it start at the top and the middle. Swapping them.
func (s *Swapper) InnerSwapLowerUpper(h *XHandle, Adesc *TensorD, A *Malloced, inverse bool) error {

	dims := Adesc.Dims()
	batches := dims[0]
	batchvol := findvol(dims[1:])
	cfg := h.LaunchConfig(batchvol)
	var isinverse int32
	if inverse {
		isinverse = 255
	}
	return s.swapupperlower.Launch(cfg.BlockCount, 1, 1, cfg.ThreadPerBlock, 1, 1, 0, h.s, cfg.Elements, batches, A, isinverse)

}

//SwapEveryOther swaps the two tensors by every other batch.  Even does the evens if not even then it does the ood.
func (s *Swapper) SwapEveryOther(h *XHandle, Adesc *TensorD, A *Malloced, Bdesc *TensorD, B *Malloced, even bool) error {
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
	cfg := h.LaunchConfig(batchvol)
	var iseven int32
	if even {
		iseven = 255
	}
	return s.swapupperlower.Launch(cfg.BlockCount, 1, 1, cfg.ThreadPerBlock, 1, 1, 0, h.s, cfg.Elements, batches, A, B, iseven)
}

//InnerSwapBatch takes two batches from the same tensor and swaps them.
func (s *Swapper) InnerSwapBatch(h *XHandle, Adesc *TensorD, A *Malloced, batcha, batchb int32) error {

	dims := Adesc.Dims()
	batches := dims[0]
	if batches < batcha || batches < batchb {
		return errors.New("BatchA or BatchB needs to be less than number of batches in tensor")
	}
	batchvol := findvol(dims[1:])
	cfg := h.LaunchConfig(batchvol)

	return s.swapupperlower.Launch(cfg.BlockCount, 1, 1, cfg.ThreadPerBlock, 1, 1, 0, h.s, cfg.Elements, A, batcha, batchb)

}

//NewBatchSwapper makes a Swapper
func (xtra Xtra) NewBatchSwapper(h *XHandle) (*Swapper, error) {
	var cu Cuda
	swapeveryother, err := cu.MakeKernel(kernels.XtraKerns{}.SwapEveryOther(), h.mod)
	if err != nil {
		return nil, err
	}
	innerswapbatch, err := cu.MakeKernel(kernels.XtraKerns{}.InnerSwapBatch(), h.mod)
	if err != nil {
		return nil, err
	}
	innerswaplowerupper, err := cu.MakeKernel(kernels.XtraKerns{}.InnerSwapLowerUpper(), h.mod)
	if err != nil {
		return nil, err
	}
	swapupperlower, err := cu.MakeKernel(kernels.XtraKerns{}.SwapUpperLower(), h.mod)
	if err != nil {
		return nil, err
	}
	return &Swapper{
		swapeveryother:     swapeveryother,
		innerswapbatch:     innerswapbatch,
		innerswaploweruper: innerswaplowerupper,
		swapupperlower:     swapupperlower,
	}, nil
}
