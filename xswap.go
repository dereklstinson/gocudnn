package gocudnn

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
func (xtra Xtra) NewSwappertensor(h *XHandle) (*XActivationD, error) {
	var nanflg PropagationNANFlag
	var nan int32
	if nanflg.NotPropagateNan() == nanprop {
		nan = 0
	} else {
		nan = 1
	}
	ctr := int32(1)
	switch amode {
	case XActivationModeFlag{}.AdvanceThreshRandomRelu():
		fwdmode, err := Cuda{}.MakeKernel(amode.tostringfwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		bwdmode, err := Cuda{}.MakeKernel(amode.tostringbwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		act := &XActivationD{
			fwdmode: fwdmode,
			bwdmode: bwdmode,
			amode:   amode,
			propnan: nan,
		}

		return act, nil
	case XActivationModeFlag{}.ParaChan():

		fwdmode, err := Cuda{}.MakeKernel(amode.tostringfwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		bwdmode, err := Cuda{}.MakeKernel(amode.tostringbwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}

		act := &XActivationD{
			fwdmode:   fwdmode,
			bwdmode:   bwdmode,
			amode:     amode,
			counter:   ctr,
			propnan:   nan,
			istrained: true,
		}

		return act, nil
	case XActivationModeFlag{}.Leaky():
		fwdmode, err := Cuda{}.MakeKernel(amode.tostringfwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		bwdmode, err := Cuda{}.MakeKernel(amode.tostringbwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		return &XActivationD{
			fwdmode: fwdmode,
			bwdmode: bwdmode,
			coef:    float32(coef),
			amode:   amode,
			propnan: nan,
		}, nil
	}
	return nil, errors.New("Unsupported Activation")