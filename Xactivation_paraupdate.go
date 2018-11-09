package gocudnn

/*
//UpdateParas will update the alphas using the optimizer specified.  Adagrad doesn't use xsum so that can be nil if using adagrad.
func (xA *XActivationD) UpdateParas(h *XHandle, dxD *TensorD, alphas, dalphas, xsum, gsum, l1, l2 *Malloced, t TrainingParams, r RegParams) error {
	//	fmt.Println("TrainingParams", t)
	//	fmt.Println("Regularization", r)
	var dtf DataTypeFlag
	dtype, _, _, err := dxD.GetDescrptor()
	if dtype != dtf.Float() {
		return errors.New("only supports Float or float32 data type")
	}

	if err != nil {
		return err
	}

	length := FindLength(alphas.ByteSize(), dtype)
	config := h.LaunchConfig(int32(length))

	if xA.rmodek == nil {
		return errors.New("regularization mode not set this is internal and if not using parmetric activation then you shouldn't update the alphas")
	}

	err = xA.rmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dalphas, alphas, l1, l2, r.batch, r.decay1, r.decay2)
	if err != nil {
		return err
	}

	switch xA.tmode {
	case TrainingModeFlag{}.Adam():
		if xA.counter < 1 {
			xA.counter = 1
		}
		err = xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alphas, gsum, xsum, dalphas, t.rate, t.beta1, t.beta2, t.eps, float32(xA.counter))
		if err != nil {
			return err
		}

		xA.counter++
	case TrainingModeFlag{}.AdaDelta():
		err = xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alphas, gsum, xsum, dalphas, t.rate, t.eps)
		if err != nil {
			return err
		}

	case TrainingModeFlag{}.AdaGrad():
		err = xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alphas, dalphas, gsum, t.rate, t.eps)
		if err != nil {
			return err
		}

	default:
		return errors.New("Unsupported Update")

	}

	return nil
}
*/
