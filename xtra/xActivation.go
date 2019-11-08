package xtra

import "C"
import (
	"errors"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cuda"
	"github.com/dereklstinson/GoCudnn/kernels"
	"github.com/dereklstinson/cutil"
)

//XActivationMode is flags for xtra activations
type XActivationMode uint

//XActivationModeFlag holds flags for a XactivationMode
type XActivationModeFlag struct {
}

func (x XActivationMode) tostringfwd(dtype gocudnn.DataType) string {
	var dtf gocudnn.DataType
	var xaflg XActivationModeFlag
	var ktf kernels.XtraKerns
	switch dtype {
	case dtf.Half():
		switch x {
		case xaflg.Leaky():
			return ktf.LeakyForwardFP16()
		case xaflg.Threshhold():
			return ktf.ThreshForwardFP16()
		case xaflg.Prelu():
			return ktf.PreluForwardFP16()
		default:
			return "error"
		}
	case dtf.Float():
		switch x {
		case xaflg.Leaky():
			return ktf.LeakyForward()
		case xaflg.Threshhold():
			return ktf.ThreshForward()
		case xaflg.Prelu():
			return ktf.PreluForward()
		default:
			return "error"
		}

	}
	return "error XActivationMode - DataTypeNotSupported"
}

func (x XActivationMode) tostringbwd(dtype gocudnn.DataType) string {
	var dtf gocudnn.DataType
	var xaflg XActivationModeFlag
	var ktf kernels.XtraKerns
	switch dtype {
	case dtf.Half():
		switch x {
		case xaflg.Leaky():
			return ktf.LeakyBackwardFP16()
		case xaflg.Threshhold():
			return ktf.ThreshBackwardFP16()
		case xaflg.Prelu():
			return ktf.PreluBackwardFP16()
		default:
			return "error"
		}
	case dtf.Float():
		switch x {
		case xaflg.Leaky():
			return ktf.LeakyBackward()
		case xaflg.Threshhold():
			return ktf.ThreshBackward()
		case xaflg.Prelu():
			return ktf.PreluBackward()
		default:
			return "error"
		}
	default:
		return "error XActivationMode - DataTypeNotSupported"
	}

}

//Leaky returns the leaky flag
func (x XActivationModeFlag) Leaky() XActivationMode {
	return XActivationMode(101)
}

//Threshhold returns the Parametric flag
func (x XActivationModeFlag) Threshhold() XActivationMode {
	return XActivationMode(102)
}

//Prelu returns the ParaChan flag and it is a weighted leaky on the just the channels
func (x XActivationModeFlag) Prelu() XActivationMode {
	return XActivationMode(103)
}

//XActivationD is the activation descriptor for the "Xtra" stuff that I added to cudnn
type XActivationD struct {
	amode     XActivationMode
	dtype     gocudnn.DataType
	tmode     TrainingMode
	counter   int32
	fwdmode   *cuda.Kernel
	bwdmode   *cuda.Kernel
	coef      float32
	propnan   int32
	istrained bool
	specials  *leakyspecials
}
type leakyspecials struct {
	alphafwd     *cuda.Kernel
	alphabwd     *cuda.Kernel
	alphabetafwd *cuda.Kernel
	alphabetabwd *cuda.Kernel
}

//NewXActivationDescriptor - Creates a descriptor for the xtra functions made for gocudnn.
//Note: Only trainable activations will be trained.  tmode will be ignored for unsupported activations
//Note: Only functions requiring coef will get it.  coef will be ignored for unsupported activations
func NewXActivationDescriptor(h *Handle, amode XActivationMode, dtype gocudnn.DataType, nanprop gocudnn.NANProp, coef float64) (*XActivationD, error) {
	var nanflg gocudnn.NANProp
	var nan int32
	if nanflg.NotPropigate() == nanprop {
		nan = 0
	} else {
		nan = 1
	}
	ctr := int32(1)
	switch amode {
	case XActivationModeFlag{}.Threshhold():
		fwdmode, err := cuda.MakeKernel(amode.tostringfwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		bwdmode, err := cuda.MakeKernel(amode.tostringbwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		act := &XActivationD{
			fwdmode: fwdmode,
			bwdmode: bwdmode,
			amode:   amode,
			propnan: nan,
			dtype:   dtype,
		}

		return act, nil
	case XActivationModeFlag{}.Prelu():

		fwdmode, err := cuda.MakeKernel(amode.tostringfwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		bwdmode, err := cuda.MakeKernel(amode.tostringbwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}

		act := &XActivationD{
			fwdmode:   fwdmode,
			bwdmode:   bwdmode,
			amode:     amode,
			counter:   ctr,
			propnan:   nan,
			dtype:     dtype,
			istrained: true,
		}

		return act, nil
	case XActivationModeFlag{}.Leaky():
		fwdmode, err := cuda.MakeKernel(amode.tostringfwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		bwdmode, err := cuda.MakeKernel(amode.tostringbwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		var ktf kernels.XtraKerns
		specials := new(leakyspecials)

		specials.alphabwd, err = cuda.MakeKernel(ktf.LeakyBackwardAlpha(), h.mod)
		if err != nil {
			return nil, err
		}
		specials.alphafwd, err = cuda.MakeKernel(ktf.LeakyForwardAlpha(), h.mod)
		if err != nil {
			return nil, err
		}
		specials.alphabetabwd, err = cuda.MakeKernel(ktf.LeakyBackwardAlphaBeta(), h.mod)
		if err != nil {
			return nil, err
		}
		specials.alphabetafwd, err = cuda.MakeKernel(ktf.LeakyForwardAlphaBeta(), h.mod)
		if err != nil {
			return nil, err
		}
		return &XActivationD{
			fwdmode:  fwdmode,
			bwdmode:  bwdmode,
			coef:     float32(coef),
			amode:    amode,
			propnan:  nan,
			dtype:    dtype,
			specials: specials,
		}, nil
	}
	return nil, errors.New("Unsupported Activation")
}

//ForwardProp does the forward propagation for xactivation.
//All of the functions  us xD, x ,yD,y..
//Prelu uses coefs. y[i]=coefs[i]* x[i] where x[i]<0
//Threshhold uses coefs and coefs1 for y[i]=x[i]*coefs[i] where x[i]>thres[i] else y[i]=x[i]*coefs1[i]
//The function will only use values that it is used to perform the calculation.  It will ignore the ones that are not used for the function
func (xA *XActivationD) ForwardProp(h *Handle, xD *gocudnn.TensorD, x cutil.Mem, yD *gocudnn.TensorD, y cutil.Mem, coefs, thresh, dthresh, coefs1 cutil.Mem, alpha, beta float64) error {
	_, dtype, dims, _, err := xD.Get()
	if err != nil {
		return err
	}
	var df gocudnn.DataType
	if dtype != df.Float() {
		return errors.New("Only Float is the supported datatype")
	}
	switch xA.amode {
	case XActivationModeFlag{}.Leaky():

		sib, err := xD.GetSizeInBytes()
		if err != nil {
			return err
		}
		length := gocudnn.FindLength(sib, dtype)
		config := h.LaunchConfig(int32(length))
		if alpha != 1 && beta != 0 {
			a := float32(alpha)
			b := float32(beta)
			return xA.specials.alphabetafwd.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, x, y, float32(xA.coef), a, b)
		} else if alpha != 1 && beta == 0 {
			a := float32(alpha)
			return xA.specials.alphafwd.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, x, y, float32(xA.coef), a)
		}
		return xA.fwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, x, y, float32(xA.coef)) //, xA.propnan)
	case XActivationModeFlag{}.Threshhold():

		length := findvolume(dims[1:])
		config := h.LaunchConfig(int32(length))

		err = xA.fwdmode.Launch(config.BlockCount, 1, 1,
			config.ThreadPerBlock, 1, 1, 0, h.s,
			config.Elements, dims[0], x, y, coefs, thresh, coefs1)
		if err != nil {
			return err
		}
		return errors.New("Unsupported XActivationMode")
		//return nil
	case XActivationModeFlag{}.Prelu():
		return errors.New("Unsupported XActivationMode")
		/*
			length := findvolume(dims[1:])
			config := h.LaunchConfig(length)

			err = xA.fwdmode.Launch(config.BlockCount, 1, 1,
				config.ThreadPerBlock, 1, 1, 0, h.s,
				config.Elements, dims[0], x, y, coefs, thresh, coefs1)
			if err != nil {
				return err
			}

			return nil
		*/
	}

	return errors.New("Unsupported XActivationMode")
}

//BackProp does the back propagation for xactivation
//All of the functions use xD, x ,dxD, dx, dyD, dy..
//Prelu uses coefs and dcoefs. dx[i]=coefs[i]* dx[i] where x[i]<0   dcoefs=dy[i]*x[i]
//Threshhold uses coefs and coefs1 thresh, dcoefs,dthresh,and dcoefs1 for dx[i]=dy[i]*coefs[i] where x[i]<thresh[i] else dx[i]=coefs1[i]*dy[i]. and dcoefs[i]+=x[i]*dy[i] same for dcoefs1
//The function will only use values that it is used to perform the calculation.  It will ignore the ones that are not used for the function
func (xA *XActivationD) BackProp(h *Handle, xD *gocudnn.TensorD, x cutil.Mem, dxD *gocudnn.TensorD, dx cutil.Mem, dyD *gocudnn.TensorD, dy cutil.Mem, coefs, dcoefs, thresh, dthresh, coefs1, dcoefs1 cutil.Mem, alpha, beta float64) error {

	switch xA.amode {
	case XActivationModeFlag{}.Leaky():
		return xA.backpropropleaky(h, xD, x, dxD, dx, dyD, dy, alpha, beta)
	case XActivationModeFlag{}.Threshhold():
		return xA.threshback(h, xD, x, dxD, dx, dyD, dy, coefs, dcoefs, thresh, dthresh, coefs1, dcoefs1)
	case XActivationModeFlag{}.Prelu():
		return xA.preluback(h, xD, x, dxD, dx, dyD, dy, coefs, dcoefs)
	}
	return errors.New("Unsupported XActivationMode")
}

func (xA *XActivationD) preluback(h *Handle, xD *gocudnn.TensorD, x cutil.Mem, dxD *gocudnn.TensorD, dx cutil.Mem, dyD *gocudnn.TensorD, dy cutil.Mem, coefs, dcoefs cutil.Mem) error {
	_, dtype, dims, _, err := xD.Get()
	if err != nil {
		return err
	}
	var df gocudnn.DataType
	if dtype != df.Float() {
		return errors.New("Only Float is the supported datatype")
	}

	length := findvolume(dims[1:])
	config := h.LaunchConfig(length)

	err = xA.bwdmode.Launch(config.BlockCount, 1, 1,
		config.ThreadPerBlock, 1, 1, 0, h.s,
		config.Elements, dims[0], dx, x, dy, coefs, dcoefs)
	if err != nil {
		return err
	}

	return nil

}
func (xA *XActivationD) backpropropleaky(h *Handle, xD *gocudnn.TensorD, x cutil.Mem, dxD *gocudnn.TensorD, dx cutil.Mem, dyD *gocudnn.TensorD, dy cutil.Mem, alpha, beta float64) error {
	_, dtype, _, _, err := xD.Get()
	if err != nil {
		return err
	}
	var df gocudnn.DataType
	if dtype != df.Float() {
		return errors.New("Only Float is the supported datatype")
	}
	sib, err := xD.GetSizeInBytes()
	if err != nil {
		return err
	}
	length := gocudnn.FindLength(sib, dtype)

	config := h.LaunchConfig(int32(length))
	//	fmt.Println("alpha beta coef length", alpha, beta, float32(xA.coef))
	//	fmt.Println("Config:", config)
	if alpha != 1 && beta != 0 {
		a := float32(alpha)
		b := float32(beta)
		return xA.specials.alphabetabwd.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, x, dx, dy, float32(xA.coef), a, b)
	} else if alpha != 1 && beta == 0 {
		a := float32(alpha)
		return xA.specials.alphabwd.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, x, dx, dy, float32(xA.coef), a)
	}
	return xA.bwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, x, dx, dy, float32(xA.coef)) //, xA.propnan)
}
func (xA *XActivationD) threshback(h *Handle, xD *gocudnn.TensorD, x cutil.Mem, dxD *gocudnn.TensorD, dx cutil.Mem, dyD *gocudnn.TensorD, dy cutil.Mem, ncoefs, dncoefs, thresh, dthresh, pcoefs1, dpcoefs1 cutil.Mem) error {
	_, dtype, dims, _, err := xD.Get()
	if err != nil {
		return err
	}
	var df gocudnn.DataType
	if dtype != df.Float() {
		return errors.New("Only Float is the supported datatype")
	}

	length := findvolume(dims[1:])
	config := h.LaunchConfig(int32(length))

	err = xA.bwdmode.Launch(config.BlockCount, 1, 1,
		config.ThreadPerBlock, 1, 1, 0, h.s,
		config.Elements, dims[0], x, dx, dy, ncoefs, dncoefs, thresh, dthresh, pcoefs1, dpcoefs1)
	if err != nil {
		return err
	}

	return nil
}
