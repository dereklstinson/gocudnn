package gocudnn

import "C"
import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCudnn/kernels"
)

//XActivationMode is flags for xtra activations
type XActivationMode uint

//XActivationModeFlag holds flags for a XactivationMode
type XActivationModeFlag struct {
}

//asdfasdf
func (x XActivationMode) tostringfwd(dtype DataType) string {
	dtf := DataTypeFlag{}
	var xaflg XActivationModeFlag
	if dtype != dtf.Float() {
		return "error XActivationMode - DataTypeNotSupported"
	}
	var ktf kernels.XtraKerns
	switch x {
	case xaflg.Leaky():
		return ktf.ForwardLeakyfloat()
	case xaflg.ParaPlus():
		return ktf.ForwardParametricfloat()
	case xaflg.ParaChan():
		return ktf.ForwardParamFloatChan()
	}
	return "error"

}

func (x XActivationMode) tostringbwd(dtype DataType) string {
	dtf := DataTypeFlag{}
	var xaflg XActivationModeFlag
	if dtype != dtf.Float() {
		return "error XActivationMode - DataTypeNotSupported"
	}
	var ktf kernels.XtraKerns
	switch x {
	case xaflg.Leaky():
		return ktf.BackwardLeakyfloat()
	case xaflg.ParaPlus():
		return ktf.BackwardParametricfloat()
	case xaflg.ParaChan():
		return ktf.BackwardParamFloatChan()
	}
	return "error"
}

//Leaky returns the leaky flag
func (x XActivationModeFlag) Leaky() XActivationMode {
	return XActivationMode(1)
}

//ParaPlus returns the Parametric flag
func (x XActivationModeFlag) ParaPlus() XActivationMode {
	return XActivationMode(2)
}

//ParaChan returns the ParaChan flag and it is a weighted leaky on the just the channels
func (x XActivationModeFlag) ParaChan() XActivationMode {
	return XActivationMode(3)
}

//XActivationD is the activation descriptor for the "Xtra" stuff that I added to cudnn
type XActivationD struct {
	data    DataType
	amode   XActivationMode
	dtype   DataType
	tmode   TrainingMode
	counter int32
	fwdmode *Kernel
	bwdmode *Kernel
	rmodek  *Kernel
	tmodek  *Kernel
	abcheck *Kernel
	abeq    []int32
	abptr   *GoPointer
	abdev   *Malloced
	coef    float64
}

//NewXActivationDescriptor - Creates a descriptor for the xtra functions made for gocudnn.
//Note: Only trainable activations will be trained.  tmode will be ignored for unsupported activations
//Note: Only functions requiring coef will get it.  coef will be ignored for unsupported activations
func (xtra Xtra) NewXActivationDescriptor(h *XHandle, amode XActivationMode, tmode TrainingMode, dtype DataType, invcoef float64) (*XActivationD, error) {

	ctr := int32(1)
	var ktf kernels.XtraKerns
	switch amode {
	case XActivationModeFlag{}.ParaPlus():
		fwdmode, err := Cuda{}.MakeKernel(amode.tostringfwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		bwdmode, err := Cuda{}.MakeKernel(amode.tostringbwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		rmodek, err := Cuda{}.MakeKernel(ktf.L1L2(), h.mod)
		if err != nil {
			return nil, err
		}
		tmodek, err := Cuda{}.MakeKernel(tmode.tostring(), h.mod)
		if err != nil {
			return nil, err
		}

		abcheck, err := Cuda{}.MakeKernel(ktf.AlpaBetaCheck(), h.mod)
		act := &XActivationD{
			fwdmode: fwdmode,
			bwdmode: bwdmode,
			rmodek:  rmodek,
			tmodek:  tmodek,
			amode:   amode,
			tmode:   tmode,
			abcheck: abcheck,
			abeq:    make([]int32, 1),
			counter: ctr,
		}
		ptr, err := MakeGoPointer(act.abeq)
		if err != nil {
			return nil, err
		}
		act.abptr = ptr
		act.abdev, err = Malloc(4)
		if err != nil {
			return nil, err
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
		rmodek, err := Cuda{}.MakeKernel(ktf.L1L2(), h.mod)
		if err != nil {
			return nil, err
		}
		tmodek, err := Cuda{}.MakeKernel(tmode.tostring(), h.mod)
		if err != nil {
			return nil, err
		}

		abcheck, err := Cuda{}.MakeKernel(ktf.AlpaBetaCheck(), h.mod)
		act := &XActivationD{
			fwdmode: fwdmode,
			bwdmode: bwdmode,
			rmodek:  rmodek,
			tmodek:  tmodek,
			amode:   amode,
			tmode:   tmode,
			abcheck: abcheck,
			abeq:    make([]int32, 1),
			counter: ctr,
		}
		ptr, err := MakeGoPointer(act.abeq)
		if err != nil {
			return nil, err
		}
		act.abptr = ptr
		act.abdev, err = Malloc(4)
		if err != nil {
			return nil, err
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
			coef:    invcoef,
			amode:   amode,
		}, nil
	}
	return nil, errors.New("Unsupported Activation")
}

//ForwardProp does the feed forward operation alphas and betas can be nil iff it is not supported (leaky right now). ParaPlus needs both alphas and betas
func (xA *XActivationD) ForwardProp(h *XHandle, xD *TensorD, x *Malloced, yD *TensorD, y *Malloced, alphas, betas *Malloced) error {
	dtype, _, dims, err := xD.GetDescrptor()
	if err != nil {
		return err
	}
	var df DataTypeFlag
	if dtype != df.Float() {
		return errors.New("Only Float is the supported datatype")
	}

	switch xA.amode {
	case XActivationModeFlag{}.Leaky():

		sib, err := xD.GetSizeInBytes()
		if err != nil {
			return err
		}
		length := FindLength(sib, dtype)
		//	length := findvolume(dims)
		config := h.LaunchConfig(int32(length))
		return xA.fwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, x, y, float32(xA.coef))
	case XActivationModeFlag{}.ParaPlus():
		length := findvolume(dims[1:])
		config := h.LaunchConfig(int32(length))
		for i := int32(0); i < dims[0]; i++ {
			err := xA.fwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, i, x, y, alphas, betas)
			if err != nil {
				return err
			}
		}
		return nil
	case XActivationModeFlag{}.ParaChan():
		tf, err := xD.GetFormat()
		if err != nil {
			return err
		}
		var tflg TensorFormatFlag
		var NHWC int32
		if tf == tflg.NHWC() {
			NHWC = int32(255)
		}
		config := h.LaunchConfig3d(dims[1], dims[2], dims[3])
		for i := int32(0); i < dims[0]; i++ {
			err := xA.fwdmode.Launch(config.BlockCountx, config.BlockCounty, config.BlockCountz,
				config.ThreadPerBlockx, config.ThreadPerBlocky, config.ThreadPerBlockz, 0, h.s,
				config.Dimx, config.Dimy, config.Dimz, i, x, y, alphas, betas, NHWC)
			if err != nil {
				return err
			}
		}
		return nil
	}

	return errors.New("Unsupported XActivationMode")
}

//BackProp does the feed forward operation alphas and dalphas can be nil iff it is not supported (leaky right now).  otherwise it needs to be the same size as x and y(parametric right now)
func (xA *XActivationD) BackProp(h *XHandle, xD *TensorD, x *Malloced, dxD *TensorD, dx *Malloced, dyD *TensorD, dy *Malloced, alphas, dalphas, betas, dbetas *Malloced) error {
	dtype, _, dims, err := xD.GetDescrptor()
	if err != nil {
		return err
	}
	var df DataTypeFlag
	if dtype != df.Float() {
		return errors.New("Only Float is the supported datatype")
	}

	switch xA.amode {
	case XActivationModeFlag{}.Leaky():

		sib, err := xD.GetSizeInBytes()
		if err != nil {
			return err
		}
		length := FindLength(sib, dtype)

		//	length := findvolume(dims)
		config := h.LaunchConfig(int32(length))
		return xA.bwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, x, dx, dy, float32(xA.coef))
	case XActivationModeFlag{}.ParaPlus():
		length := findvolume(dims[1:])
		config := h.LaunchConfig(int32(length))
		for i := int32(0); i < dims[0]; i++ {
			err := xA.bwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, i, x, dx, dy, alphas, dalphas, betas, dbetas)
			if err != nil {
				return err
			}
		}
		return nil
	case XActivationModeFlag{}.ParaChan():
		tf, err := xD.GetFormat()
		if err != nil {
			return err
		}
		var tflg TensorFormatFlag
		var NHWC int32
		if tf == tflg.NHWC() {
			NHWC = int32(255)
		}
		config := h.LaunchConfig3d(dims[1], dims[2], dims[3])
		for i := int32(0); i < dims[0]; i++ {
			err := xA.bwdmode.Launch(config.BlockCountx, config.BlockCounty, config.BlockCountz,
				config.ThreadPerBlockx, config.ThreadPerBlocky, config.ThreadPerBlockz, 0, h.s,
				config.Dimx, config.Dimy, config.Dimz, i, x, dx, dy, alphas, dalphas, betas, dbetas, NHWC)
			if err != nil {
				return err
			}
		}
		return nil
	}
	return errors.New("Unsupported XActivationMode")
}

//UpdateParas will update the alphas using the optimizer specified.  Adagrad doesn't use xsum so that can be nil if using adagrad.
func (xA *XActivationD) UpdateParas(h *XHandle, dxD *TensorD, alphas, dalphas, betas, dbetas, xsuma, gsuma, xsumb, gsumb, l1, l2 *Malloced, t TrainingParams, r RegParams) error {
	fmt.Println("updating params for activation")
	var dtf DataTypeFlag
	dtype, _, _, err := dxD.GetDescrptor()
	if dtype != dtf.Float() {
		return errors.New("only supports Float or float32 data type")
	}
	sizeinbytes, err := dxD.GetSizeInBytes()
	if err != nil {
		return err
	}

	length := FindLength(sizeinbytes, dtype)
	config := h.LaunchConfig(int32(length))

	if xA.rmodek == nil {
		return errors.New("regularization mode not set this is internal and if not using parmetric activation then you shouldn't update the alphas")
	}
	if r.decay1 == 0 && r.decay2 == 0 {
		err = xA.rmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dalphas, nil, nil, nil, r.batch, r.decay1, r.decay2)
	} else if r.decay1 == 0 {
		err = xA.rmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dalphas, alphas, nil, l2, r.batch, r.decay1, r.decay2)
	} else if r.decay2 == 0 {
		err = xA.rmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dalphas, alphas, l1, nil, r.batch, r.decay1, r.decay2)
	} else {
		err = xA.rmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dalphas, alphas, l1, l2, r.batch, r.decay1, r.decay2)
	}
	if err != nil {
		return err
	}
	if r.decay1 == 0 && r.decay2 == 0 {
		err = xA.rmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dbetas, nil, nil, nil, r.batch, r.decay1, r.decay2)
	} else if r.decay1 == 0 {
		err = xA.rmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dbetas, betas, nil, l2, r.batch, r.decay1, r.decay2)
	} else if r.decay2 == 0 {
		err = xA.rmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dbetas, betas, l1, nil, r.batch, r.decay1, r.decay2)
	} else {
		err = xA.rmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dbetas, betas, l1, l2, r.batch, r.decay1, r.decay2)
	}

	if err != nil {
		return err
	}
	switch xA.tmode {
	case TrainingModeFlag{}.Adam():

		err = xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alphas, gsuma, xsuma, dalphas, t.rate, t.beta1, t.beta2, t.eps, float32(xA.counter))
		if err != nil {
			return err
		}
		err = xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, betas, gsumb, xsumb, dbetas, t.rate, t.beta1, t.beta2, t.eps, float32(xA.counter))
		if err != nil {
			return err
		}
		xA.counter++

	case TrainingModeFlag{}.AdaDelta():
		err = xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alphas, gsuma, xsuma, dalphas, t.rate, t.eps)
		if err != nil {
			return err
		}
		err = xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, betas, gsumb, xsumb, dbetas, t.rate, t.eps)
		if err != nil {
			return err
		}

	case TrainingModeFlag{}.AdaGrad():
		err = xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alphas, dalphas, gsuma, t.rate, t.eps)
		if err != nil {
			return err
		}
		err = xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, betas, dbetas, gsumb, t.rate, t.eps)
		if err != nil {
			return err
		}

	default:
		return errors.New("Unsupported Update")
	}
	err = xA.abcheck.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alphas, betas, xA.abdev)
	if err != nil {
		return err
	}
	err = CudaMemCopy(xA.abptr, xA.abdev, 4, MemcpyKindFlag{}.DeviceToHost())
	if err != nil {
		return err
	}
	if length == uint32(xA.abeq[0]) {
		return errors.New("Betas[i] == Alphas[i] for all of i. You have lost all nonlinearity")
	}
	return nil
}
