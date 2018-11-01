package gocudnn

import "C"
import (
	"errors"

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
	case xaflg.parametric():
		return ktf.ForwardParametricfloat()
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
	case xaflg.parametric():
		return ktf.BackwardParametricfloat()
	}
	return "error"
}

//Leaky returns the leaky flag
func (x XActivationModeFlag) Leaky() XActivationMode {
	return XActivationMode(1)
}

//Parametric returns the Parametric flag
func (x XActivationModeFlag) parametric() XActivationMode {
	return XActivationMode(2)
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
	coef    float64
}

//NewXActivationDescriptor - Creates a descriptor for the xtra functions made for gocudnn.
//Note: Only trainable activations will be trained.  tmode will be ignored for unsupported activations
//Note: Only functions requiring coef will get it.  coef will be ignored for unsupported activations
func (xtra Xtra) NewXActivationDescriptor(h *XHandle, amode XActivationMode, tmode TrainingMode, dtype DataType, invcoef float64) (*XActivationD, error) {

	ctr := int32(1)
	var ktf kernels.XtraKerns
	switch amode {
	case XActivationModeFlag{}.parametric():
		fwdmode, err := Cuda{}.MakeKernel(amode.tostringfwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		bwdmode, err := Cuda{}.MakeKernel(amode.tostringbwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		rmodek, err := Cuda{}.MakeKernel(ktf.Batch(), h.mod)
		if err != nil {
			return nil, err
		}
		tmodek, err := Cuda{}.MakeKernel(tmode.tostring(), h.mod)
		if err != nil {
			return nil, err
		}
		return &XActivationD{
			fwdmode: fwdmode,
			bwdmode: bwdmode,
			rmodek:  rmodek,
			tmodek:  tmodek,
			amode:   amode,
			tmode:   tmode,
			counter: ctr,
		}, nil
	default:
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

}

//ForwardProp does the feed forward operation alphas can be nil iff it is not supported (leaky right now).  otherwise it needs to be the same size as x and y
func (xA *XActivationD) ForwardProp(h *XHandle, xD *TensorD, x Memer, yD *TensorD, y Memer, alphas Memer) error {
	dtype, _, _, err := xD.GetDescrptor()
	sizeinbytes, err := xD.GetSizeInBytes()
	if err != nil {
		return err
	}
	length := FindLength(sizeinbytes, dtype)

	switch xA.amode {
	case XActivationModeFlag{}.Leaky():
		config := h.LaunchConfig(int32(length))
		return xA.fwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, x, y, float32(xA.coef))

	}
	return errors.New("Unsupported XActivationMode")
}

//BackProp does the feed forward operation alphas and dalphas can be nil iff it is not supported (leaky right now).  otherwise it needs to be the same size as x and y(parametric right now)
func (xA *XActivationD) BackProp(h *XHandle, xD *TensorD, x Memer, dxD *TensorD, dx Memer, dyD *TensorD, dy Memer, alphas, dalphas Memer) error {
	dtype, _, _, err := dxD.GetDescrptor()
	sizeinbytes, err := dxD.GetSizeInBytes()
	if err != nil {
		return err
	}
	length := FindLength(sizeinbytes, dtype)

	switch xA.amode {
	case XActivationModeFlag{}.Leaky():
		config := h.LaunchConfig(int32(length))
		return xA.bwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, x, dx, dy, float32(xA.coef))
	}
	return errors.New("Unsupported XActivationMode")
}

//UpdateAlphas will update the alphas using the optimizer specified.  Adagrad doesn't use xsum so that can be nil if using adagrad.
func (xA *XActivationD) UpdateAlphas(h *XHandle, batch int, dxD *TensorD, alphas, dalphas, xsum, gsum Memer, t TrainingParams) error {
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
	err = xA.rmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dalphas, float32(batch))
	if err != nil {
		return err
	}
	switch xA.tmode {
	case TrainingModeFlag{}.Adam():

		err = xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alphas, gsum, xsum, dalphas, t.rate, t.beta1, t.beta2, t.eps, float32(xA.counter))

		///void adamfloat(const int length,float *w,float *gsum,float *xsum,float *dw,const float rate,const float beta1,const float beta2,const float eps,const float counter){
		xA.counter++
		return err
	case TrainingModeFlag{}.AdaDelta():
		return xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alphas, gsum, xsum, dalphas, t.rate, t.eps)
	case TrainingModeFlag{}.AdaGrad():
		return xA.tmodek.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alphas, dalphas, gsum, t.rate, t.eps)

	}

	return errors.New("Unsupported Update")
}
