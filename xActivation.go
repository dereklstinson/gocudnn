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
	case xaflg.AdvanceThreshRandomRelu():
		return ktf.AdvanceThreshRandomReluForward()
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
	case xaflg.AdvanceThreshRandomRelu():
		return ktf.AdvanceThreshRandomReluBackward()
	case xaflg.ParaChan():

		return ktf.BackwardParamFloatChan()
	}
	return "error"
}

//Leaky returns the leaky flag
func (x XActivationModeFlag) Leaky() XActivationMode {
	return XActivationMode(101)
}

//AdvanceThreshRandomRelu returns the Parametric flag
func (x XActivationModeFlag) AdvanceThreshRandomRelu() XActivationMode {
	return XActivationMode(102)
}

//ParaChan returns the ParaChan flag and it is a weighted leaky on the just the channels
func (x XActivationModeFlag) ParaChan() XActivationMode {
	return XActivationMode(103)
}

//XActivationD is the activation descriptor for the "Xtra" stuff that I added to cudnn
type XActivationD struct {
	data      DataType
	amode     XActivationMode
	dtype     DataType
	tmode     TrainingMode
	counter   int32
	fwdmode   *Kernel
	bwdmode   *Kernel
	coef      float32
	propnan   int32
	istrained bool
}

//NewXActivationDescriptor - Creates a descriptor for the xtra functions made for gocudnn.
//Note: Only trainable activations will be trained.  tmode will be ignored for unsupported activations
//Note: Only functions requiring coef will get it.  coef will be ignored for unsupported activations
func (xtra Xtra) NewXActivationDescriptor(h *XHandle, amode XActivationMode, dtype DataType, nanprop PropagationNAN, coef float64) (*XActivationD, error) {
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
}

//ForwardProp does the forward propagation for xactivation.
//All of the functions use alpha, beta,xD, x ,yD,y..
//alpha and beta is used like y=alpha *Op +beta*y for all xactivation modes
//Parachan uses coefs. y[i]=coefs[i]* x[i] where x[i]<0
//AdvanceThreshRandomRelu uses coefs and coefs1 for y[i]=x[i]*coefs[i] where x[i]<coefs1[i]
//The function will only use values that it is used to perform the calculation.  It will ignore the ones that are not used for the function
func (xA *XActivationD) ForwardProp(h *XHandle, alpha, beta CScalar, xD *TensorD, x *Malloced, yD *TensorD, y *Malloced, coefs, coefs1 *Malloced) error {
	dtype, dims, _, err := xD.GetDescrptor()
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
		config := h.LaunchConfig(int32(length))

		return xA.fwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alpha, beta, x, y, float32(xA.coef), xA.propnan) //, xA.propnan)
	case XActivationModeFlag{}.AdvanceThreshRandomRelu():

		length := findvolume(dims[1:])
		config := h.LaunchConfig(int32(length))
		for i := int32(0); i < dims[0]; i++ {
			err := xA.fwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, i, alpha, beta, x, y, coefs, coefs1, xA.propnan)
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
		} else {
			NHWC = 0
		}
		config := h.LaunchConfig3d(dims[1], dims[2], dims[3])
		for i := int32(0); i < dims[0]; i++ {
			err := xA.fwdmode.Launch(config.BlockCountx, config.BlockCounty, config.BlockCountz,
				config.ThreadPerBlockx, config.ThreadPerBlocky, config.ThreadPerBlockz, 0, h.s,
				config.Dimx, config.Dimy, config.Dimz, i, alpha, beta, x, y, coefs, NHWC, xA.propnan)
			if err != nil {
				return err
			}
		}
		return nil
	}

	return errors.New("Unsupported XActivationMode")
}

//BackProp does the forward propagation for xactivation
//All of the functions use alpha, beta,xD, x ,dxD, dx, dyD, dy..
//alpha and beta is used like dx=alpha *Op +beta*dx for all xactivation modes
//Parachan uses coefs and dcoefs. dx[i]=coefs[i]* dx[i] where x[i]<0   dcoefs=dy[i]*x[i]
//AdvanceThreshRandomRelu uses coefs and coefs1 for dx[i]=dy[i]*coefs[i] where x[i]<coefs1[i]
//The function will only use values that it is used to perform the calculation.  It will ignore the ones that are not used for the function
func (xA *XActivationD) BackProp(h *XHandle, alpha, beta CScalar, xD *TensorD, x *Malloced, dxD *TensorD, dx *Malloced, dyD *TensorD, dy *Malloced, coefs, dcoefs *Malloced) error {

	switch xA.amode {
	case XActivationModeFlag{}.Leaky():
		return xA.backpropropleaky(h, alpha, beta, xD, x, dxD, dx, dyD, dy)
	case XActivationModeFlag{}.AdvanceThreshRandomRelu():
		return xA.backpropParaPlus(h, alpha, beta, xD, x, dxD, dx, dyD, dy, coefs, dcoefs)
	case XActivationModeFlag{}.ParaChan():
		return xA.backpropparachan(h, alpha, beta, xD, x, dxD, dx, dyD, dy, coefs, dcoefs)
	}
	return errors.New("Unsupported XActivationMode")
}

func (xA *XActivationD) backpropparachan(h *XHandle, alpha CScalar, beta CScalar, xD *TensorD, x *Malloced, dxD *TensorD, dx *Malloced, dyD *TensorD, dy *Malloced, coefs, dcoefs *Malloced) error {
	dtype, dims, _, err := xD.GetDescrptor()
	if err != nil {
		return err
	}
	var df DataTypeFlag
	if dtype != df.Float() {
		return errors.New("Only Float is the supported datatype")
	}
	tf, err := xD.GetFormat()
	if err != nil {
		return err
	}
	var tflg TensorFormatFlag
	var NHWC int32
	if tf == tflg.NHWC() {
		NHWC = int32(255)
	}
	//	err = dalphas.Set(0)
	if err != nil {
		return err
	}
	config := h.LaunchConfig3d(dims[1], dims[2], dims[3])
	for i := int32(0); i < dims[0]; i++ {
		err := xA.bwdmode.Launch(config.BlockCountx, config.BlockCounty, config.BlockCountz,
			config.ThreadPerBlockx, config.ThreadPerBlocky, config.ThreadPerBlockz, 0, h.s,
			config.Dimx, config.Dimy, config.Dimz, i, alpha, beta, x, dx, dy, coefs, dcoefs, NHWC, xA.propnan)
		if err != nil {
			return err
		}
	}
	return nil

}
func (xA *XActivationD) backpropropleaky(h *XHandle, alpha, beta CScalar, xD *TensorD, x *Malloced, dxD *TensorD, dx *Malloced, dyD *TensorD, dy *Malloced) error {
	dtype, _, _, err := xD.GetDescrptor()
	if err != nil {
		return err
	}
	var df DataTypeFlag
	if dtype != df.Float() {
		return errors.New("Only Float is the supported datatype")
	}
	sib, err := xD.GetSizeInBytes()
	if err != nil {
		return err
	}
	length := FindLength(sib, dtype)

	config := h.LaunchConfig(int32(length))
	//	fmt.Println("alpha beta coef length", alpha, beta, float32(xA.coef))
	//	fmt.Println("Config:", config)
	return xA.bwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, alpha, beta, x, dx, dy, float32(xA.coef), xA.propnan) //, xA.propnan)
}
func (xA *XActivationD) backpropParaPlus(h *XHandle, alpha, beta CScalar, xD *TensorD, x *Malloced, dxD *TensorD, dx *Malloced, dyD *TensorD, dy *Malloced, coefs, dcoefs *Malloced) error {
	dtype, dims, _, err := xD.GetDescrptor()
	if err != nil {
		return err
	}
	var df DataTypeFlag
	if dtype != df.Float() {
		return errors.New("Only Float is the supported datatype")
	}

	length := findvolume(dims[1:])
	config := h.LaunchConfig(int32(length))
	for i := int32(0); i < dims[0]; i++ {
		err := xA.bwdmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, i, alpha, beta, x, dx, dy, coefs, dcoefs, xA.propnan)
		if err != nil {
			return err
		}
	}
	return nil
}
