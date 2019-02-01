package gocudnn

import (
	"errors"

	"github.com/dereklstinson/GoCudnn/kernels"
)

//XInt8ToFloatD converts an Int8 to A Float
type XInt8ToFloatD struct {
	inttofloat *Kernel
	//innerswaploweruper *Kernel
	//innerswapbatch     *Kernel
	inttofloatnorm *Kernel
	dflg           DataTypeFlag
}

//MakeIntToFloatD will return an IntToFloatD which will do a conversion from ints to float.  It is a good way to save on memory.
func (x Xtra) MakeIntToFloatD(handle *XHandle) (*XInt8ToFloatD, error) {
	var kern kernels.XtraKerns
	var cu Cuda
	k1, err := cu.MakeKernel(kern.Int8ToFloat32(), handle.mod)
	if err != nil {
		return nil, err
	}
	k2, err := cu.MakeKernel(kern.Int8ToFloat32Normalize(), handle.mod)
	if err != nil {
		return nil, err
	}
	return &XInt8ToFloatD{
		inttofloat:     k1,
		inttofloatnorm: k2,
	}, nil
}

//Int8ToFloat converts an Int8 intput to a FLoat
func (xint *XInt8ToFloatD) Int8ToFloat(handle *XHandle, xD *TensorD, x *Malloced, yD *TensorD, y *Malloced, normal bool) error {

	xdata, xdims, _, err := xD.GetDescrptor()
	if err != nil {
		return err
	}
	if xdata != xint.dflg.Int8() {
		return errors.New("Input Data Type Needs to be Int8")
	}
	ydata, ydims, _, err := yD.GetDescrptor()
	if err != nil {
		return err
	}
	if ydata != xint.dflg.Float() {
		return errors.New("Output Data Type Needs to be Float")
	}
	if findvol(xdims) != findvol(ydims) {
		return errors.New("Size of xD, and yD tensors need to be the same")
	}

	cfg := handle.LaunchConfig(findvol(xdims))
	if normal {
		return xint.inttofloatnorm.Launch(cfg.BlockCount, 1, 1, cfg.ThreadPerBlock, 1, 1, 0, handle.s, cfg.Elements, x, y)
	}
	return xint.inttofloat.Launch(cfg.BlockCount, 1, 1, cfg.ThreadPerBlock, 1, 1, 0, handle.s, cfg.Elements, x, y)

}
