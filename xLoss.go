package gocudnn

import (
	"errors"

	"github.com/dereklstinson/GoCudnn/kernels"
)

//XLossD is the loss descriptor for the loss function
type XLossD struct {
	mode     XLossMode
	lossfunc *Kernel
	loss     *Malloced
	flg      XLossModeFlag
	dflg     DataTypeFlag
	memflg   MemcpyKindFlag
	managed  bool
}

//XLossModeFlag passes XLossMode flags through methods
type XLossModeFlag struct {
}

//MSE is Mean Squared Error
func (x XLossModeFlag) MSE() XLossMode {
	return XLossMode(1)
}

//XLossMode are the flags for XLoss
type XLossMode int

//NewLossDescriptor creates a loss destriptor to calculate loss
func (xtra Xtra) NewLossDescriptor(h *XHandle, mode XLossMode, unified bool) (*XLossD, error) {
	var cu Cuda
	var ktf kernels.XtraKerns
	var flg XLossModeFlag
	switch mode {
	case flg.MSE():
		mse, err := cu.MakeKernel(ktf.MSELoss(), h.mod)
		if err != nil {
			return nil, err
		}
		if unified == true {
			lossptr, err := UnifiedMangedGlobal(SizeT(4))
			lossptr.Set(0)
			if err != nil {
				return nil, err
			}
			return &XLossD{
				mode:     mode,
				lossfunc: mse,
				loss:     lossptr,
				managed:  true,
			}, nil
		}
		lossptr, err := Malloc(SizeT(4))
		lossptr.Set(0)
		if err != nil {
			return nil, err
		}
		return &XLossD{
			mode:     mode,
			lossfunc: mse,
			loss:     lossptr,
			managed:  false,
		}, nil

	}
	return nil, errors.New("Not a supported Loss Mode")
}

//CalculateErrorAndLoss calculates the error going back and the loss going forward dxD yD dyD need to have the same dims and size
//and right now they can only be datatype float
func (l *XLossD) CalculateErrorAndLoss(h *XHandle,
	dxD *TensorD, //output -errors going back
	dx Memer, // output -errors going back
	yD *TensorD, //input - target values
	y Memer, //input -target values
	dyD *TensorD, //input network output values
	dy Memer, //input network output values
	//loss Memer, // this is the loss calculated by the network
) (float32, error) {
	dxdtype, dxdims, _, err := dxD.GetDescrptor()
	if err != nil {
		return -1, err
	}
	ydtype, ydims, _, err := yD.GetDescrptor()
	if err != nil {
		return -1, err
	}
	dydtype, dydims, _, err := dyD.GetDescrptor()
	if err != nil {
		return -1, err
	}
	if dxdtype != ydtype || dxdtype != dydtype {

		return -1, errors.New("descriptors datatype not matching")
	}
	if dxdtype != l.dflg.Float() {
		return -1, errors.New("Only Supported Datatype is Float for now")
	}
	if comparedims(dxdims, ydims, dydims) == false {
		return -1, errors.New("Dims for tensors Don't Match")
	}
	length := findvolume(dxdims)
	batch := float32(dxdims[0])
	switch l.mode {
	case l.flg.MSE():
		config := h.LaunchConfig(int32(length))
		err = l.lossfunc.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dx, dy, y, l.loss)
		if err != nil {
			return -1, err
		}
		loss := make([]float32, 1)
		ptr, _ := MakeGoPointer(loss)
		if l.managed == true {
			err = CudaMemCopy(ptr, l.loss, 4, l.memflg.Default())
			return loss[0] / batch, err
		}
		err = CudaMemCopy(ptr, l.loss, 4, l.memflg.DeviceToHost())
		return loss[0] / batch, err

	}
	return 0, errors.New("Unsupported Loss Function")
}

//MSE Loss takes (const int length, *target ,*networkout, *errors, *loss)
