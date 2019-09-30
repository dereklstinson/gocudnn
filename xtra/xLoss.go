package xtra

import (
	"C"
	"errors"

	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cuda"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/kernels"
	"github.com/dereklstinson/cutil"
)


//XLossD is the loss descriptor for the loss function
type XLossD struct {
	mode        XLossMode
	lossfunc    *cuda.Kernel
	lossfuncfp16 *cuda.Kernel
	loss        cutil.Mem
	lossfp16 cutil.Mem
	cpuloss     []float32
	cpulossfp16 []half.Float16
	cpuptr      cutil.Mem
	cpuptrfp16
	flg         XLossModeFlag
	dflg        gocudnn.DataType
	memcopykind cudart.MemcpyKind
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
func NewLossDescriptor(h *Handle, mode XLossMode) (*XLossD, error) {
	var ktf kernels.XtraKerns
	var flg XLossModeFlag
	var memflg cudart.MemcpyKind
	switch mode {
	case flg.MSE():
		mse, err := cuda.MakeKernel(ktf.MSELoss(), h.mod)
		msefp16, err := cuda.MakeKernel(ktf.MSELossFP16(), h.mod)
		if err != nil {
			return nil, err
		}

		gpu := new(gocu.CudaPtr)
		gpu16:=new(gocu.CudaPtr)
		err = cudart.MallocManagedGlobal(gpu, 4)
		err = cudart.MallocManagedGlobal(gpu16, 2)
		//	gpu, err := nvidia.MallocGlobal(h, 4)
		if err != nil { 
			return nil, err
		}
		cudart.Memset(gpu, 0, 4)
		if err != nil {
			return nil, err
		}
		cudart.Memset(gpu16, 0, 2)
		if err != nil {
			return nil, err
		}
		cpuloss := make([]float32, 1)
		cpulossfp16 :=make([]half.Float16,1)
		cpuptr, err := gocu.MakeGoMem(cpuloss)
		cpuptrfp16, err := gocu.MakeGoMem(cpulossfp16)
		if err != nil {
			return nil, err
		}
		return &XLossD{
			mode:        mode,
			lossfunc:    mse,
			loss:        gpu,
			lossfp16:gpu16,
			cpuloss:     cpuloss,
			cpulossfp16: cpulossfp16,
			
			cpuptr:      cpuptr,
			cpuptrfp16:cpuptrfp16,
			memcopykind: memflg.DeviceToHost(),
		}, nil

	}
	return nil, errors.New("Not a supported Loss Mode")
}

//CalculateErrorAndLoss calculates the error going back and the loss going forward dxD yD dyD need to have the same dims and size
//and right now they can only be datatype float
func (l *XLossD) CalculateErrorAndLoss(h *Handle,
	dxD *gocudnn.TensorD, //output -errors going back
	dx cutil.Mem, // output -errors going back
	yD *gocudnn.TensorD, //input - target values
	y cutil.Mem, //input -target values
	dyD *gocudnn.TensorD, //input network output values
	dy cutil.Mem, //input network output values
	alpha, beta float64,
) (float32, error) {
	_, dxdtype, dxdims, _, err := dxD.Get()
	if err != nil {
		return -1, err
	}
	_, ydtype, ydims, _, err := yD.Get()
	if err != nil {
		return -1, err
	}
	_, dydtype, dydims, _, err := dyD.Get()
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

	if err != nil {
		return 0, err
	}
	switch l.mode {
	case l.flg.MSE():
		err = h.s.Sync()
		if err != nil {
			return -1, err
		}
		switch dxdtype{
		case l.dlflg.Float():
			a := float32(alpha)
			b := float32(beta)
			config := h.LaunchConfig(int32(length))
			err = l.lossfunc.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dx, dy, y, l.loss, a, b)
			if err != nil {
				return -1, err
			}
			err = h.s.Sync()
			if err != nil {
				return -1, err
			}
			err = cudart.MemCpy(l.cpuptr, l.loss, 4, l.memcopykind)
	
			err = h.s.Sync()
			if err != nil {
				return -1, err
			}
			return l.cpuloss[0] / batch, err
		case l.dlflg.Half():
			a2 := float32(alpha)
			b2 := float32(beta)
			a:=half.ToFloat16(a2)
			b:=half.ToFloat16(b2)
			config := h.LaunchConfig(int32(length))
			err = l.lossfunc.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dx, dy, y, l.lossfp16, a, b)
			if err != nil {
				return -1, err
			}
			err = h.s.Sync()
			if err != nil {
				return -1, err
			}
			err = cudart.MemCpy(l.cpuptrfp16, l.lossfp16, 2, l.memcopykind)
	
			err = h.s.Sync()
			if err != nil {
				return -1, err
			}
			return half.ToFloat32(l.cpulossfp16[0] )/ batch, err
		}
		

	}
	return 0, errors.New("Unsupported Loss Function")
}

//MSE Loss takes (const int length, *target ,*networkout, *errors, *loss)
