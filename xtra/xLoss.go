package xtra

import (
	"C"
	"errors"

	"github.com/dereklstinson/GoCuNets/utils"

	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cuda"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/kernels"
	"github.com/dereklstinson/cutil"
	"github.com/dereklstinson/half"
)

//XLossD is the loss descriptor for the loss function
type XLossD struct {
	mode         XLossMode
	lossfunc     *cuda.Kernel
	lossfuncfp16 *cuda.Kernel
	loss         cutil.Mem
	lossfp16     cutil.Mem
	cpuloss      []float32
	cpulossfp16  []half.Float16
	cpuptr       cutil.Mem
	cpuptrfp16   cutil.Mem
	flg          XLossModeFlag
	dflg         gocudnn.DataType
	memcopykind  cudart.MemcpyKind
}

//XLossModeFlag passes XLossMode flags through methods
type XLossModeFlag struct {
}

//MSE is Mean Squared Error
func (x XLossModeFlag) MSE() XLossMode {
	return XLossMode(1)
}

/*
//SoftMaxAverage does the average -log loss for the output
func (x XLossModeFlag) SoftMaxAverage() XLossMode {
	return XLossMode(2)
}
*/

//XLossMode are the flags for XLoss
type XLossMode int

//SofMaxLogLoss holds the function needed to do the log loss of the soft max function
type SofMaxLogLoss struct {
	previouscpuloss float32
	cpuloss         []float32
	cpulossptr      cutil.Mem
	gpuloss         *gocu.CudaPtr
	smloss          *cuda.Kernel
	memcopykind     cudart.MemcpyKind
}

func newsofmaxneglogloss(h *Handle) (smll *SofMaxLogLoss, err error) {

	var ktf kernels.XtraKerns
	smll = new(SofMaxLogLoss)
	smll.memcopykind.Default()
	smll.cpuloss = make([]float32, 1)
	smll.gpuloss = new(gocu.CudaPtr)
	smll.cpulossptr, err = gocu.MakeGoMem(smll.cpuloss)
	if err != nil {
		return nil, err
	}
	err = cudart.MallocManagedGlobal(smll.gpuloss, 4)
	if err != nil {
		return nil, err
	}

	smll.smloss, err = cuda.MakeKernel(ktf.SoftMaxAverageLoss(), h.mod)
	if err != nil {
		return nil, err
	}

	return smll, nil

}

//NewSoftMaxNegLogLoss creates a softmaxlogloss handler.
func NewSoftMaxNegLogLoss(h *Handle) (*SofMaxLogLoss, error) {
	var smll *SofMaxLogLoss
	var err error
	if h.w != nil {
		err = h.w.Work(func() error {
			var werr error
			smll, werr = newsofmaxneglogloss(h)
			if werr != nil {
				return werr
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	} else {
		smll, err = newsofmaxneglogloss(h)
		if err != nil {
			return nil, err
		}
	}
	return smll, nil

}

//FindAverageLogLoss returns the average log loss
func (s *SofMaxLogLoss) FindAverageLogLoss(h *Handle, alpha float64,
	yD *gocudnn.TensorD, y cutil.Mem, beta float64,
	targetD *gocudnn.TensorD, target cutil.Mem) (loss float32, err error) {
	s.previouscpuloss = s.cpuloss[0]
	err = cudart.Memset(s.gpuloss, 0, 4)
	if err != nil {
		return -1, errors.New("(s *SofMaxLogLoss) FindAverageLogLoss(): GPU mem not set")
	}
	frmt, dtype, dims, stride, err := yD.Get()
	if err != nil {
		return -1, err
	}
	fflg := frmt
	var ntarget int32

	switch frmt {
	case fflg.NCHW():
		//the target values are found on the channel dim.
		// So the number of targets will be a multipe of the batches and other spacial dims.
		// So you can find the by taking the volume of the tensor and divide it by the number of channels
		ntarget = utils.FindVolumeInt32(dims, stride) / dims[1]

	case fflg.NHWC():
		//the target values are found on the channel dim.
		// So the number of targets will be a multipe of the batches and other spacial dims.
		// So you can find the by taking the volume of the tensor and divide it by the number of channels
		ntarget = utils.FindVolumeInt32(dims, stride) / dims[len(dims)-1]
	default:
		return -1, errors.New("(s *SofMaxLogLoss) FindAverageLogLoss Unsupported Format")
	}
	dflg := dtype
	switch dtype {

	case dflg.Float():
		length := utils.FindVolumeInt32(dims, stride)
		config := h.LaunchConfig(int32(length))
		err = s.smloss.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, ntarget, target, y, s.gpuloss)
		if err != nil {
			return -1, err
		}
		if h.s != nil {
			err = h.s.Sync()
			if err != nil {
				return -1, err
			}
		} else {
			err = cudart.SyncNillStream()
			if err != nil {
				return -1, nil
			}
		}

		err = cudart.MemCpy(s.cpulossptr, s.gpuloss, 4, s.memcopykind)
		if err != nil {
			return -1, err
		}
		if h.s != nil {
			err = h.s.Sync()
			if err != nil {
				return -1, err
			}
		} else {
			err = cudart.SyncNillStream()
			if err != nil {
				return -1, nil
			}
		}
		s.cpuloss[0] = s.cpuloss[0]*float32(alpha) + s.previouscpuloss*float32(beta)
		err = cudart.MemCpy(s.gpuloss, s.cpulossptr, 4, s.memcopykind)
		return s.cpuloss[0], nil
	default:
		return -1, errors.New("(s *SofMaxLogLoss) FindAverageLogLoss: Unsupported Tensor Type ")
	}

}

//NewLossDescriptor creates a loss destriptor to calculate loss
func NewLossDescriptor(h *Handle, mode XLossMode) (*XLossD, error) {
	var loss *XLossD
	var err error
	if h.w != nil {
		err = h.w.Work(func() error {
			loss, err = newLossDescriptor(h, mode)
			return err
		})
	} else {
		loss, err = newLossDescriptor(h, mode)
	}

	return loss, err
}

func newLossDescriptor(h *Handle, mode XLossMode) (*XLossD, error) {
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
		gpu16 := new(gocu.CudaPtr)
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
		cpulossfp16 := make([]half.Float16, 1)
		cpuptr, err := gocu.MakeGoMem(cpuloss)
		cpuptrfp16, err := gocu.MakeGoMem(cpulossfp16)
		if err != nil {
			return nil, err
		}
		return &XLossD{
			mode:         mode,
			lossfunc:     mse,
			lossfuncfp16: msefp16,
			loss:         gpu,
			lossfp16:     gpu16,
			cpuloss:      cpuloss,
			cpulossfp16:  cpulossfp16,
			cpuptr:       cpuptr,
			cpuptrfp16:   cpuptrfp16,
			memcopykind:  memflg.DeviceToHost(),
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
	var loss float32
	var err error
	if h.w != nil {
		err = h.w.Work(func() error {
			loss, err = l.calculateErrorAndLoss(h, dxD, dx, yD, y, dyD, dy, alpha, beta)
			return err
		})
	} else {
		loss, err = l.calculateErrorAndLoss(h, dxD, dx, yD, y, dyD, dy, alpha, beta)
	}
	return loss, err
}
func (l *XLossD) calculateErrorAndLoss(h *Handle,
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
		switch dxdtype {
		case l.dflg.Float():
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
		case l.dflg.Half():
			a2 := float32(alpha)
			b2 := float32(beta)
			a := half.NewFloat16(a2)
			b := half.NewFloat16(b2)
			config := h.LaunchConfig(int32(length))
			sharedmem := uint32(4)
			err = l.lossfuncfp16.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, sharedmem, h.s, config.Elements, dx, dy, y, l.lossfp16, a, b)
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
			return l.cpulossfp16[0].Float32() / batch, err
		}

	}
	return 0, errors.New("Unsupported Loss Function")
}

//MSE Loss takes (const int length, *target ,*networkout, *errors, *loss)
