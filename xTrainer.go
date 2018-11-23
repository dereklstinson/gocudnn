package gocudnn

import "C"
import (
	"errors"
	"strconv"

	"github.com/dereklstinson/GoCudnn/kernels"
)

/*
 Since momentum and vanilla can be made with optensor. Only AdaGrad, AdaDelta, and Adam are going to be used.  I might add more if my thesis requires it.
 L1 and L2 regularization are available too.  If you don't want it then too bad.  Just write your own functions using the kernels subpackage. :-)
Currently only float is available for training.  I will make a double for the trainer, too. but that will be later.
Trainers get there own Context. Which is different than the cudnn handle.  So, make sure you make a Cuda variable.
example:

var cu gocudnn.Cuda

ctx,err:= cu.CtxCreate(flag,device)

Written in the style of cudnn/GoCudnn. This is an added set of functions to calculate loss.

*/

//TrainerD is the descriptor of the trainer
type TrainerD struct {
	data    DataType
	mode    TrainingMode
	counter uint32
	kmode   *Kernel
	kreg    *Kernel
}

//RegParams holds the regulator paramaters
type RegParams struct {
	decay1 float32
	decay2 float32
	batch  float32
}

//CreateRegParamsFloat32 creates the RegParams for float32 ,,, I really don't like this function.
//It was kind of a shortcut since I use a gtx1080ti. Since the new rtx line has come out. Users will probably want to take
//advantage of other types than single precision.
func (xtra Xtra) CreateRegParamsFloat32(decay1, decay2, batch float32) RegParams {
	return RegParams{
		decay1: decay1,
		decay2: decay2,
		batch:  batch,
	}
}

//SetDecay1 sets decay1
func (a *RegParams) SetDecay1(decay1 float32) {
	a.decay1 = decay1
}

//SetDecay2 sets decay 2
func (a *RegParams) SetDecay2(decay2 float32) {
	a.decay2 = decay2
}

//SetBatch SetsBatch
func (a *RegParams) SetBatch(batch float32) {
	a.batch = batch
}

//TrainingParams is a struct can be use for training params.
//When selecting the training mode the params that are not part of the training mode will be ignored.
type TrainingParams struct {
	eps   float32
	rate  float32
	beta1 float32
	beta2 float32
}

//SetBeta1 sets beta1
func (a *TrainingParams) SetBeta1(beta1 float32) {
	a.beta1 = beta1
}

//SetBeta2 sets beta2
func (a *TrainingParams) SetBeta2(beta2 float32) {
	a.beta2 = beta2
}

//SetRate sets rate
func (a *TrainingParams) SetRate(rate float32) {
	a.rate = rate
}

//SetEps sets eps
func (a *TrainingParams) SetEps(eps float32) {
	a.eps = eps
}

//CreateParamsFloat32 creates float32 paramaters for the different types of optimization
func (xtra Xtra) CreateParamsFloat32(eps, rate, beta1, beta2 float32) TrainingParams {
	return TrainingParams{

		eps:   eps,
		rate:  rate,
		beta1: beta1,
		beta2: beta2,
	}
}

//TrainingModeFlag is a nil struct that passes TrainingMode Flags through methods.
type TrainingModeFlag struct {
}

//TrainingMode are flags to pass for training mode
type TrainingMode int32

//AdaGrad performs the adagrad algo
func (t TrainingModeFlag) AdaGrad() TrainingMode {
	return TrainingMode(2)
}

//AdaDelta Performs the adadelta algo
func (t TrainingModeFlag) AdaDelta() TrainingMode {
	return TrainingMode(3)
}

//Adam performs adam function
func (t TrainingModeFlag) Adam() TrainingMode {
	return TrainingMode(4)
}
func (t TrainingMode) tostring() string {
	f := TrainingModeFlag{}
	x := kernels.XtraKerns{}
	switch t {
	case f.Adam():
		return x.Adam()
	case f.AdaDelta():
		return x.AdaDelta()
	case f.AdaGrad():
		return x.AdaGrad()
	}
	return "Not Supported"
}

//NewTrainingDescriptor Creates and sets a TrainingD.  All modes get decay1, decay2, rate, -- all but vanilla get eps,
func (xtra Xtra) NewTrainingDescriptor(h *XHandle, mode TrainingMode, data DataType) (*TrainerD, error) {
	var ktf kernels.XtraKerns
	var cu Cuda

	regname := ktf.L1L2()

	var mflg TrainingModeFlag
	var mname string
	switch mode {
	case mflg.AdaDelta():
		mname = ktf.AdaDelta()
	case mflg.AdaGrad():
		mname = ktf.AdaGrad()
	case mflg.Adam():
		mname = ktf.Adam()
	default:
		return nil, errors.New("TrainingMode Not Supported")
	}

	var tflag Tensor
	dt := tflag.Flgs.Data
	switch data {

	case dt.Float(): //this is just used to check if it is true.
	//case dt.Double():
	default:
		return nil, errors.New("NewTrainingDescriptor: unsupported Datatype") //if not true then return error
	}
	kmode, err := cu.MakeKernel(mname, h.mod)
	if err != nil {
		return nil, err
	}
	kreg, err := cu.MakeKernel(regname, h.mod)
	if err != nil {
		return nil, err
	}
	if err != nil {
		return nil, err
	}
	return &TrainerD{ //all true then we will set TrainerD
		mode:    mode,
		data:    data,
		kmode:   kmode,
		kreg:    kreg,
		counter: uint32(1),
	}, nil
}

//GetTrainingDescriptor returns the info that was set for the training descriptor
func (d *TrainerD) GetTrainingDescriptor() (TrainingMode, DataType) {
	return d.mode, d.data
}
func (d *TrainerD) adam(gx, gy, gz, bx, by, bz, shared uint32, stream *Stream, length int32, w, gsum, xsum, dw *Malloced, rate, beta1, beta2, eps, counter interface{}) error {
	return d.kmode.Launch(gx, gy, gz, bx, by, bz, shared, stream, length, w, gsum, xsum, dw, rate, beta1, beta2, eps, counter)
}

//L1L2Regularization does the l1l2 regularization
func (d *TrainerD) L1L2Regularization(h *XHandle, dw, w, l1, l2 *Malloced, params RegParams) error {
	var size int32
	switch d.data {
	case DataTypeFlag{}.Float():
		size = int32(w.ByteSize() / SizeT(4))
	default:
		return errors.New("Unsupported Type")

	}
	config := h.LaunchConfig(size)
	if params.decay1 == 0 && params.decay2 == 0 {
		return d.kreg.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dw, w, nil, nil, params.batch, params.decay1, params.decay2)
	} else if params.decay1 == 0 {
		return d.kreg.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dw, w, nil, l2, params.batch, params.decay1, params.decay2)
	} else if params.decay2 == 0 {
		return d.kreg.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dw, w, l1, nil, params.batch, params.decay1, params.decay2)
	} else {
		return d.kreg.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, dw, w, l1, l2, params.batch, params.decay1, params.decay2)
	}

	//return errors.New("Shouldn't have Reached here")

}

//TrainValues  Adagrad requires gsum, but not xsum.  If Adagrad is used then  nil can be passed for xsum.
func (d *TrainerD) TrainValues(h *XHandle, dw, w, gsum, xsum *Malloced, params TrainingParams) error {
	var size int32
	var err error
	if xsum != nil {
		if w.ByteSize() != gsum.ByteSize() || w.ByteSize() != xsum.ByteSize() || w.ByteSize() != dw.ByteSize() {
			sp := " "
			wbs := strconv.Itoa(int(w.ByteSize()))
			dwbs := strconv.Itoa(int(dw.ByteSize()))
			gsbs := strconv.Itoa(int(gsum.ByteSize()))
			xsbs := strconv.Itoa(int(xsum.ByteSize()))
			return errors.New("Sizes don't match" + sp + wbs + sp + dwbs + sp + gsbs + sp + xsbs)
		}

	} else {
		if w.ByteSize() != gsum.ByteSize() || w.ByteSize() != dw.ByteSize() {
			sp := " "
			wbs := strconv.Itoa(int(w.ByteSize()))
			dwbs := strconv.Itoa(int(dw.ByteSize()))
			gsbs := strconv.Itoa(int(gsum.ByteSize()))
			return errors.New("Sizes don't match" + sp + wbs + sp + dwbs + sp + gsbs + sp)
		}

	}

	var dflg DataTypeFlag
	switch d.data {
	case dflg.Float():
		size = int32(w.ByteSize() / SizeT(4))
	default:
		return errors.New("Unsupported Type")
	}

	config := h.LaunchConfig(size)

	switch d.mode {
	case TrainingModeFlag{}.Adam():

		err = d.adam(config.BlockCount, uint32(1), uint32(1), config.ThreadPerBlock, uint32(1), uint32(1), 0, h.s, config.Elements, w, gsum, xsum, dw, params.rate, params.beta1, params.beta2, params.eps, float32(d.counter))
		if err != nil {
			return err
		}
		d.counter++
		if d.counter < 1 {
			d.counter = 1
		}

		return nil

	case TrainingModeFlag{}.AdaDelta():
		config := h.LaunchConfig(size)
		err = d.kmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, w, gsum, dw, params.rate, params.eps)
		if err != nil {
			return err
		}
	case TrainingModeFlag{}.AdaGrad():
		config := h.LaunchConfig(size)
		err = d.kmode.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, h.s, config.Elements, w, gsum, dw, params.rate, params.eps)
		if err != nil {
			return err
		}
	default:
		return errors.New("Unsopported Training Mode")
	}
	return nil
}
