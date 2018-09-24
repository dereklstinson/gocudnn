package gocudnn

import "C"
import (
	"errors"

	"github.com/dereklstinson/GoCudnn/kernels"
)

//Contexter to use this it must return nil on the ones it is not. error will be saying that is not it. Helpful when making new packages
type Contexter interface {
	GetCudnnHandle() (*Handle, error)
	GetCudaContext() (*Context, error)
	GetTContext() (*TContext, error)
}

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
func (t *TContext) GetCudnnHandle() (*Handle, error) {
	return nil, errors.New("Not a CudnnHandle")
}
func (t *TContext) GetCudaContext() (*Context, error) {
	return nil, errors.New("Not a CudaContext")
}
func (t *TContext) GetTContext() (*TContext, error) {
	return t, nil
}

//Xtra is a holder for Xtra functions that are made by me, and not cuda or cudnn
type Xtra struct {
}

//TrainerD is the descriptor of the trainer
type TrainerD struct {
	data DataType
	mode TrainingMode
	reg  Regularization
	//	params  TrainingParams
	counter int32
	kmode   *Kernel
	kreg    *Kernel
}

//TrainingParams is a struct can be use for training params.
//When selecting the training mode the params that are not part of the training mode will be ignored.
type TrainingParams struct {
	decay1 interface{}
	decay2 interface{}
	batch  interface{}
	eps    interface{}
	rate   interface{}
	beta1  interface{}
	beta2  interface{}
}

//SetDecay1 sets decay1
func (a *TrainingParams) SetDecay1(decay1 interface{}) {
	a.decay1 = decay1
}

//SetDecay2 sets decay 2
func (a *TrainingParams) SetDecay2(decay2 interface{}) {
	a.decay2 = decay2
}

//SetBeta1 sets beta1
func (a *TrainingParams) SetBeta1(beta1 interface{}) {
	a.beta1 = beta1
}

//SetBeta2 sets beta2
func (a *TrainingParams) SetBeta2(beta2 interface{}) {
	a.beta2 = beta2
}

//SetRate sets rate
func (a *TrainingParams) SetRate(rate interface{}) {
	a.rate = rate
}

//SetEps sets eps
func (a *TrainingParams) SetEps(eps interface{}) {
	a.eps = eps
}
func (a *TrainingParams) SetBatch(batch interface{}) {
	a.batch = batch
}

func CreateParamsFloat32(decay1, decay2, batch, eps, rate, beta1, beta2 float32) TrainingParams {
	//c := CScalarConversion
	return TrainingParams{
		decay1: decay1,
		decay2: decay2,
		batch:  batch,
		eps:    eps,
		rate:   rate,
		beta1:  beta1,
		beta2:  beta2,
	}
}

type TContext struct {
	ctx *Context
	mod *Module
	ptx string
}

func (xtra Xtra) MakeTrainingContext(flags uint32, dev *Device, trainingfloatdir string) (*TContext, error) {
	var cu Cuda
	ctx, err := cu.CtxCreate(flags, dev)
	if err != nil {
		return nil, err
	}
	x := kernels.MakeMakeFile(trainingfloatdir, "trainingfloat", dev)
	kerncode := kernels.LoadPTXFile(trainingfloatdir, x)
	mod, err := cu.NewModule(kerncode)
	if err != nil {
		return nil, err
	}
	//	kern,err:=cu.MakeKernel()
	return &TContext{
		ctx: ctx,
		mod: mod,
		ptx: kerncode,
	}, nil
}

//Regularization will regulate the training.  L1 and/or L2
type Regularization int32

type RegularizationFlag struct {
}

func (r RegularizationFlag) L1() Regularization {
	return Regularization(1)
}
func (r RegularizationFlag) L2() Regularization {
	return Regularization(2)
}
func (r RegularizationFlag) L1L2() Regularization {
	return Regularization(12)
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

//NewTrainingDescriptor Creates and sets a TrainingD.  All modes get decay1, decay2, rate, -- all but vanilla get eps,
func (xtra Xtra) NewTrainingDescriptor(tctx *TContext, mode TrainingMode, data DataType, reg Regularization) (*TrainerD, error) {
	err := tctx.ctx.Push()
	var ktf kernels.TrainingFloat
	if err != nil {
		return nil, err
	}
	var cu Cuda

	var rflg RegularizationFlag
	var regname string
	switch reg {
	case rflg.L1():
		regname = ktf.L1()
	case rflg.L2():
		regname = ktf.L2()
	case rflg.L1L2():
		regname = ktf.L1L2()
	default:
		return nil, errors.New("Regularization Not Supported")
	}
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
	kmode, err := cu.MakeKernel(mname, tctx.mod)
	if err != nil {
		return nil, err
	}
	kreg, err := cu.MakeKernel(regname, tctx.mod)
	if err != nil {
		return nil, err
	}
	_, err = cu.CtxPopCurrent()
	if err != nil {
		return nil, err
	}
	return &TrainerD{ //all true then we will set TrainerD
		mode:  mode,
		data:  data,
		reg:   reg,
		kmode: kmode,
		kreg:  kreg,
	}, nil
}

//GetTrainingDescriptor returns the info that was set for the training descriptor
func (d *TrainerD) GetTrainingDescriptor() (TrainingMode, DataType, Regularization) {
	return d.mode, d.data, d.reg
}

//TrainValues. Adagrad requires gsum, but not xsum.  If Adagrad is used then  nil can be passed for xsum.
func (d *TrainerD) TrainValues(ctx *TContext, blocksize uint32, dw, w, l1, l2, gsum, xsum Memer, params TrainingParams) error { //Not working yet.
	var size uint32
	w.ByteSize()
	var dflg DataTypeFlag
	switch d.data {
	case dflg.Float():
		size = uint32(w.ByteSize() / SizeT(4))
	default:
		return errors.New("Unsupported Type")
	}
	gridsize := kernels.SimpleGridCalculator(blocksize, size)

	//var rflgs RegularizationFlag
	var cu Cuda
	err := ctx.ctx.Push()
	if err != nil {
		return err
	}
	defer cu.CtxPopCurrent()

	err = d.kreg.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, nil, dw.Ptr(), w.Ptr(), l1, l2, params.batch, params.decay1, params.decay2)
	if err != nil {
		return err
	}

	switch d.mode {
	case TrainingModeFlag{}.Adam():

		err = d.kmode.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, nil, w.Ptr(), gsum.Ptr(), xsum.Ptr(), dw.Ptr(), params.beta1, params.beta2, params.eps, float32(d.counter))
		if err != nil {
			return err
		}
		d.counter++
		return nil

	case TrainingModeFlag{}.AdaDelta():
		err = d.kmode.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, nil, w.Ptr(), gsum.Ptr(), dw.Ptr(), params.rate, params.eps)
		if err != nil {
			return err
		}
	case TrainingModeFlag{}.AdaGrad():
		err = d.kmode.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, nil, w.Ptr(), gsum.Ptr(), dw.Ptr(), params.rate, params.eps)
		if err != nil {
			return err
		}
	default:
		return errors.New("Unsopported Training Mode")
	}
	return nil
}
