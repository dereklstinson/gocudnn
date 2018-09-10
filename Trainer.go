package gocudnn

import "C"
import (
	"errors"

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
	data DataType
	mode TrainingMode
	reg  Regularization
	//	params  TrainingParams
	l1loss  CScalar
	l2loss  CScalar
	counter CScalar
	kmode   *Kernel
	kreg    *Kernel
}

//TrainingParams is a struct can be use for training params.
//When selecting the training mode the params that are not part of the training mode will be ignored.
type TrainingParams struct {
	decay1 CScalar
	decay2 CScalar
	batch  CScalar
	ro     CScalar
	rate   CScalar
	beta1  CScalar
	beta2  CScalar
}
type TContext struct {
	ctx *Context
	mod *Module
	ptx string
}

func MakeTrainingContext(flags uint32, dev *Device, trainingfloatdir string) (*TContext, error) {
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
func NewTrainingDescriptor(tctx *TContext, mode TrainingMode, data DataType, reg Regularization) (*TrainerD, error) {
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
func (d *TrainerD) TrainValues(ctx *TContext, blocksize uint32, dw, w, l1, l2, gsum, xsum Memer, params TrainingParams) error {
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

	return d.kreg.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, nil, dw.Ptr(), w.Ptr(), l1, l2, params.batch, params.decay1, params.decay2)

}
