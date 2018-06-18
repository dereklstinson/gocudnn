package gocudnn

import (
	"errors"
)

/*

Written in the style of cudnn/GoCudnn. This is an added set of functions to calculate loss.

*/

//TrainerD is the descriptor of the trainer
type TrainerD struct {
	data    DataType
	mode    TrainingMode
	params  TrainingParams
	l1loss  CScalar
	l2loss  CScalar
	counter CScalar
}

//TrainingParams is a struct can be use for training params.
//When selecting the training mode the params that are not part of the training mode will be ignored.
type TrainingParams struct {
	Decay1   CScalar
	Decay2   CScalar
	Ro       CScalar
	Eps      CScalar
	Momentum CScalar
	Rate     CScalar
	Beta1    CScalar
	Beta2    CScalar
}

//TrainingModeFlag is a nil struct that passes TrainingMode Flags through methods
type TrainingModeFlag struct {
}

//TrainingMode are flags to pass for training mode
type TrainingMode int32

//Vanilla Update Weight Mode
func (t TrainingModeFlag) Vanilla() TrainingMode {
	return TrainingMode(0)
}

//Momentum does the momentum algo
func (t TrainingModeFlag) Momentum() TrainingMode {
	return TrainingMode(1)
}

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
func NewTrainingDescriptor(mode TrainingMode, data DataType, params TrainingParams) (*TrainerD, error) {
	if mode < 0 || mode > 4 {
		return nil, errors.New("NewTrainingDescriptor: Unsupported Trainingmode")
	}
	var flg Flags
	dt := flg.DataType
	switch data {
	case dt.Double(): //this is just used to check if it is true.
	case dt.Float():
	case dt.Int32():
	case dt.Int8():
	case dt.UInt8():
	default:
		return nil, errors.New("NewTrainingDescriptor: unsupported Datatype") //if not true then return error
	}
	return &TrainerD{ //all true then we will set TrainerD
		mode:   mode,
		data:   data,
		params: params,
	}, nil
}

//GetTrainingDescriptor returns the info that was set for the training descriptor
func (d *TrainerD) GetTrainingDescriptor() (TrainingMode, DataType, TrainingParams) {
	return d.mode, d.data, d.params
}

//ChangeTrainingParams passes a TrainingParam struct and it changes the values of the training parameters.
func (d *TrainerD) ChangeTrainingParams(params TrainingParams) {
	d.params = params
}

/*

//Connection is one thing that most of the neurons have in common.  All but the pooling neurons
type Connection struct {
	Weight        float64
	Gradientadder float64
	deltaweight   float64
	decay1        float64
	decay2        float64
	ro            float64
	eps           float64
	momentum      float64
	rate          float64
	trainingtype  string

	gsum float64
	xsum float64

	counter uint64
	beta1   float64
	beta2   float64
}



decay1 := link.decay1
if link.Weight < 0 {
	decay1 = -decay1
}
decay2 := float64(link.Weight * link.decay2)
link.Gradientadder = (link.Gradientadder / float64(batch)) + decay1 + decay2

if link.trainingtype == "adagrad" {
	link.deltaweight = link.deltaweight + float64(link.Gradientadder*link.Gradientadder)
	link.Weight += -(link.rate * link.Gradientadder) / (math.Sqrt(link.deltaweight) + link.eps)
} else if link.trainingtype == "rmsprop" {
	decayrate := .999
	link.deltaweight = decayrate*link.deltaweight + (1-decayrate)*link.Gradientadder*link.Gradientadder
	x := -link.rate * link.Gradientadder / (math.Sqrt(link.deltaweight) + link.eps)
	link.Weight += x
} else if link.trainingtype == "adadelta" {
	link.gsum = float64(link.ro*link.gsum) + (1-link.ro)*link.Gradientadder*link.Gradientadder
	holder := -math.Sqrt((link.xsum+link.eps)/(link.gsum+link.eps)) * link.Gradientadder
	link.xsum = float64(link.ro*link.xsum) + ((1 - link.ro) * holder * holder)
	link.Weight += holder

} else if link.trainingtype == "adam" {
	link.gsum = float64(link.beta1*link.gsum) + float64((1.0-link.beta1)*link.Gradientadder)
	gsumt := link.gsum / (1.0 - math.Pow(link.beta1, float64(link.counter)))
	link.xsum = (link.beta2 * link.xsum) + ((1.0 - link.beta2) * link.Gradientadder * link.Gradientadder)
	xsumt := link.xsum / (1.0 - math.Pow(link.beta2, float64(link.counter)))
	link.Weight += -(link.rate * gsumt) / (math.Sqrt(xsumt) + link.eps)

} else if link.trainingtype == "momentum" {
	link.deltaweight = (link.deltaweight * link.momentum) - (link.Gradientadder * link.rate)
	link.Weight += link.deltaweight

} else {
	link.Weight += -(link.rate * link.Gradientadder)
}

link.Gradientadder = 0
link.counter++

*/
