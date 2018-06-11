package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//ActivationModeFlag is used to "safely" pass flags by the use of methods.
//If it was me I would call it like:
//var ActMode gocudnn.ActivationModeFlag
//Then use the methods like ActMode.Sigmoid() to pass the sigmoid flag.
type ActivationModeFlag struct {
}

//ActivationMode is used for activation discriptor flags use ActivationModeFlag struct to pass flags through its methods
type ActivationMode C.cudnnActivationMode_t

//Sigmoid returnsActivationMode(C.CUDNN_ACTIVATION_SIGMOID)
func (a ActivationModeFlag) Sigmoid() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_SIGMOID)
}

//Relu returns ActivationMode(C.CUDNN_ACTIVATION_RELU)
func (a ActivationModeFlag) Relu() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_RELU)
}

//Tanh returns ActivationMode(C.CUDNN_ACTIVATION_TANH)
func (a ActivationModeFlag) Tanh() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_TANH)
}

//ClippedRelu returns  ActivationMode(C.CUDNN_ACTIVATION_CLIPPED_RELU)
func (a ActivationModeFlag) ClippedRelu() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_CLIPPED_RELU)
}

//Elu returns  ActivationMode(C.CUDNN_ACTIVATION_ELU)
func (a ActivationModeFlag) Elu() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_ELU)
}

//Identity returns  ActivationMode(C.CUDNN_ACTIVATION_IDENTITY)
func (a ActivationModeFlag) Identity() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_IDENTITY)
}

func (a ActivationMode) c() C.cudnnActivationMode_t { return C.cudnnActivationMode_t(a) }
