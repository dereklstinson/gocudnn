package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//BatchNormModeFlag used to pass BatchNormMode Flags user safe like using methods
type BatchNormModeFlag struct {
}

//BatchNormMode used for BatchNormMode Flags
type BatchNormMode C.cudnnBatchNormMode_t

//PerActivation return  BatchNormMode(C.CUDNN_BATCHNORM_PER_ACTIVATION) flag
func (bnm BatchNormModeFlag) PerActivation() BatchNormMode {
	return BatchNormMode(C.CUDNN_BATCHNORM_PER_ACTIVATION)
}

//Spacial returns  BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL) flag
func (bnm BatchNormModeFlag) Spacial() BatchNormMode {
	return BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL)
}

// SpatialPersistent returns  BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL_PERSISTENT) flag
func (bnm BatchNormModeFlag) SpatialPersistent() BatchNormMode {
	return BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL_PERSISTENT)
}
func (bnm BatchNormMode) c() C.cudnnBatchNormMode_t { return C.cudnnBatchNormMode_t(bnm) }

const bnMinEpsilon = float64(1e-5)
