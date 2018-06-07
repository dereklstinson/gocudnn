package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import "errors"

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

//type BatchNormD TensorD

//DeriveBNTensorDescriptor Derives a BN Tensor Descriptor from the one passed.
/*
* Derives a tensor descriptor from layer data descriptor for BatchNormalization
* scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
* bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
 */
func DeriveBNTensorDescriptor(xDesc *TensorD, mode BatchNormMode) (*TensorD, error) {
	if xDesc.dims > 5 || xDesc.dims < 4 {
		return nil, errors.New("dims for descriptor must be 4 or 5")
	}
	var descriptor C.cudnnTensorDescriptor_t
	err := Status(C.cudnnCreateTensorDescriptor(&descriptor)).error("DeriveBNTensorDescriptor-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnDeriveBNTensorDescriptor(descriptor, xDesc.descriptor, mode.c())).error("DeriveBNTensorDescriptor-Derive")
	if err != nil {
		return nil, err
	}
	return &TensorD{
		descriptor: descriptor,
		dims:       xDesc.dims,
	}, nil
}
