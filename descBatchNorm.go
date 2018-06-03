package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import "errors"

//BatchNormMode used for BatchNormMode Flags
type BatchNormMode C.cudnnBatchNormMode_t

//Flags for batchnormmode
const (

	/* bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice) */
	BatchNormPerActivation BatchNormMode = C.CUDNN_BATCHNORM_PER_ACTIVATION

	/* bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors) */
	BatchNormSpacial BatchNormMode = C.CUDNN_BATCHNORM_SPATIAL

	/*
	 * bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors).
	 * May be faster than CUDNN_BATCHNORM_SPATIAL but imposes some limits on the range of values
	 */
	BatchNormSpatialPersistent BatchNormMode = C.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
)

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
