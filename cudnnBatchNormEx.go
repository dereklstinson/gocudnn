package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//BatchNormDEx is a gocudnn original.  This is to make the batchnorm operation similar to the majority cudnn.
type BatchNormDEx struct {
	mode C.cudnnBatchNormMode_t
	op   C.cudnnBatchNormOps_t
	set  bool
	gogc bool
}

//CreateBatchNormDescriptorEx creates a new BatchNormDEx
func CreateBatchNormDescriptorEx() *BatchNormDEx {
	return new(BatchNormDEx)
}

//Set sets the BatchNormMode and BatchNormOps held in the descriptor
func (b *BatchNormDEx) Set(mode BatchNormMode, op BatchNormOps) error {
	b.mode = mode.c()
	b.op = op.c()
	b.set = true
	b.gogc = true
	return nil
}

//Get gets the BatchNormMode and BatchNormOps held in the descriptor
func (b *BatchNormDEx) Get() (mode BatchNormMode, op BatchNormOps, err error) {
	if !b.set {
		return BatchNormMode(b.mode), BatchNormOps(b.op), errors.New("BatchNormD not set")
	}
	return BatchNormMode(b.mode), BatchNormOps(b.op), nil
}

//DeriveBNTensorDescriptor derives a tensor used for the batch norm operation
func (b *BatchNormDEx) DeriveBNTensorDescriptor(xDesc *TensorD) (bndesc *TensorD, err error) {
	if !b.set {
		return nil, errors.New("BatchNormD not set")
	}
	return cudnnDeriveBNTensorDescriptor(xDesc, BatchNormMode(b.mode), b.gogc)
}

//GetForwardTrainingWorkspaceSize gets the forward training ex workspacesize
func (b *BatchNormDEx) GetForwardTrainingWorkspaceSize(h *Handle,
	mode BatchNormMode,
	op BatchNormOps,
	xD, zD, yD, bnScaleBiasMeanVarDesc *TensorD,
	actD *ActivationD) (wspaceSIB uint, err error) {
	if !b.set {
		return 0, errors.New("BatchNormD not set")
	}
	var sib C.size_t
	err = Status(C.cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
		h.x,
		mode.c(),
		op.c(),
		xD.descriptor,
		zD.descriptor,
		yD.descriptor,
		bnScaleBiasMeanVarDesc.descriptor,
		actD.descriptor,
		&sib)).error("GetForwardTrainingWorkspaceSize")
	wspaceSIB = uint(sib)
	return wspaceSIB, err
}

//GeBackwardWorkspaceSize gets the workspace size in bytes for the backward operation
func (b *BatchNormDEx) GeBackwardWorkspaceSize(
	h *Handle,
	xD, yD, dyD, dzD, dxD, dbnScaleBiasMeanVarDesc *TensorD,
	actD *ActivationD,
) (wspaceSIB uint, err error) {
	if !b.set {
		return 0, errors.New("BatchNormD not set")
	}
	var sib C.size_t
	err = Status(C.cudnnGetBatchNormalizationBackwardExWorkspaceSize(h.x, b.mode, b.op, xD.descriptor, yD.descriptor, dyD.descriptor, dzD.descriptor, dxD.descriptor, dbnScaleBiasMeanVarDesc.descriptor, actD.descriptor, &sib)).error("GeBackwardWorkspaceSize")
	wspaceSIB = uint(sib)
	return wspaceSIB, err
}

//GetTrainingReserveSpaceSize gets the reserve space size for ex operation
func (b *BatchNormDEx) GetTrainingReserveSpaceSize(h *Handle,
	actD *ActivationD,
	xD *TensorD,
) (rspaceSIB uint, err error) {
	if !b.set {
		return 0, errors.New("BatchNormD not set")
	}
	var sib C.size_t
	err = Status(C.cudnnGetBatchNormalizationTrainingExReserveSpaceSize(h.x, b.mode, b.op, actD.descriptor, xD.descriptor, &sib)).error("GetTrainingReserveSpaceSize")
	rspaceSIB = uint(sib)
	return rspaceSIB, err
}

//ForwardTraining does the forward training ex algorithm.
func (b *BatchNormDEx) ForwardTraining(
	h *Handle,
	alpha, beta float64, //alpha -result blend factor, beta - dest blend factor
	xD *TensorD,
	x gocu.Mem,
	zD *TensorD,
	z gocu.Mem,
	yD *TensorD,
	y gocu.Mem, //output
	bnScaleBiasMeanVarDesc *TensorD,
	scale gocu.Mem,
	bias gocu.Mem,
	expoAverageFactor float64,
	resultRunningMean gocu.Mem,
	resultRunningVariance gocu.Mem,
	epsilon float64,
	resultSaveMean gocu.Mem, //optional can be null this and reslutSaveInVariance either both have to be null or not // output
	reslutSaveInVariance gocu.Mem, //optional can be null this and resultSaveMean either both have to be null or not. //output
	actD *ActivationD,
	wspace gocu.Mem,
	wspacesib uint,
	rspace gocu.Mem,
	rspacesib uint,
) error {
	if !b.set {
		return errors.New("BatchNormD not set")
	}
	a := cscalarbydatatype(yD.dtype, alpha)
	be := cscalarbydatatype(yD.dtype, beta)

	if resultSaveMean == nil || reslutSaveInVariance == nil {
		return Status(C.cudnnBatchNormalizationForwardTrainingEx(
			h.x,
			b.mode,
			b.op,
			a.CPtr(), be.CPtr(),
			xD.descriptor,
			x.Ptr(),
			zD.descriptor,
			z.Ptr(),
			yD.descriptor,
			y.Ptr(),
			bnScaleBiasMeanVarDesc.descriptor,
			scale.Ptr(),
			bias.Ptr(),
			C.double(expoAverageFactor),
			resultRunningMean.Ptr(),
			resultRunningVariance.Ptr(),
			C.double(epsilon),
			nil,
			nil,
			actD.descriptor,
			wspace.Ptr(),
			C.size_t(wspacesib),
			rspace.Ptr(),
			C.size_t(rspacesib),
		)).error("ForwardTrainingEx")
	}
	return Status(C.cudnnBatchNormalizationForwardTrainingEx(
		h.x,
		b.mode,
		b.op,
		a.CPtr(), be.CPtr(),
		xD.descriptor,
		x.Ptr(),
		zD.descriptor,
		z.Ptr(),
		yD.descriptor,
		y.Ptr(),
		bnScaleBiasMeanVarDesc.descriptor,
		scale.Ptr(),
		bias.Ptr(),
		C.double(expoAverageFactor),
		resultRunningMean.Ptr(),
		resultRunningVariance.Ptr(),
		C.double(epsilon),
		resultSaveMean.Ptr(),
		reslutSaveInVariance.Ptr(),
		actD.descriptor,
		wspace.Ptr(),
		C.size_t(wspacesib),
		rspace.Ptr(),
		C.size_t(rspacesib),
	)).error("ForwardTrainingEx")
}

//ForwardTrainingUS is loke ForwardTraining but using unsafe.Pointers instead of gocu.Mems
func (b *BatchNormDEx) ForwardTrainingUS(
	h *Handle,
	alpha, beta float64, //alpha -result blend factor, beta - dest blend factor
	xD *TensorD,
	x unsafe.Pointer,
	zD *TensorD,
	z unsafe.Pointer,
	yD *TensorD,
	y unsafe.Pointer, //output
	bnScaleBiasMeanVarDesc *TensorD,
	scale unsafe.Pointer,
	bias unsafe.Pointer,
	expoAverageFactor float64,
	resultRunningMean unsafe.Pointer,
	resultRunningVariance unsafe.Pointer,
	epsilon float64,
	resultSaveMean unsafe.Pointer, //optional can be null this and reslutSaveInVariance either both have to be null or not // output
	reslutSaveInVariance unsafe.Pointer, //optional can be null this and resultSaveMean either both have to be null or not. //output
	actD *ActivationD,
	wspace unsafe.Pointer,
	wspacesib uint,
	rspace unsafe.Pointer,
	rspacesib uint,
) error {
	if !b.set {
		return errors.New("BatchNormD not set")
	}
	a := cscalarbydatatype(yD.dtype, alpha)
	be := cscalarbydatatype(yD.dtype, beta)

	return Status(C.cudnnBatchNormalizationForwardTrainingEx(
		h.x,
		b.mode,
		b.op,
		a.CPtr(), be.CPtr(),
		xD.descriptor, x,
		zD.descriptor, z,
		yD.descriptor, y,
		bnScaleBiasMeanVarDesc.descriptor, scale, bias,
		C.double(expoAverageFactor),
		resultRunningMean,
		resultRunningVariance,
		C.double(epsilon),
		resultSaveMean,
		reslutSaveInVariance,
		actD.descriptor,
		wspace,
		C.size_t(wspacesib),
		rspace,
		C.size_t(rspacesib),
	)).error("ForwardTrainingEx")
}

//Backward does the backward ex algorithm.
func (b *BatchNormDEx) Backward(
	h *Handle,
	alphadata, betadata, alphaparam, betaparam float64, //alpha -result blend factor, beta - dest blend factor
	xD *TensorD,
	x gocu.Mem,
	yD *TensorD,
	y gocu.Mem,
	dyD *TensorD,
	dy gocu.Mem,
	dzD *TensorD,
	dz gocu.Mem,
	dxD *TensorD,
	dx gocu.Mem,
	dbnScaleBiasMeanVarDesc *TensorD,
	scale gocu.Mem,
	bias gocu.Mem,
	dscale gocu.Mem, //output - for training scale and bias
	dbias gocu.Mem, //output - for training scale and bias
	epsilon float64, //input - use the same as forward pass
	fromresultSaveMean gocu.Mem, //optional can be null this and reslutSaveInVariance either both have to be null or not // input
	fromreslutSaveInVariance gocu.Mem, //optional can be null this and resultSaveMean either both have to be null or not. //input
	actD *ActivationD,
	wspace gocu.Mem,
	wspacesib uint,
	rspace gocu.Mem,
	rspacesib uint,
) error {
	if !b.set {
		return errors.New("BatchNormDEx not set")
	}
	ad := cscalarbydatatype(yD.dtype, alphadata)
	bd := cscalarbydatatype(yD.dtype, betadata)
	ap := cscalarbydatatype(yD.dtype, alphaparam)
	bp := cscalarbydatatype(yD.dtype, betaparam)
	if fromreslutSaveInVariance == nil || fromresultSaveMean == nil {
		return Status(C.cudnnBatchNormalizationBackwardEx(
			h.x,
			b.mode,
			b.op,
			ad.CPtr(), bd.CPtr(), ap.CPtr(), bp.CPtr(),
			xD.descriptor,
			x.Ptr(),
			yD.descriptor,
			y.Ptr(),
			dyD.descriptor,
			dy.Ptr(),
			dzD.descriptor,
			dz.Ptr(),
			dxD.descriptor,
			dx.Ptr(),
			dbnScaleBiasMeanVarDesc.descriptor,
			scale.Ptr(),
			bias.Ptr(),
			dscale.Ptr(),
			dbias.Ptr(),
			C.double(epsilon),
			nil,
			nil,
			actD.descriptor,
			wspace.Ptr(),
			C.size_t(wspacesib),
			rspace.Ptr(),
			C.size_t(rspacesib),
		)).error("BackwardEx")
	}
	return Status(C.cudnnBatchNormalizationBackwardEx(
		h.x,
		b.mode,
		b.op,
		ad.CPtr(), bd.CPtr(), ap.CPtr(), bp.CPtr(),
		xD.descriptor,
		x.Ptr(),
		yD.descriptor,
		y.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		dzD.descriptor,
		dz.Ptr(),
		dxD.descriptor,
		dx.Ptr(),
		dbnScaleBiasMeanVarDesc.descriptor,
		scale.Ptr(),
		bias.Ptr(),
		dscale.Ptr(),
		dbias.Ptr(),
		C.double(epsilon),
		fromresultSaveMean.Ptr(),
		fromreslutSaveInVariance.Ptr(),
		actD.descriptor,
		wspace.Ptr(),
		C.size_t(wspacesib),
		rspace.Ptr(),
		C.size_t(rspacesib),
	)).error("BackwardEx")
}

//BackwardUS is just like Backward but uses unsafe.Pointers instead of gocu.Mem.
func (b *BatchNormDEx) BackwardUS(
	h *Handle,
	alphadata, betadata, alphaparam, betaparam float64, //alpha -result blend factor, beta - dest blend factor
	xD *TensorD,
	x unsafe.Pointer,
	yD *TensorD,
	y unsafe.Pointer,
	dyD *TensorD,
	dy unsafe.Pointer,
	dzD *TensorD,
	dz unsafe.Pointer,
	dxD *TensorD,
	dx unsafe.Pointer,
	dbnScaleBiasMeanVarDesc *TensorD,
	scale unsafe.Pointer,
	bias unsafe.Pointer,
	dscale unsafe.Pointer, //output - for training scale and bias
	dbias unsafe.Pointer, //output - for training scale and bias
	epsilon float64, //input - use the same as forward pass
	fromresultSaveMean unsafe.Pointer, //optional can be null this and reslutSaveInVariance either both have to be null or not // input
	fromreslutSaveInVariance unsafe.Pointer, //optional can be null this and resultSaveMean either both have to be null or not. //input
	actD *ActivationD,
	wspace unsafe.Pointer,
	wspacesib uint,
	rspace unsafe.Pointer,
	rspacesib uint,
) error {
	if !b.set {
		return errors.New("BatchNormDEx not set")
	}
	ad := cscalarbydatatype(yD.dtype, alphadata)
	bd := cscalarbydatatype(yD.dtype, betadata)
	ap := cscalarbydatatype(yD.dtype, alphaparam)
	bp := cscalarbydatatype(yD.dtype, betaparam)

	return Status(C.cudnnBatchNormalizationBackwardEx(
		h.x,
		b.mode,
		b.op,
		ad.CPtr(), bd.CPtr(), ap.CPtr(), bp.CPtr(),
		xD.descriptor, x,
		yD.descriptor, y,
		dyD.descriptor, dy,
		dzD.descriptor, dz,
		dxD.descriptor, dx,
		dbnScaleBiasMeanVarDesc.descriptor, scale, bias, dscale, dbias,
		C.double(epsilon),
		fromresultSaveMean,
		fromreslutSaveInVariance,
		actD.descriptor,
		wspace,
		C.size_t(wspacesib),
		rspace,
		C.size_t(rspacesib),
	)).error("BackwardEx")
}

/*ForwardInference info was pulled from cudnn documentation

This function performs the forward BatchNormalization layer computation for inference phase.
This layer is based on the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", S. Ioffe, C. Szegedy, 2015.
Note: Only 4D and 5D tensors are supported.
Note: The input transformation performed by this function is defined as: y := alpha*y + beta *(bnScale * (x-estimatedMean)/sqrt(epsilon + estimatedVariance)+bnBias)
Note: The epsilon value has to be the same during training, backpropagation and inference.
Note: For training phase use cudnnBatchNormalizationForwardTraining.
Note: Much higher performance when HW-packed tensors are used for all of x, dy, dx.

Parameters::

handle(input):  Handle to a previously created cuDNN library descriptor.

mode(input):    Mode of operation (spatial or per-activation). BatchNormMode

alpha, beta (input):  Pointers to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows:
					  dstValue = alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section for additional details.

xDesc, yDesc, x, y:   Tensor descriptors and pointers in device memory for the layer's x and y data.

bnScaleBiasMeanVarDesc, bnScaleData, bnBiasData(inputs):  Tensor descriptor and pointers in device memory for the batch normalization scale and bias parameters
														  (in the original paper bias is referred to as beta and scale as gamma).

estimatedMean, estimatedVariance (inputs):  Mean and variance tensors (these have the same descriptor as the bias and scale).
											It is suggested that resultRunningMean, resultRunningVariance from the cudnnBatchNormalizationForwardTraining
											call accumulated during the training phase are passed as inputs here.

epsilon(input):  Epsilon value used in the batch normalization formula. Minimum allowed value is CUDNN_BN_MIN_EPSILON defined in cudnn.h.

Possible error values returned by this function and their meanings are listed below.

Returns

CUDNN_STATUS_SUCCESS

    The computation was performed successfully.
CUDNN_STATUS_NOT_SUPPORTED

    The function does not support the provided configuration.
CUDNN_STATUS_BAD_PARAM

    At least one of the following conditions are met:

        One of the pointers alpha, beta, x, y, bnScaleData, bnBiasData, estimatedMean, estimatedInvVariance is NULL.
        Number of xDesc or yDesc tensor descriptor dimensions is not within the [4,5] range.
        bnScaleBiasMeanVarDesc dimensions are not 1xC(x1)x1x1 for spatial or 1xC(xD)xHxW for per-activation mode (parenthesis for 5D).
        epsilon value is less than CUDNN_BN_MIN_EPSILON
        Dimensions or data types mismatch for xDesc, yDesc





*/
/*
* Performs Batch Normalization during Inference:
* y[i] = bnScale[k]*(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + bnBias[k]
* with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
* according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
* above for notes on function arguments.
 */
func (b *BatchNormDEx) ForwardInference(
	handle *Handle,
	alpha, beta float64, /* alpha[0] = result blend factor, beta[0] = dest layer blend factor */
	xD *TensorD,
	x gocu.Mem, /* NxCxHxW */
	yD *TensorD,
	y gocu.Mem, /* NxCxHxW */
	ScaleBiasMeanVarDesc *TensorD,
	scale, bias, estimatedMean, estimatedVariance gocu.Mem, //all share the ScaleBiasMeanVarDesc descriptor
	epsilon float64,

) error {
	if !b.set {
		return errors.New("BatchNormDEx not set")
	}
	a := cscalarbydatatype(yD.dtype, alpha)
	be := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnBatchNormalizationForwardInference(
		handle.x,
		b.mode,
		a.CPtr(),
		be.CPtr(),
		xD.descriptor,
		x.Ptr(),
		yD.descriptor,
		y.Ptr(),
		ScaleBiasMeanVarDesc.descriptor,
		scale.Ptr(), bias.Ptr(), estimatedMean.Ptr(), estimatedVariance.Ptr(),
		C.double(epsilon),
	)).error("BatchNormalizationForwardInference")
}

//ForwardInferenceUS is just like ForwardInference but uses unsafe.Pointers instead of gocu.Mem
func (b *BatchNormDEx) ForwardInferenceUS(
	handle *Handle,
	alpha, beta float64, /* alpha[0] = result blend factor, beta[0] = dest layer blend factor */
	xD *TensorD,
	x unsafe.Pointer, /* NxCxHxW */
	yD *TensorD,
	y unsafe.Pointer, /* NxCxHxW */
	ScaleBiasMeanVarDesc *TensorD,
	scale, bias, estimatedMean, estimatedVariance unsafe.Pointer, //all share the ScaleBiasMeanVarDesc descriptor
	epsilon float64,

) error {
	if !b.set {
		return errors.New("BatchNormDEx not set")
	}
	a := cscalarbydatatype(yD.dtype, alpha)
	be := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnBatchNormalizationForwardInference(
		handle.x,
		b.mode,
		a.CPtr(),
		be.CPtr(),
		xD.descriptor, x,
		yD.descriptor, y,
		ScaleBiasMeanVarDesc.descriptor,
		scale, bias, estimatedMean, estimatedVariance,
		C.double(epsilon),
	)).error("BatchNormalizationForwardInference")
}

//MinEpsilon is the Minimum Epsilon required.  It is now zero, but it used to be 1e-5
func (b *BatchNormDEx) MinEpsilon() float64 {
	return float64(C.CUDNN_BN_MIN_EPSILON)
}
