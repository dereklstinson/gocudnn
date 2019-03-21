package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

/*
type BatchNormD struct{
	mode  C.cudnnBatchNormMode_t
}
type BatchNormExD struct{
	mode C.cudnnBatchNormMode_t
	ops C.cudnnBatchNormOps_t
}
*/

//BatchD is the Descriptor that holds the batchnorm descriptor
type BatchD struct {
	d *TensorD
}

//DeriveBNTensorDescriptor Derives a BN Tensor Descriptor from the one passed.
/*
* Derives a tensor descriptor from layer data descriptor for BatchNormalization
* scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
* bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
 */
func DeriveBNTensorDescriptor(xDesc *TensorD, mode BatchNormMode) (descriptor *TensorD, err error) {
	if xDesc.dims > 5 || xDesc.dims < 4 {
		return nil, errors.New("dims for descriptor must be 4 or 5")
	}
	if setfinalizer {
		descriptor, err = createtensordescriptor(true, true)
	} else {
		descriptor, err = createtensordescriptor(true, false)
	}

	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnDeriveBNTensorDescriptor(descriptor.descriptor, xDesc.descriptor, mode.c())).error("DeriveBNTensorDescriptor-Derive")
	if setfinalizer == true {
		runtime.SetFinalizer(descriptor, destroytensordescriptor)
	}
	if err != nil {
		return nil, err
	}

	descriptor.frmt, descriptor.dtype, descriptor.dimsarray, descriptor.stride, err = descriptor.GetDescrptor()

	if err != nil {
		return nil, err
	}
	return descriptor, nil
}

//GetForwardTrainingExWorkspaceSize gets the forward training ex workspacesize
func GetForwardTrainingExWorkspaceSize(h *Handle,
	mode BatchNormMode,
	op BatchNormOps,
	xD, zD, yD, bnScaleBiasMeanVarDesc *TensorD,
	actD *ActivationD) (uint, error) {

	var sib C.size_t
	err := Status(C.cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
		h.x,
		mode.c(),
		op.c(),
		xD.descriptor,
		zD.descriptor,
		yD.descriptor,
		bnScaleBiasMeanVarDesc.descriptor,
		actD.descriptor,
		&sib)).error("GetForwardTrainingExWorkspaceSize")
	return uint(sib), err
}

//GetBackwardExWorkspaceSize gets the workspace size in bytes for the backward operation
func GetBackwardExWorkspaceSize(
	h *Handle,
	mode BatchNormMode,
	op BatchNormOps,
	xD, yD, dyD, dzD, dxD, dbnScaleBiasMeanVarDesc *TensorD,
	actD *ActivationD,
) (uint, error) {
	var sib C.size_t
	err := Status(C.cudnnGetBatchNormalizationBackwardExWorkspaceSize(h.x, mode.c(), op.c(), xD.descriptor, yD.descriptor, dyD.descriptor, dzD.descriptor, dxD.descriptor, dbnScaleBiasMeanVarDesc.descriptor, actD.descriptor, &sib)).error("BatchNormWSsizeex")
	return uint(sib), err
}

//GetTrainingExReserveSpaceSize gets the reserve space size for ex operation
func GetTrainingExReserveSpaceSize(h *Handle,
	mode BatchNormMode,
	ops BatchNormOps,
	actD *ActivationD,
	xD *TensorD,
) (uint, error) {
	var sib C.size_t
	err := Status(C.cudnnGetBatchNormalizationTrainingExReserveSpaceSize(h.x, mode.c(), ops.c(), actD.descriptor, xD.descriptor, &sib)).error("GetTrainingExReserveSpaceSize")
	return uint(sib), err
}

type batchNormFuncs struct {
}

//ForwardTraining from the documentation
/*
This function performs the forward BatchNormalization layer computation for training phase.
Note: Only 4D and 5D tensors are supported.
Note: The epsilon value has to be the same during training, backpropagation and inference.
Note: For inference phase use cudnnBatchNormalizationForwardInference.
Note: Much higher performance for HW-packed tensors for both x and y.

Parameters(
handle:                  Handle to a previously created cuDNN library descriptor.

mode:                    Mode of operation (spatial or per-activation). cudnnBatchNormMode_t

alpha, beta (Inputs):    Pointers to scaling factors (in host memory) used to blend the layer output value
						 with prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue.
						 Please refer to programming model.

xD, yD, x, y:            Tensor descriptors and pointers in device memory for the layer's x and y data.

bnScaleBiasMeanVar:      Shared tensor descriptor desc for all the 6 tensors below in the argument list.
						 The dimensions for this tensor descriptor are dependent on the normalization mode.

bnScale, bnBias(Inputs): Pointers in device memory for the batch normalization scale and bias parameters (in original paper bias is referred to as beta and scale as gamma).
						 Note that bnBias parameter can replace the previous layer's bias parameter for improved efficiency.

expAveFactor (input):	 Factor used in the moving average computation runningMean = newMean*factor + runningMean*(1-factor).
						 Use a factor=1/(1+n) at N-th call to the function to get Cumulative Moving Average (CMA) behavior CMA[n] = (x[1]+...+x[n])/n.
						 Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1)= ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) = CMA[n]*(1-1/(n+1))+x[n+1]*1/(n+1)

resultRunningMean,resultRunningVariance (input/output):  Running mean and variance tensors (these have the same descriptor as the bias and scale).
														 Both of these pointers can be NULL but only at the same time.
														 The value stored in resultRunningVariance (or passed as an input in inference mode) is the moving average of variance[x]
														 where variance is computed either over batch or spatial+batch dimensions depending on the mode.
														 If these pointers are not NULL, the tensors should be initialized to some reasonable values or to 0.

epsilon:                 Epsilon value used in the batch normalization formula. Minimum allowed value is CUDNN_BN_MIN_EPSILON defined in cudnn.h.
						 Same epsilon value should be used in forward and backward functions.

resultSaveMean, resultSaveInvVariance (outputs):  Optional cache to save intermediate results computed during the forward pass
												 - these can then be reused to speed up the backward pass.
												 For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called.
												 Note that both of these parameters can be NULL but only at the same time.
												 It is recommended to use this cache since memory overhead is relatively small because these tensors have a much lower product of dimensions than the data tensors.


Possible error values returned by this function and their meanings are listed below.

Returns

CUDNN_STATUS_SUCCESS

    The computation was performed successfully.
CUDNN_STATUS_NOT_SUPPORTED

    The function does not support the provided configuration.
CUDNN_STATUS_BAD_PARAM

    At least one of the following conditions are met:

        One of the pointers alpha, beta, x, y, bnScaleData, bnBiasData is NULL.
        Number of xDesc or yDesc tensor descriptor dimensions is not within the [4,5] range.
        bnScaleBiasMeanVarDesc dimensions are not 1xC(x1)x1x1 for spatial or 1xC(xD)xHxW for per-activation mode (parens for 5D).
        Exactly one of resultSaveMean, resultSaveInvVariance pointers is NULL.
        Exactly one of resultRunningMean, resultRunningInvVariance pointers is NULL.
        epsilon value is less than CUDNN_BN_MIN_EPSILON
        Dimensions or data types mismatch for xDesc, yDesc


*/
/* Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
func (bnf batchNormFuncs) ForwardTraining(
	handle *Handle,
	mode BatchNormMode,
	alpha float64, /* alpha[0] = result blend factor */
	beta float64, /* beta[0] = dest layer blend factor */
	xD *TensorD,
	x gocu.Mem,
	yD *TensorD,
	y gocu.Mem,
	/* Shared desc for the next 6 tensors in the argument list.
	   Data type to be set as follows:
	   type = (typeOf(x) == double) ? double : float
	   Dimensions for this descriptor depend on normalization mode
	   - Spatial Normalization : tensors are expected to have dims 1xCx1x1
		(normalization is performed across NxHxW)
	   - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW
		(normalization is performed across N) */
	bnScaleBiasMeanVar *TensorD,
	/* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation */
	scale gocu.Mem,
	bias gocu.Mem,
	/* MUST use factor=1 in the very first call of a complete training cycle.
	        Use a factor=1/(1+n) at N-th call to the function to get
	        Cumulative Moving Average (CMA) behavior
	        CMA[n] = (x[1]+...+x[n])/n
	        Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
	        ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
			CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
	expAveFactor float64,
	/* Used in Training phase only.
	   runningMean = newMean*factor + runningMean*(1-factor) */
	resultrunningmean gocu.Mem, //output
	/* Output in training mode, input in inference. Is the moving average
	   of  variance[x] (factor is applied in the same way as for runningMean) */
	resultRunningVariance gocu.Mem, //output
	epsilon float64, /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
	resultSaveMean gocu.Mem, //output /* Optionally save intermediate results from the forward pass here	- can be reused to speed up backward pass. NULL if unused */
	resultSaveInvVariance gocu.Mem, //output /* Optionally save intermediate results from the forward pass here	- can be reused to speed up backward pass. NULL if unused */

) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if resultSaveInvVariance == nil || resultSaveMean == nil {
		return Status(C.cudnnBatchNormalizationForwardTraining(
			handle.x,
			mode.c(),
			a.CPtr(),
			b.CPtr(),
			xD.descriptor,
			x.Ptr(),
			yD.descriptor,
			y.Ptr(),
			bnScaleBiasMeanVar.descriptor,
			scale.Ptr(), bias.Ptr(),
			C.double(expAveFactor),
			resultrunningmean.Ptr(), resultRunningVariance.Ptr(),
			C.double(epsilon),
			nil, nil,
		)).error("BatchNormalizationForwardTraining")
	}
	return Status(C.cudnnBatchNormalizationForwardTraining(
		handle.x,
		mode.c(),
		a.CPtr(),
		b.CPtr(),
		xD.descriptor,
		x.Ptr(),
		yD.descriptor,
		y.Ptr(),
		bnScaleBiasMeanVar.descriptor,
		scale.Ptr(),
		bias.Ptr(),
		C.double(expAveFactor),
		resultrunningmean.Ptr(), resultRunningVariance.Ptr(),
		C.double(epsilon),
		resultSaveMean.Ptr(), resultSaveInvVariance.Ptr(),
	)).error("BatchNormalizationForwardTraining")
}

//ForwardTrainingEx does the forward training ex algorithm.
func (bnf batchNormFuncs) ForwardTrainingEx(
	h *Handle,
	mode BatchNormMode,
	ops BatchNormOps,
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
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)

	if resultSaveMean == nil || reslutSaveInVariance == nil {
		return Status(C.cudnnBatchNormalizationForwardTrainingEx(
			h.x,
			mode.c(),
			ops.c(),
			a.CPtr(), b.CPtr(),
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
		mode.c(),
		ops.c(),
		a.CPtr(), b.CPtr(),
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

//BackwardEx does the backward ex algorithm.
func (bnf batchNormFuncs) BackwardEx(
	h *Handle,
	mode BatchNormMode,
	ops BatchNormOps,
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
	ad := cscalarbydatatype(yD.dtype, alphadata)
	bd := cscalarbydatatype(yD.dtype, betadata)
	ap := cscalarbydatatype(yD.dtype, alphaparam)
	bp := cscalarbydatatype(yD.dtype, betaparam)
	if fromreslutSaveInVariance == nil || fromresultSaveMean == nil {
		return Status(C.cudnnBatchNormalizationBackwardEx(
			h.x,
			mode.c(),
			ops.c(),
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
		mode.c(),
		ops.c(),
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
func (bnf batchNormFuncs) ForwardInference(
	handle *Handle,
	mode BatchNormMode,
	alpha, beta float64, /* alpha[0] = result blend factor, beta[0] = dest layer blend factor */
	xD *TensorD,
	x gocu.Mem, /* NxCxHxW */
	yD *TensorD,
	y gocu.Mem, /* NxCxHxW */
	ScaleBiasMeanVarDesc *TensorD,
	scale, bias, estimatedMean, estimatedVariance gocu.Mem, //all share the ScaleBiasMeanVarDesc descriptor
	epsilon float64,

) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnBatchNormalizationForwardInference(
		handle.x,
		mode.c(),
		a.CPtr(),
		b.CPtr(),
		xD.descriptor,
		x.Ptr(),
		yD.descriptor,
		y.Ptr(),
		ScaleBiasMeanVarDesc.descriptor,
		scale.Ptr(), bias.Ptr(), estimatedMean.Ptr(), estimatedVariance.Ptr(),
		C.double(epsilon),
	)).error("BatchNormalizationForwardInference")
}

//Backward I don't know if this is set up correctly
/*
* Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
func (bnf batchNormFuncs) Backward(
	handle *Handle,
	mode BatchNormMode,
	alphadata, betadata, alphaparam, betaparam float64,
	xD *TensorD, /* same desc for x, dx, dy */
	x gocu.Mem,
	dyD *TensorD,
	dy gocu.Mem,
	dxD *TensorD,
	dx gocu.Mem,
	/* Shared tensor desc for the 4 tensors below */
	dBnScaleBiasDesc *TensorD,
	scale gocu.Mem, /* bias doesn't affect backpropagation */
	/* scale and bias diff are not backpropagated below this layer */
	dscale gocu.Mem, //output for training
	dbias gocu.Mem, //output for training
	/* Same epsilon as forward pass */
	epsilon float64,
	/* Optionally cached intermediate results from forward pass */
	savedMean gocu.Mem,
	savedInvVariance gocu.Mem,
) error {
	a := cscalarbydatatype(xD.dtype, alphadata)
	b := cscalarbydatatype(xD.dtype, betadata)
	ap := cscalarbydatatype(xD.dtype, alphaparam)
	bp := cscalarbydatatype(xD.dtype, betaparam)
	return Status(C.cudnnBatchNormalizationBackward(
		handle.x,
		mode.c(),
		a.CPtr(),
		b.CPtr(),
		ap.CPtr(),
		bp.CPtr(),
		xD.descriptor,
		x.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		dxD.descriptor,
		dx.Ptr(),
		dBnScaleBiasDesc.descriptor,
		scale.Ptr(),
		dscale.Ptr(),
		dbias.Ptr(),
		C.double(epsilon),
		savedMean.Ptr(),
		savedInvVariance.Ptr(),
	)).error("BatchNormalizationBackward")
}
func (bnf batchNormFuncs) MinEpsilon() float64 {
	return float64(C.CUDNN_BN_MIN_EPSILON)
}

/*








Version 2











*/

//DeriveBNTensorDescriptorV2 is a new one
func DeriveBNTensorDescriptorV2(xDesc *TensorD, mode BatchNormMode) (descriptor *BatchD, err error) {
	if xDesc.dims > 5 || xDesc.dims < 4 {
		return nil, errors.New("dims for descriptor must be 4 or 5")
	}
	var desc C.cudnnTensorDescriptor_t
	err = Status(C.cudnnCreateTensorDescriptor(&desc)).error("DeriveBNTensorDescriptor-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnDeriveBNTensorDescriptor(desc, xDesc.descriptor, mode.c())).error("DeriveBNTensorDescriptor-Derive")
	if err != nil {
		return nil, err
	}
	descriptor = &BatchD{
		d: &TensorD{
			descriptor: desc,
			dims:       xDesc.dims,
		},
	}

	if setfinalizer == true {
		runtime.SetFinalizer(descriptor.d, destroytensordescriptor)
	}
	return descriptor, nil
}

//BatchNormalizationForwardTrainingV2 is like regular but the method is for the shared descriptor
/* Shared desc for the next 6 tensors in the argument list.
   Data type to be set as follows:
   type = (typeOf(x) == double) ? double : float
   Dimensions for this descriptor depend on normalization mode
   - Spatial Normalization : tensors are expected to have dims 1xCx1x1
	(normalization is performed across NxHxW)
   - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW
	(normalization is performed across N) */
func (bnd *BatchD) BatchNormalizationForwardTrainingV2(
	handle *Handle,
	mode BatchNormMode,
	alpha float64, /* alpha[0] = result blend factor */
	beta float64, /* beta[0] = dest layer blend factor */
	xD *TensorD,
	x gocu.Mem,
	yD *TensorD,
	y gocu.Mem,

	/* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation */
	bnscale gocu.Mem,
	bnBias gocu.Mem,
	/* MUST use factor=1 in the very first call of a complete training cycle.
	        Use a factor=1/(1+n) at N-th call to the function to get
	        Cumulative Moving Average (CMA) behavior
	        CMA[n] = (x[1]+...+x[n])/n
	        Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
	        ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
			CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
	expAveFactor float64,

	/* Used in Training phase only.
	   runningMean = newMean*factor + runningMean*(1-factor) */
	resultrunningmean gocu.Mem,
	/* Output in training mode, input in inference. Is the moving average
	   of  variance[x] (factor is applied in the same way as for runningMean) */
	resultRunningVariance gocu.Mem,
	/* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
	epsilon float64,
	/* Optionally save intermediate results from the forward pass here
	- can be reused to speed up backward pass. NULL if unused */
	resultSaveMean gocu.Mem,
	resultSaveInvVariance gocu.Mem,

) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnBatchNormalizationForwardTraining(
		handle.x,
		mode.c(),
		a.CPtr(),
		b.CPtr(),
		xD.descriptor,
		x.Ptr(),
		yD.descriptor,
		y.Ptr(),
		bnd.d.descriptor,
		bnscale.Ptr(),
		bnBias.Ptr(),
		C.double(expAveFactor),
		resultrunningmean.Ptr(),
		resultRunningVariance.Ptr(),
		C.double(epsilon),
		resultSaveMean.Ptr(),
		resultSaveInvVariance.Ptr(),
	)).error("BatchNormalizationForwardTraining")
}

//BatchNormalizationForwardInferenceV2 is like the original but  With the batch norm descriptor moved
func (bnd *BatchD) BatchNormalizationForwardInferenceV2(
	handle *Handle,
	mode BatchNormMode,
	alpha float64, /* alpha[0] = result blend factor */
	beta float64, /* beta[0] = dest layer blend factor */
	xD *TensorD,
	x gocu.Mem, /* NxCxHxW */
	yD *TensorD,
	y gocu.Mem, /* NxCxHxW */

	bnscale gocu.Mem,
	bnBias gocu.Mem,
	estimatedMean gocu.Mem, //same descriptor as bias and scale
	estimatedVariance gocu.Mem, //same descriptor as bias and scale
	epsilon float64,

) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnBatchNormalizationForwardInference(
		handle.x,
		mode.c(),
		a.CPtr(),
		b.CPtr(),
		xD.descriptor,
		x.Ptr(),
		yD.descriptor,
		y.Ptr(),
		bnd.d.descriptor,
		bnscale.Ptr(),
		bnBias.Ptr(),
		estimatedMean.Ptr(),     //from resultRunningMean
		estimatedVariance.Ptr(), //from  resultRunningVariance
		C.double(epsilon),
	)).error("BatchNormalizationForwardInference")
}

//BatchNormalizationBackwardV2 I don't know if this is set up correctly
/*
* Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
func (bnd *BatchD) BatchNormalizationBackwardV2(
	handle *Handle,
	mode BatchNormMode,
	alphaDataDiff float64,
	betaDataDiff float64,
	alphaParamDiff float64,
	betaParamDiff float64,
	xD *TensorD, /* same desc for x, dx, dy */
	x gocu.Mem,
	dyD *TensorD,
	dy gocu.Mem,
	dxD *TensorD,
	dx gocu.Mem,
	/* Shared tensor desc for the 4 tensors below */
	bnScale gocu.Mem, /* bnBias doesn't affect backpropagation */
	/* scale and bias diff are not backpropagated below this layer */
	dBnScaleResult gocu.Mem,
	dBnBiasResult gocu.Mem,
	/* Same epsilon as forward pass */
	epsilon float64,
	/* Optionally cached intermediate results from forward pass */
	savedMean gocu.Mem,
	savedInvVariance gocu.Mem,
) error {
	a := cscalarbydatatype(xD.dtype, alphaDataDiff)
	b := cscalarbydatatype(xD.dtype, betaDataDiff)
	ap := cscalarbydatatype(xD.dtype, alphaParamDiff)
	bp := cscalarbydatatype(xD.dtype, betaParamDiff)
	return Status(C.cudnnBatchNormalizationBackward(
		handle.x,
		mode.c(),
		a.CPtr(),
		b.CPtr(),
		ap.CPtr(),
		bp.CPtr(),
		xD.descriptor,
		x.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		dxD.descriptor,
		dx.Ptr(),
		bnd.d.descriptor,
		bnScale.Ptr(),
		dBnScaleResult.Ptr(),
		dBnBiasResult.Ptr(),
		C.double(epsilon),
		savedMean.Ptr(),
		savedInvVariance.Ptr(),
	)).error("BatchNormalizationBackward")
}

/*






FLAGS








*/

//BatchNormOps are flags for BatchNormOps when needed
type BatchNormOps C.cudnnBatchNormOps_t

func (b BatchNormOps) c() C.cudnnBatchNormOps_t {
	return C.cudnnBatchNormOps_t(b)
}

//Normal sets b to  BatchNormOps(C.CUDNN_BATCHNORM_OPS_BN) and returns that new value /* do batch normalization only */
func (b *BatchNormOps) Normal() BatchNormOps { *b = BatchNormOps(C.CUDNN_BATCHNORM_OPS_BN); return *b }

//Activation sets b to BatchNormOps(C.CUDNN_BATCHNORM_OPS_BN_ACTIVATION) /* do batchNorm, then activation */
func (b *BatchNormOps) Activation() BatchNormOps {
	*b = BatchNormOps(C.CUDNN_BATCHNORM_OPS_BN_ACTIVATION)
	return *b
}

//AddActivation sets b to BatchNormOps(C.CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) /* do batchNorm, then elemWiseAdd, then activation */
func (b *BatchNormOps) AddActivation() BatchNormOps {
	*b = BatchNormOps(C.CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION)
	return *b
}

//BatchNormMode used for BatchNormMode Flags
type BatchNormMode C.cudnnBatchNormMode_t

//PerActivation sets b to BatchNormMode(C.CUDNN_BATCHNORM_PER_ACTIVATION) and returns that new value
func (b *BatchNormMode) PerActivation() BatchNormMode {
	*b = BatchNormMode(C.CUDNN_BATCHNORM_PER_ACTIVATION)
	return *b
}

//Spatial sets b to BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL) and returns that new value
func (b *BatchNormMode) Spatial() BatchNormMode {
	*b = BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL)
	return *b
}

// SpatialPersistent sets b to BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL_PERSISTENT) and returns that new value
func (b *BatchNormMode) SpatialPersistent() BatchNormMode {
	*b = BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL_PERSISTENT)
	return *b
}
func (b BatchNormMode) c() C.cudnnBatchNormMode_t { return C.cudnnBatchNormMode_t(b) }

const bnMinEpsilon = float64(1e-5)
