package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//BatchNormalizationForwardTraining needs to be relooked at
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
func (handle *Handle) BatchNormalizationForwardTraining(
	mode BatchNormMode,
	alpha CScaler, /* alpha[0] = result blend factor */
	beta CScaler, /* beta[0] = dest layer blend factor */
	xD TensorD,
	x Memer,
	yD TensorD,
	y Memer,
	/* Shared desc for the next 6 tensors in the argument list.
	   Data type to be set as follows:
	   type = (typeOf(x) == double) ? double : float
	   Dimensions for this descriptor depend on normalization mode
	   - Spatial Normalization : tensors are expected to have dims 1xCx1x1
		(normalization is performed across NxHxW)
	   - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW
		(normalization is performed across N) */
	bnScaleBiasMeanVar TensorD,
	/* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation */
	bnscale Memer,
	bnBias Memer,
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
	resultrunningmean Memer,
	/* Output in training mode, input in inference. Is the moving average
	   of  variance[x] (factor is applied in the same way as for runningMean) */
	resultRunningVariance Memer,
	/* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
	epsilon float64,
	/* Optionally save intermediate results from the forward pass here
	- can be reused to speed up backward pass. NULL if unused */
	resultSaveMean Memer,
	resultSaveInvVariance Memer,

) error {
	return Status(C.cudnnBatchNormalizationForwardTraining(
		handle.x,
		mode.c(),
		alpha.CPtr(),
		beta.CPtr(),
		xD.descriptor,
		x.Ptr(),
		yD.descriptor,
		y.Ptr(),
		bnScaleBiasMeanVar.descriptor,
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

/*BatchNormalizationForwardInference info was pulled from cudnn documentation

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
func (handle *Handle) BatchNormalizationForwardInference(
	mode BatchNormMode,
	alpha CScaler, /* alpha[0] = result blend factor */
	beta CScaler, /* beta[0] = dest layer blend factor */
	xD TensorD,
	x Memer, /* NxCxHxW */
	yD TensorD,
	y Memer, /* NxCxHxW */
	bnScaleBiasMeanVarDesc TensorD,
	bnscale Memer,
	bnBias Memer,
	estimatedMean Memer, //same descriptor as bias and scale
	estimatedVariance Memer, //same descriptor as bias and scale
	epsilon float64,

) error {
	return Status(C.cudnnBatchNormalizationForwardInference(
		handle.x,
		mode.c(),
		alpha.CPtr(),
		beta.CPtr(),
		xD.descriptor,
		x.Ptr(),
		yD.descriptor,
		y.Ptr(),
		bnScaleBiasMeanVarDesc.descriptor,
		bnscale.Ptr(),
		bnBias.Ptr(),
		estimatedMean.Ptr(),     //from resultRunningMean
		estimatedVariance.Ptr(), //from  resultRunningVariance
		C.double(epsilon),
	)).error("BatchNormalizationForwardInference")
}

//BatchNormalizationBackward I don't know if this is set up correctly
/*
* Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
func (handle *Handle) BatchNormalizationBackward(
	mode BatchNormMode,
	alphaDataDiff CScaler,
	betaDataDiff CScaler,
	alphaParamDiff CScaler,
	betaParamDiff CScaler,
	xD TensorD, /* same desc for x, dx, dy */
	x Memer,
	dyD TensorD,
	dy Memer,
	dxD TensorD,
	dx Memer,
	/* Shared tensor desc for the 4 tensors below */
	dBnScaleBiasDesc TensorD,
	bnScale Memer, /* bnBias doesn't affect backpropagation */
	/* scale and bias diff are not backpropagated below this layer */
	dBnScaleResult Memer,
	dBnBiasResult Memer,
	/* Same epsilon as forward pass */
	epsilon float64,
	/* Optionally cached intermediate results from forward pass */
	savedMean Memer,
	savedInvVariance Memer,
) error {
	return Status(C.cudnnBatchNormalizationBackward(
		handle.x,
		mode.c(),
		alphaDataDiff.CPtr(),
		betaDataDiff.CPtr(),
		alphaParamDiff.CPtr(),
		betaParamDiff.CPtr(),
		xD.descriptor,
		x.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		dxD.descriptor,
		dx.Ptr(),
		dBnScaleBiasDesc.descriptor,
		bnScale.Ptr(),
		dBnScaleResult.Ptr(),
		dBnBiasResult.Ptr(),
		C.double(epsilon),
		savedMean.Ptr(),
		savedInvVariance.Ptr(),
	)).error("BatchNormalizationBackward")
}
