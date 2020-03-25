package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//BatchNormD is a gocudnn original.  This is to make the batchnorm operation similar to the majority cudnn.
type BatchNormD struct {
	mode C.cudnnBatchNormMode_t
	set  bool
	gogc bool
}

//CreateBatchNormDescriptor creates a new BatchNormD
func CreateBatchNormDescriptor() *BatchNormD {
	return new(BatchNormD)
}

//Set sets the values used in the batchnorm descriptor
func (b *BatchNormD) Set(mode BatchNormMode) error {
	b.mode = mode.c()
	b.set = true
	b.gogc = true
	return nil
}

//Get gets the values stored in BatchNormMode
func (b *BatchNormD) Get() (mode BatchNormMode, err error) {
	if !b.set {
		return 0, errors.New("BatchNormD not set")
	}
	return BatchNormMode(b.mode), nil
}
func (b *BatchNormD) String() string {
	return fmt.Sprintf("BatchNormD{\n%v,\n}\n", BatchNormMode(b.mode))

}

//DeriveBNTensorDescriptor Derives a BN Tensor Descriptor from the one passed.
/*
* Derives a tensor descriptor from layer data descriptor for BatchNormalization
* scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
* bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
 */
func (b *BatchNormD) DeriveBNTensorDescriptor(xDesc *TensorD) (bndesc *TensorD, err error) {
	if !b.set {
		return nil, errors.New("BatchNormD not set")
	}
	return cudnnDeriveBNTensorDescriptor(xDesc, BatchNormMode(b.mode), b.gogc)
}

//ForwardInference info was pulled from cudnn documentation
//This function performs the forward BatchNormalization layer computation for inference phase.
//This layer is based on the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", S. Ioffe, C. Szegedy, 2015.
//
//Notes:
//
//1)Only 4D and 5D tensors are supported.
//
//2)The input transformation performed by this function is defined as: y := alpha*y + beta *(bnScale * (x-estimatedMean)/sqrt(epsilon + estimatedVariance)+bnBias)
//
//3)The epsilon value has to be the same during training, backpropagation and inference.
//
//4)For training phase use cudnnBatchNormalizationForwardTraining.
//
//5)Much higher performance when HW-packed tensors are used for all of x, dy, dx.
//
//Parameters:
//
//	----
//	handle(input):
//
//	Handle to a previously created cuDNN library descriptor.
//	----
//	---
//	mode(input):
//
//	Mode of operation (spatial or per-activation). BatchNormMode
//	---
//	----
//	alpha, beta (input):
//
//	Scaling factors in host mem y = alpha *result + beta *y
//	----
//	---
//	xDesc (input), yDesc (input), x (input), y (output):
//
//   Descriptors and pointers to mem
//	---
//	----
//	bnScaleBiasMeanVarDesc, bnScaleData, bnBiasData(inputs):
//
//	Tensor descriptor and pointers in device memory for
//	the batch normalization scale and bias parameters
//	----
//	---
//	estimatedMean, estimatedVariance (inputs):
//
//	Mean and variance tensors (these have the same descriptor as the bias and scale).
//	It is suggested that resultRunningMean, resultRunningVariance from the cudnnBatchNormalizationForwardTraining
//	call accumulated during the training phase are passed as inputs here.
//	---
//	----
//	epsilon(input):
//
//	Epsilon value used in the batch normalization formula.
//	Minimum allowed value is found in  MinEpsilon() method. (It is now zero)
//	----
//
//Returns:
//
//	nil - The computation was performed successfully.
//
//	CUDNN_STATUS_NOT_SUPPORTED - The function does not support the provided configuration.
//
//	CUDNN_STATUS_BAD_PARAM - At least one of the following conditions are met:
//
//		1)One of the pointers alpha, beta, x, y, bnScaleData, bnBiasData, estimatedMean, estimatedInvVariance is NULL.
//		2)Number of xDesc or yDesc tensor descriptor dimensions is not within the [4,5] range.
//		3)bnScaleBiasMeanVarDesc dimensions are not 1xC(x1)x1x1 for spatial or 1xC(xD)xHxW for per-activation mode (parenthesis for 5D).
//		4)epsilon value is less than CUDNN_BN_MIN_EPSILON
//		5)Dimensions or data types mismatch for xDesc, yDesc
//
func (b *BatchNormD) ForwardInference(
	handle *Handle,
	alpha, beta float64, /* alpha[0] = result blend factor, beta[0] = dest layer blend factor */
	xD *TensorD, x cutil.Mem, /* NxCxHxW */
	yD *TensorD, y cutil.Mem, /* NxCxHxW */
	ScaleBiasMeanVarDesc *TensorD, scale, bias, estimatedMean, estimatedVariance cutil.Mem, //all share the ScaleBiasMeanVarDesc descriptor
	epsilon float64,

) error {
	if !b.set {
		return errors.New("(b *BatchNormD) ForwardInference: BatchNormD not set")
	}
	a := cscalarbydatatype(yD.dtype, alpha)
	be := cscalarbydatatype(yD.dtype, beta)

	if handle.w != nil {
		return handle.w.Work(func() error {
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
			)).error("(b *BatchNormD) ForwardInference")
		})
	}
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
	)).error("(b *BatchNormD) ForwardInference")
}

//ForwardInferenceUS is like ForwardInference but uses unsafe.Pointers instead of cutil.Mems
func (b *BatchNormD) ForwardInferenceUS(
	handle *Handle,
	alpha, beta float64, /* alpha[0] = result blend factor, beta[0] = dest layer blend factor */
	xD *TensorD, x unsafe.Pointer, /* NxCxHxW */
	yD *TensorD, y unsafe.Pointer, /* NxCxHxW */
	ScaleBiasMeanVarDesc *TensorD, scale, bias, estimatedMean, estimatedVariance unsafe.Pointer, //all share the ScaleBiasMeanVarDesc descriptor
	epsilon float64,

) error {
	if !b.set {
		return errors.New("BatchNormD not set")
	}
	a := cscalarbydatatype(yD.dtype, alpha)
	be := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
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
			)).error("(b *BatchNormD) ForwardInferenceUS")
		})
	}
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
	)).error("(b *BatchNormD) ForwardInferenceUS")
}

//Backward - Performs backward pass of Batch Normalization layer.
//
//  Outputs: dx (backprop data), dscale (training scale), dbias (training bias)
//
// 	Scalars: alphadata, betadata, alphaparam, betaparam - are smoothing factors. y = alpha * operation + beta * y
//
//
//	Note: savedMean, savedInvVariance - These are cached results if used by the layer in the forward pass.
//					    These can be null iff they are both null.
func (b *BatchNormD) Backward(
	handle *Handle,
	alphadata, betadata, alphaparam, betaparam float64,
	xD *TensorD, x cutil.Mem, /* same desc for x, dx, dy */
	dyD *TensorD, dy cutil.Mem,
	dxD *TensorD, dx cutil.Mem,
	dBnScaleBiasDesc *TensorD, scale, dscale, dbias cutil.Mem, /* Shared tensor desc for the 4 tensors below */
	epsilon float64, /* Same epsilon as forward pass */
	/* Optionally cached intermediate results from forward pass */
	savedMean, savedInvVariance cutil.Mem,
) error {
	if !b.set {
		return errors.New("BatchNormD not set")
	}
	a := cscalarbydatatype(xD.dtype, alphadata)
	be := cscalarbydatatype(xD.dtype, betadata)
	ap := cscalarbydatatype(xD.dtype, alphaparam)
	bp := cscalarbydatatype(xD.dtype, betaparam)
	var smptr unsafe.Pointer
	var sinvptr unsafe.Pointer
	if savedMean != nil {
		smptr = savedMean.Ptr()
	} else {
		smptr = nil
	}
	if savedInvVariance != nil {
		sinvptr = savedInvVariance.Ptr()
	} else {
		sinvptr = nil
	}
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnBatchNormalizationBackward(
				handle.x,
				b.mode,
				a.CPtr(), be.CPtr(), ap.CPtr(), bp.CPtr(),
				xD.descriptor, x.Ptr(),
				dyD.descriptor, dy.Ptr(),
				dxD.descriptor, dx.Ptr(),
				dBnScaleBiasDesc.descriptor,
				scale.Ptr(),
				dscale.Ptr(),
				dbias.Ptr(),
				C.double(epsilon),
				smptr,
				sinvptr,
			)).error("(b *BatchNormD) Backward")
		})
	}
	return Status(C.cudnnBatchNormalizationBackward(
		handle.x,
		b.mode,
		a.CPtr(), be.CPtr(), ap.CPtr(), bp.CPtr(),
		xD.descriptor, x.Ptr(),
		dyD.descriptor, dy.Ptr(),
		dxD.descriptor, dx.Ptr(),
		dBnScaleBiasDesc.descriptor,
		scale.Ptr(),
		dscale.Ptr(),
		dbias.Ptr(),
		C.double(epsilon),
		smptr,
		sinvptr,
	)).error("(b *BatchNormD) Backward")
}

//BackwardUS is like Backward but uses unsafe.Pointers instead of cutil.Mem
func (b *BatchNormD) BackwardUS(
	handle *Handle,
	alphadata, betadata, alphaparam, betaparam float64,
	xD *TensorD, x unsafe.Pointer, /* same desc for x, dx, dy */
	dyD *TensorD, dy unsafe.Pointer,
	dxD *TensorD, dx unsafe.Pointer,
	dBnScaleBiasDesc *TensorD, scale, dscale, dbias unsafe.Pointer, /* Shared tensor desc for the 4 tensors below */
	epsilon float64, /* Same epsilon as forward pass */
	/* Optionally cached intermediate results from forward pass */
	savedMean, savedInvVariance unsafe.Pointer,
) error {
	if !b.set {
		return errors.New("BatchNormD not set")
	}
	a := cscalarbydatatype(xD.dtype, alphadata)
	be := cscalarbydatatype(xD.dtype, betadata)
	ap := cscalarbydatatype(xD.dtype, alphaparam)
	bp := cscalarbydatatype(xD.dtype, betaparam)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnBatchNormalizationBackward(
				handle.x,
				b.mode,
				a.CPtr(), be.CPtr(), ap.CPtr(), bp.CPtr(),
				xD.descriptor, x,
				dyD.descriptor, dy,
				dxD.descriptor, dx,
				dBnScaleBiasDesc.descriptor,
				scale,
				dscale,
				dbias,
				C.double(epsilon),
				savedMean,
				savedInvVariance,
			)).error("(b *BatchNormD) BackwardUS")
		})
	}
	return Status(C.cudnnBatchNormalizationBackward(
		handle.x,
		b.mode,
		a.CPtr(), be.CPtr(), ap.CPtr(), bp.CPtr(),
		xD.descriptor, x,
		dyD.descriptor, dy,
		dxD.descriptor, dx,
		dBnScaleBiasDesc.descriptor,
		scale,
		dscale,
		dbias,
		C.double(epsilon),
		savedMean,
		savedInvVariance,
	)).error("(b *BatchNormD) BackwardUS")
}

//MinEpsilon is the Minimum Epsilon required.  It is now zero, but it used to be 1e-5
func (b *BatchNormD) MinEpsilon() float64 {
	return float64(C.CUDNN_BN_MIN_EPSILON)
}

//ForwardTraining from the documentation
//This function performs the forward BatchNormalization layer computation for training phase.
//
//Notes:
//
//1)Only 4D and 5D tensors are supported.
//
//2)The epsilon value has to be the same during training, backpropagation and inference.
//
//3)For inference phase use cudnnBatchNormalizationForwardInference.
//
//4)Much higher performance for HW-packed tensors for both x and y.
//
//Parameters:
//	----
//	handle:
//
//	Handle to a previously created cuDNN library descriptor.
//	----
//	---
//	alpha, beta (Inputs):
//
//	Scaling Factors y= alpha*opresult + beta*y
//	---
//	----
//	xD, yD, x, y:
//
//	Tensor descriptors and pointers in device memory for the layer's x and y data.
//	----
//	---
//	bnScaleBiasMeanVar:
//
//	Shared tensor descriptor desc for all the 6 tensors below in the argument list.
//	The dimensions for this tensor descriptor are dependent on the normalization mode.
//	---
//	----
//	scal, bias(Inputs):
//
//	Pointers in device memory for the batch normalization scale and bias parameters.
//	Note: Since bias isn't used during the backward pass.  You can use bias for other batchnorm layers.
//	----
//	---
//	expAveFactor (input):
//
//	Factor used in the moving average computation runningMean = newMean*factor + runningMean*(1-factor).
//	Use a factor=1/(1+n) at N-th call to the function to get Cumulative Moving Average (CMA) behavior CMA[n] = (x[1]+...+x[n])/n.
//	Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1)= ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) = CMA[n]*(1-1/(n+1))+x[n+1]*1/(n+1)
//	---
//	----
//	resultRunningMean,resultRunningVariance (input/output):
//
//	Running mean and variance tensors (these have the same descriptor as the bias and scale).
//	Both of these pointers can be NULL but only at the same time.
//	The value stored in resultRunningVariance (or passed as an input in inference mode) is the moving average of variance[x]
//	where variance is computed either over batch or spatial+batch dimensions depending on the mode.
//	If these pointers are not NULL, the tensors should be initialized to some reasonable values or to 0.
//	----
//	---
//	epsilon:
//
//	Epsilon value used in the batch normalization formula. Minimum allowed value is CUDNN_BN_MIN_EPSILON defined in cudnn.h.
//	Same epsilon value should be used in forward and backward functions.
//	---
//	----
//	resultSaveMean, resultSaveInvVariance (outputs):
//
//	Optional cache to save intermediate results computed during the forward pass
//	these can then be reused to speed up the backward pass.
//	For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called.
//	Note that both of these parameters can be NULL but only at the same time.
//	It is recommended to use this cache since memory overhead is relatively small because these tensors have a much lower product of dimensions than the data tensors.
//	----
//
//Returns:
//
//	nil - The computation was performed successfully.
//
//
//	CUDNN_STATUS_NOT_SUPPORTED - The function does not support the provided configuration.
//
//
//	CUDNN_STATUS_BAD_PARAM - At least one of the following conditions are met:
//
//		1)One of the pointers alpha, beta, x, y, bnScaleData, bnBiasData is NULL.
//		2)Number of xDesc or yDesc tensor descriptor dimensions is not within the [4,5] range.
//		3)bnScaleBiasMeanVarDesc dimensions are not 1xC(x1)x1x1 for spatial or 1xC(xD)xHxW for per-activation mode (parens for 5D).
//		4)Exactly one of resultSaveMean, resultSaveInvVariance pointers is NULL.
//		5)Exactly one of resultRunningMean, resultRunningInvVariance pointers is NULL.
//		6)epsilon value is less than MinEpsilon()
//		7)Dimensions or data types mismatch for xDesc, yDesc
//
func (b *BatchNormD) ForwardTraining(
	handle *Handle,
	alpha float64, /* alpha[0] = result blend factor */
	beta float64, /* beta[0] = dest layer blend factor */
	xD *TensorD,
	x cutil.Mem,
	yD *TensorD,
	y cutil.Mem,
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
	scale cutil.Mem,
	bias cutil.Mem,
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
	resultrunningmean cutil.Mem, //output
	/* Output in training mode, input in inference. Is the moving average
	   of  variance[x] (factor is applied in the same way as for runningMean) */
	resultRunningVariance cutil.Mem, //output
	epsilon float64, /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
	resultSaveMean cutil.Mem, //output /* Optionally save intermediate results from the forward pass here	- can be reused to speed up backward pass. NULL if unused */
	resultSaveInvVariance cutil.Mem, //output /* Optionally save intermediate results from the forward pass here	- can be reused to speed up backward pass. NULL if unused */

) error {
	if !b.set {
		return errors.New("BatchNormD not set")
	}
	a := cscalarbydatatype(yD.dtype, alpha)
	be := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			if resultSaveInvVariance == nil || resultSaveMean == nil {
				return Status(C.cudnnBatchNormalizationForwardTraining(
					handle.x,
					b.mode,
					a.CPtr(),
					be.CPtr(),
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
				)).error("(b *BatchNormD) ForwardTraining")
			}
			return Status(C.cudnnBatchNormalizationForwardTraining(
				handle.x,
				b.mode,
				a.CPtr(),
				be.CPtr(),
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
			)).error("(b *BatchNormD) ForwardTraining")
		})
	}
	if resultSaveInvVariance == nil || resultSaveMean == nil {
		return Status(C.cudnnBatchNormalizationForwardTraining(
			handle.x,
			b.mode,
			a.CPtr(),
			be.CPtr(),
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
		)).error("(b *BatchNormD) ForwardTraining")
	}
	return Status(C.cudnnBatchNormalizationForwardTraining(
		handle.x,
		b.mode,
		a.CPtr(),
		be.CPtr(),
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
	)).error("(b *BatchNormD) ForwardTraining")
}

//ForwardTrainingUS is just like ForwardTraining but uses unsafe.Pointers.
func (b *BatchNormD) ForwardTrainingUS(
	handle *Handle,
	alpha float64, /* alpha[0] = result blend factor */
	beta float64, /* beta[0] = dest layer blend factor */
	xD *TensorD,
	x unsafe.Pointer,
	yD *TensorD,
	y unsafe.Pointer,
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
	scale unsafe.Pointer,
	bias unsafe.Pointer,
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
	resultrunningmean unsafe.Pointer, //output
	/* Output in training mode, input in inference. Is the moving average
	   of  variance[x] (factor is applied in the same way as for runningMean) */
	resultRunningVariance unsafe.Pointer, //output
	epsilon float64, /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
	resultSaveMean unsafe.Pointer, //output /* Optionally save intermediate results from the forward pass here	- can be reused to speed up backward pass. NULL if unused */
	resultSaveInvVariance unsafe.Pointer, //output /* Optionally save intermediate results from the forward pass here	- can be reused to speed up backward pass. NULL if unused */

) error {
	if !b.set {
		return errors.New("BatchNormD not set")
	}
	a := cscalarbydatatype(yD.dtype, alpha)
	be := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnBatchNormalizationForwardTraining(
				handle.x,
				b.mode,
				a.CPtr(),
				be.CPtr(),
				xD.descriptor, x,
				yD.descriptor, y,
				bnScaleBiasMeanVar.descriptor,
				scale,
				bias,
				C.double(expAveFactor),
				resultrunningmean, resultRunningVariance,
				C.double(epsilon),
				resultSaveMean, resultSaveInvVariance,
			)).error("b *BatchNormD) ForwardTrainingUS")
		})
	}
	return Status(C.cudnnBatchNormalizationForwardTraining(
		handle.x,
		b.mode,
		a.CPtr(),
		be.CPtr(),
		xD.descriptor, x,
		yD.descriptor, y,
		bnScaleBiasMeanVar.descriptor,
		scale,
		bias,
		C.double(expAveFactor),
		resultrunningmean, resultRunningVariance,
		C.double(epsilon),
		resultSaveMean, resultSaveInvVariance,
	)).error("b *BatchNormD) ForwardTrainingUS")
}

/*








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
func (b BatchNormOps) String() string {
	var x string
	bflg := b
	switch b {
	case bflg.Normal():
		x = "Normal"
	case bflg.Activation():
		x = "Activation"
	case bflg.AddActivation():
		x = "AddActivation"
	default:
		x = "Unsupported Flag"
	}
	return "BatchNormOps: " + x
}

//BatchNormMode used for BatchNormMode Flags
type BatchNormMode C.cudnnBatchNormMode_t

//PerActivation sets b to BatchNormMode(C.CUDNN_BATCHNORM_PER_ACTIVATION) and returns that new value
//Normalization is performed per-activation. This mode is intended to be used after the non-convolutional network layers.
//In this mode, the tensor dimensions of bnBias and bnScale and the parameters used in the cudnnBatchNormalization* functions, are 1xCxHxW.
func (b *BatchNormMode) PerActivation() BatchNormMode {
	*b = BatchNormMode(C.CUDNN_BATCHNORM_PER_ACTIVATION)
	return *b
}

//Spatial sets b to BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL) and returns that new value.
//Normalization is performed over N+spatial dimensions.
//This mode is intended for use after convolutional layers (where spatial invariance is desired).
//In this mode the bnBias and bnScale tensor dimensions are 1xCx1x1.
func (b *BatchNormMode) Spatial() BatchNormMode {
	*b = BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL)
	return *b
}

// SpatialPersistent sets b to BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL_PERSISTENT) and returns that new value
//This mode is similar to CUDNN_BATCHNORM_SPATIAL but it can be faster for some tasks.
func (b *BatchNormMode) SpatialPersistent() BatchNormMode {
	*b = BatchNormMode(C.CUDNN_BATCHNORM_SPATIAL_PERSISTENT)
	return *b
}
func (b BatchNormMode) String() string {
	var x string
	bflg := b

	switch b {
	case bflg.PerActivation():
		x = "PerActivation"
	case bflg.Spatial():
		x = "Spatial"
	case bflg.SpatialPersistent():
		x = "SpatialPersistent"
	default:
		x = "Unsupported Flag"
	}
	return "BatchNormMode: " + x
}
func (b BatchNormMode) c() C.cudnnBatchNormMode_t { return C.cudnnBatchNormMode_t(b) }

//Private Func
func cudnnDeriveBNTensorDescriptor(xDesc *TensorD, mode BatchNormMode, gogc bool) (descriptor *TensorD, err error) {
	if xDesc.dims > 5 || xDesc.dims < 4 {
		return nil, errors.New("dims for descriptor must be 4 or 5")
	}
	if setfinalizer || gogc {
		descriptor, err = createtensordescriptor(true, true)
	} else {
		descriptor, err = createtensordescriptor(true, false)
	}

	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnDeriveBNTensorDescriptor(descriptor.descriptor, xDesc.descriptor, mode.c())).error("DeriveBNTensorDescriptor-Derive")
	frmt, dtype, shape, stride, err := descriptor.Get()
	if err != nil {
		return nil, err
	}

	descriptor.frmt = frmt
	descriptor.dtype = dtype
	descriptor.shape = shape
	descriptor.stride = stride

	return descriptor, nil
}
