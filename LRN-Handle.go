package gocudnn

/*
#include <cudnn.h>
*/
import "C"

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

//LRNCrossChannelForward  LRN cross-channel forward computation. Double parameters cast to tensor data type
func (handle *Handle) LRNCrossChannelForward(
	norm *LRND,
	mode LRNmode,
	alpha CScalar,
	xD *TensorD,
	x Memer,
	beta CScalar,
	yD *TensorD,
	y Memer,
) error {
	return Status(C.cudnnLRNCrossChannelForward(
		handle.x,
		norm.descriptor,
		mode.c(),
		alpha.CPtr(),
		xD.descriptor,
		x.Ptr(),
		beta.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("LRNCrossChannelForward")
}

//LRNCrossChannelBackward  LRN cross-channel backward computation. Double parameters cast to tensor data type
func (handle *Handle) LRNCrossChannelBackward(
	norm *LRND,
	mode LRNmode,
	alpha CScalar,
	yD *TensorD,
	y Memer,
	dyD *TensorD,
	dy Memer,
	xD *TensorD,
	x Memer,
	beta CScalar,
	dxD *TensorD,
	dx Memer,
) error {
	return Status(C.cudnnLRNCrossChannelBackward(
		handle.x,
		norm.descriptor,
		mode.c(),
		alpha.CPtr(),
		yD.descriptor,
		y.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		xD.descriptor,
		x.Ptr(),
		beta.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
	)).error("LRNCrossChannelForward")
}

//DivisiveNormalizationForward   LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y
func (handle *Handle) DivisiveNormalizationForward(
	norm LRND,
	mode DivNormMode,
	alpha CScalar,
	xD TensorD, /* same desc for means, temp, temp2 */
	x Memer,
	means Memer, /* if NULL, means are assumed to be zero */
	temp Memer,
	temp2 Memer,
	beta CScalar,
	yD TensorD,
	y Memer,
) error {
	return Status(C.cudnnDivisiveNormalizationForward(
		handle.x,
		norm.descriptor,
		mode.c(),
		alpha.CPtr(),
		xD.descriptor,
		x.Ptr(),
		means.Ptr(),
		temp.Ptr(),
		temp2.Ptr(),
		beta.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("DivisiveNormalizationForward")
}

//DivisiveNormalizationBackward  LRN cross-channel backward computation. Double parameters cast to tensor data type
func (handle *Handle) DivisiveNormalizationBackward(
	norm *LRND,
	mode DivNormMode,
	alpha CScalar,
	xD *TensorD, /* same desc for x, means, dy, temp, temp2 */
	x Memer,
	means Memer, /* if NULL, means are assumed to be zero */
	dy Memer,
	temp Memer,
	temp2 Memer,
	beta CScalar,
	dXdMeansDesc *TensorD, /* same desc for dx, dMeans */
	dx Memer, /* output x differential */
	dMeans Memer, /* output means differential, can be NULL */
) error {
	return Status(C.cudnnDivisiveNormalizationBackward(
		handle.x,
		norm.descriptor,
		mode.c(),
		alpha.CPtr(),
		xD.descriptor,
		x.Ptr(),
		means.Ptr(),
		dy.Ptr(),
		temp.Ptr(),
		temp2.Ptr(),
		beta.CPtr(),
		dXdMeansDesc.descriptor,
		dx.Ptr(),
		dMeans.Ptr(),
	)).error("DivisiveNormalizationBackward")
}
