package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//SpatialTfGridGeneratorForward This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
func (handle *Handle) SpatialTfGridGeneratorForward(
	st SpatialTransformerD,
	theta Memer, //Input. Affine transformation matrix. It should be of size n*2*3 for a 2d transformation, n is the number of images.
	grid Memer, /*Output. A grid of coordinates. It is of size n*h*w*2 for a 2d transformation, where n,
	h, w is specified in stDesc . In the 4th dimension, the first coordinate is x, and the
	second coordinate is y*/

) error {
	return Status(C.cudnnSpatialTfGridGeneratorForward(
		handle.x,
		st.descriptor,
		theta.Ptr(),
		grid.Ptr(),
	)).error("SpatialTfGridGeneratorForward")
}

//SpatialTfGridGeneratorBackward - This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
func (handle *Handle) SpatialTfGridGeneratorBackward(
	st SpatialTransformerD,
	grid Memer,
	theta Memer,

) error {
	return Status(C.cudnnSpatialTfGridGeneratorBackward(
		handle.x,
		st.descriptor,
		grid.Ptr(),
		theta.Ptr(),
	)).error("SpatialTfGridGeneratorBackward")
}

//SpatialTfSamplerForward performs the spatialtfsampleforward
func (handle *Handle) SpatialTfSamplerForward(
	st SpatialTransformerD,
	alpha CScaler,
	xD TensorD,
	x Memer,
	grid Memer,
	beta CScaler,
	yD TensorD,
	y Memer,
) error {
	return Status(C.cudnnSpatialTfSamplerForward(
		handle.x,
		st.descriptor,
		alpha.CPtr(),
		xD.descriptor,
		x.Ptr(),
		grid.Ptr(),
		beta.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("SpatialTfSamplerForward")
}

//SpatialTfSamplerBackward does the spatial Tranform Sample Backward
func (handle *Handle) SpatialTfSamplerBackward(
	st SpatialTransformerD,
	alpha CScaler,
	xD TensorD,
	x Memer,
	beta CScaler,
	dxD TensorD,
	dx Memer,
	alphaDgrid CScaler,
	dyD TensorD,
	dy Memer,
	grid Memer,
	betaDgrid CScaler,
	dGrid Memer,

) error {
	return Status(C.cudnnSpatialTfSamplerBackward(
		handle.x,
		st.descriptor,
		alpha.CPtr(),
		xD.descriptor,
		x.Ptr(),
		beta.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
		alphaDgrid.CPtr(),
		dyD.descriptor,
		dy.Ptr(),
		grid.Ptr(),
		betaDgrid.CPtr(),
		dGrid.Ptr(),
	)).error("SpatialTfSamplerBackward")
}

/*

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerBackward(
	cudnnHandle_t                              handle,
	cudnnSpatialTransformerDescriptor_t        stDesc,
	const void                                *alpha,
	const cudnnTensorDescriptor_t              xDesc,
	const void                                *x,
	const void                                *beta,
	const cudnnTensorDescriptor_t              dxDesc,
	void                                      *dx,
	const void                                *alphaDgrid,
	const cudnnTensorDescriptor_t              dyDesc,
	const void                                *dy,
	const void                                *grid,
	const void                                *betaDgrid,
	void                                      *dgrid);
*/
