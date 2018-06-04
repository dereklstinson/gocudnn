package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//DropoutGetStateSize returns the  state size in bytes
func (handle *Handle) DropoutGetStateSize() (SizeT, error) {
	var size C.size_t
	err := Status(C.cudnnDropoutGetStatesSize(handle.x, &size)).error("DropoutGetStateSize")
	return SizeT(size), err
}

//DropoutGetReserveSpaceSize returns the size of reserve space in bytes
func (t *TensorD) DropoutGetReserveSpaceSize() (SizeT, error) {
	var size C.size_t
	err := Status(C.cudnnDropoutGetReserveSpaceSize(t.descriptor, &size)).error("DropoutGetReserveSpaceSize")
	return SizeT(size), err
}

//DropoutForward performs the dropoutForward
func (handle *Handle) DropoutForward(
	doD *DropOutD, //intput
	xD *TensorD, //input
	x Memer, //input
	yD *TensorD, //input
	y Memer, //input/output
	reserveSpace Memer, //input/output
) error {
	return Status(C.cudnnDropoutForward(
		handle.x,
		doD.descriptor,
		xD.descriptor,
		x.Ptr(),
		yD.descriptor,
		y.Ptr(),
		reserveSpace.Ptr(),
		reserveSpace.ByteSize().c(),
	)).error("DropoutForward")
}

//DropoutBackward performs the dropoutForward
func (handle *Handle) DropoutBackward(
	doD *DropOutD, //intput
	dyD *TensorD, //input
	dy Memer, //input
	dxD *TensorD, //input
	dx Memer, //input/output
	reserveSpace Memer, //input/output
) error {
	return Status(C.cudnnDropoutBackward(
		handle.x,
		doD.descriptor,
		dyD.descriptor,
		dy.Ptr(),
		dxD.descriptor,
		dx.Ptr(),
		reserveSpace.Ptr(),
		reserveSpace.ByteSize().c(),
	)).error("DropoutBackward")
}
