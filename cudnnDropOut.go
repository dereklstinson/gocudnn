package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import "unsafe"

//DropOut is the struct that is used to call dropout functions
type DropOut struct {
	Funcs DropOutFuncs
}

//DropOutFuncs is an empty struct used to call DropOut Functions
type DropOutFuncs struct {
}

//DropOutD holds the dropout descriptor
type DropOutD struct {
	descriptor C.cudnnDropoutDescriptor_t
}

//CreateDropoutDescriptor creates a dropout descriptor
func (d DropOut) CreateDropoutDescriptor() (*DropOutD, error) {
	var desc C.cudnnDropoutDescriptor_t
	err := Status(C.cudnnCreateDropoutDescriptor(&desc)).error("CreateDropoutDescriptor")
	if err != nil {
		return nil, err
	}
	return &DropOutD{
		descriptor: desc,
	}, nil
}

//DestroyDescriptor destroys the dropout descriptor
func (d *DropOutD) DestroyDescriptor() error {
	return Status(C.cudnnDestroyDropoutDescriptor(d.descriptor)).error("DestroyDescriptor")
}

//SetDropoutDescriptor sets the Drop Out Descriptor
func (d *DropOutD) SetDropoutDescriptor(
	handle *Handle,
	dropout float32, //probability that the input value is set to zero
	states Memer,
	bytes SizeT,
	seed uint64,
) error {
	return Status(C.cudnnSetDropoutDescriptor(
		d.descriptor,
		handle.x,
		C.float(dropout),
		states.Ptr(),
		C.size_t(bytes),
		C.ulonglong(seed),
	)).error("SetDropoutDescriptor")
}

//RestoreDropoutDescriptor restors the descriptor to a previously saved-off state
func (d *DropOutD) RestoreDropoutDescriptor(
	handle *Handle,
	dropout float32, //probability that the input value is set to zero
	states Memer,
	bytes SizeT,
	seed uint64,
) error {
	return Status(C.cudnnRestoreDropoutDescriptor(
		d.descriptor,
		handle.x,
		C.float(dropout),
		states.Ptr(),
		C.size_t(bytes),
		C.ulonglong(seed),
	)).error("RestoreDropoutDescriptor")
}

//GetDropoutDescriptor gets the descriptor to a previously saved-off state
func (d *DropOutD) GetDropoutDescriptor(
	handle *Handle,
	states Memer,

) (float32, Memer, uint64, error) {
	var seed C.ulonglong
	var dropout C.float
	var x unsafe.Pointer
	x = states.Ptr()
	err := Status(C.cudnnGetDropoutDescriptor(
		d.descriptor,
		handle.x,
		&dropout,
		&x,
		&seed,
	)).error("GetDropoutDescriptor")
	return float32(dropout), states, uint64(seed), err
}

//DropoutGetStateSize returns the  state size in bytes
func (d DropOutFuncs) DropoutGetStateSize(handle *Handle) (SizeT, error) {
	var size C.size_t
	err := Status(C.cudnnDropoutGetStatesSize(handle.x, &size)).error("DropoutGetStateSize")
	return SizeT(size), err
}

//DropoutGetReserveSpaceSize returns the size of reserve space in bytes
func (d DropOutFuncs) DropoutGetReserveSpaceSize(t *TensorD) (SizeT, error) {
	var size C.size_t
	err := Status(C.cudnnDropoutGetReserveSpaceSize(t.descriptor, &size)).error("DropoutGetReserveSpaceSize")
	return SizeT(size), err
}

//DropoutForward performs the dropoutForward
func (d DropOutFuncs) DropoutForward(
	handle *Handle,
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
func (d DropOutFuncs) DropoutBackward(
	handle *Handle,
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
