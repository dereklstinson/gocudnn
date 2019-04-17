package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//DropOutD holds the dropout descriptor
type DropOutD struct {
	descriptor C.cudnnDropoutDescriptor_t
	gogc       bool
}

//CreateDropOutDescriptor creates a drop out descriptor to be set
func CreateDropOutDescriptor() (*DropOutD, error) {
	dod := new(DropOutD)

	err := Status(C.cudnnCreateDropoutDescriptor(&dod.descriptor)).error("CreateDropoutDescriptor")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		dod.gogc = true
		runtime.SetFinalizer(dod, destroydropoutdescriptor)
	}
	return nil, nil
}

//Set sets the drop out descriptor
func (d *DropOutD) Set(handle *Handle, dropout float32, states gocu.Mem, bytes uint, seed uint64) error {
	return Status(C.cudnnSetDropoutDescriptor(
		d.descriptor,
		handle.x,
		C.float(dropout),
		states.Ptr(),
		C.size_t(bytes),
		C.ulonglong(seed),
	)).error("SetDropoutDescriptor")
}

//SetUS is like Set but uses unsafe.Pointer instead of gocu.Mem
func (d *DropOutD) SetUS(handle *Handle, dropout float32, states unsafe.Pointer, bytes uint, seed uint64) error {
	return Status(C.cudnnSetDropoutDescriptor(
		d.descriptor,
		handle.x,
		C.float(dropout), states,
		C.size_t(bytes),
		C.ulonglong(seed),
	)).error("SetDropoutDescriptor")
}

//Destroy destroys the dropout descriptor unless the the finalizer flag was set.
func (d *DropOutD) Destroy() error {
	if setfinalizer || d.gogc {
		return nil
	}
	return destroydropoutdescriptor(d)
}
func destroydropoutdescriptor(d *DropOutD) error {
	return Status(C.cudnnDestroyDropoutDescriptor(d.descriptor)).error("DestroyDescriptor")
}

//Restore restores the descriptor to a previously saved-off state
func (d *DropOutD) Restore(
	handle *Handle,
	dropout float32, //probability that the input value is set to zero
	states gocu.Mem,
	bytes uint,
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

//RestoreUS is like Restore but uses unsafe.Pointer instead of gocu.Mem
func (d *DropOutD) RestoreUS(
	handle *Handle,
	dropout float32, //probability that the input value is set to zero
	states unsafe.Pointer,
	bytes uint,
	seed uint64,
) error {

	return Status(C.cudnnRestoreDropoutDescriptor(
		d.descriptor,
		handle.x,
		C.float(dropout),
		states,
		C.size_t(bytes),
		C.ulonglong(seed),
	)).error("RestoreDropoutDescriptor")
}

//Get gets the descriptor to a previously saved-off state
func (d *DropOutD) Get(
	handle *Handle,
) (float32, gocu.Mem, uint64, error) {
	var seed C.ulonglong
	var dropout C.float
	var x unsafe.Pointer
	err := Status(C.cudnnGetDropoutDescriptor(
		d.descriptor,
		handle.x,
		&dropout,
		&x,
		&seed,
	)).error("GetDropoutDescriptor")

	return float32(dropout), gocu.WrapUnsafe(x), uint64(seed), err
}

//GetUS is like GetUS but uses unsafe.Pointer instead of gocu.Mem
func (d *DropOutD) GetUS(handle *Handle) (float32, unsafe.Pointer, uint64, error) {
	var seed C.ulonglong
	var dropout C.float
	var x unsafe.Pointer
	err := Status(C.cudnnGetDropoutDescriptor(
		d.descriptor,
		handle.x,
		&dropout,
		&x,
		&seed,
	)).error("GetDropoutDescriptor")

	return float32(dropout), x, uint64(seed), err
}

//GetStateSize returns the  state size in bytes
//Method calls a function that doesn't use DropOutD, but it is a dropout type function, and is
//used to get the size the gocu.Mem, or unsafe.Pointer needs to for state.
func (d *DropOutD) GetStateSize(handle *Handle) (uint, error) {
	var size C.size_t
	err := Status(C.cudnnDropoutGetStatesSize(handle.x, &size)).error("DropoutGetStateSize")

	return uint(size), err
}

//GetReserveSpaceSize returns the size of reserve space in bytes.  Method calls a function that doesn't
//use the DropOutD, but function is releveant to the DropOut operation
func (d *DropOutD) GetReserveSpaceSize(t *TensorD) (uint, error) {
	var size C.size_t
	err := Status(C.cudnnDropoutGetReserveSpaceSize(t.descriptor, &size)).error("DropoutGetReserveSpaceSize")
	return uint(size), err
}

//Forward performs the dropoutForward
//
//	Input/Output: y,reserveSpace
func (d *DropOutD) Forward(
	handle *Handle,
	xD *TensorD, //input
	x gocu.Mem, //input
	yD *TensorD, //input
	y gocu.Mem, //input/output
	reserveSpace gocu.Mem, //input/output
	reservesize uint,
) error {

	return Status(C.cudnnDropoutForward(
		handle.x,
		d.descriptor,
		xD.descriptor,
		x.Ptr(),
		yD.descriptor,
		y.Ptr(),
		reserveSpace.Ptr(),
		C.size_t(reservesize),
	)).error("DropoutForward")
}

//ForwardUS is like Forward but uses unsafe.Pointer instead of gocu.Mem
func (d *DropOutD) ForwardUS(
	handle *Handle,
	xD *TensorD, x unsafe.Pointer, //input
	yD *TensorD, y unsafe.Pointer, //input/output
	reserveSpace unsafe.Pointer, reservesize uint,
) error {

	return Status(C.cudnnDropoutForward(
		handle.x,
		d.descriptor,
		xD.descriptor, x,
		yD.descriptor, y,
		reserveSpace,
		C.size_t(reservesize),
	)).error("DropoutForward")
}

//Backward performs the dropoutForward
//
//	Input/Output: dx,reserveSpace
func (d *DropOutD) Backward(
	handle *Handle,
	dyD *TensorD, //input
	dy gocu.Mem, //input
	dxD *TensorD, //input
	dx gocu.Mem, //input/output
	reserveSpace gocu.Mem, //input/output
	reservesize uint,
) error {

	return Status(C.cudnnDropoutBackward(
		handle.x,
		d.descriptor,
		dyD.descriptor,
		dy.Ptr(),
		dxD.descriptor,
		dx.Ptr(),
		reserveSpace.Ptr(),
		C.size_t(reservesize),
	)).error("DropoutBackward")
}

//BackwardUS is like Backward but uses unsafe.Pointer instead of gocu.Mem
func (d *DropOutD) BackwardUS(
	handle *Handle,
	dyD *TensorD, //input
	dy unsafe.Pointer, //input
	dxD *TensorD, //input
	dx unsafe.Pointer, //input/output
	reserveSpace unsafe.Pointer, //input/output
	reservesize uint,
) error {

	return Status(C.cudnnDropoutBackward(
		handle.x,
		d.descriptor,
		dyD.descriptor, dy,
		dxD.descriptor, dx,
		reserveSpace, C.size_t(reservesize),
	)).error("DropoutBackward")
}
