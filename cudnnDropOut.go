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

func (d *DropOutD) keepsalive() {
	runtime.KeepAlive(d)
}

//NewDropoutDescriptor creates a dropout descriptor
func (d DropOut) NewDropoutDescriptor(handle *Handle,
	dropout float32, //probability that the input value is set to zero
	states gocu.Mem,
	bytes uint,
	seed uint64,
) (descriptor *DropOutD, err error) {
	var desc C.cudnnDropoutDescriptor_t
	err = Status(C.cudnnCreateDropoutDescriptor(&desc)).error("CreateDropoutDescriptor")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetDropoutDescriptor(
		desc,
		handle.x,
		C.float(dropout),
		states.Ptr(),
		C.size_t(bytes),
		C.ulonglong(seed),
	)).error("SetDropoutDescriptor")
	if err != nil {
		return nil, err
	}
	descriptor = &DropOutD{
		descriptor: desc,
	}
	if setfinalizer {
		runtime.SetFinalizer(descriptor, destroydropoutdescriptor)
	}
	return descriptor, nil
}

//DestroyDescriptor destroys the dropout descriptor
func (d *DropOutD) DestroyDescriptor() error {
	return destroydropoutdescriptor(d)
}
func destroydropoutdescriptor(d *DropOutD) error {
	return Status(C.cudnnDestroyDropoutDescriptor(d.descriptor)).error("DestroyDescriptor")
}

//RestoreDropoutDescriptor restores the descriptor to a previously saved-off state
func (d *DropOutD) RestoreDropoutDescriptor(
	handle *Handle,
	dropout float32, //probability that the input value is set to zero
	states gocu.Mem,
	bytes uint,
	seed uint64,
) error {
	if setkeepalive {
		keepsalivebuffer(d, handle, states)
	}
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
	states gocu.Mem,

) (float32, gocu.Mem, uint64, error) {
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
	if setkeepalive {
		keepsalivebuffer(d, handle, states)
	}
	return float32(dropout), states, uint64(seed), err
}

//DropoutGetStateSize returns the  state size in bytes
func (d DropOutFuncs) DropoutGetStateSize(handle *Handle) (uint, error) {
	var size C.size_t
	err := Status(C.cudnnDropoutGetStatesSize(handle.x, &size)).error("DropoutGetStateSize")
	if setkeepalive {
		keepsalivebuffer(handle)
	}
	return uint(size), err
}

//DropoutGetReserveSpaceSize returns the size of reserve space in bytes
func (d DropOutFuncs) DropoutGetReserveSpaceSize(t *TensorD) (uint, error) {
	var size C.size_t
	err := Status(C.cudnnDropoutGetReserveSpaceSize(t.descriptor, &size)).error("DropoutGetReserveSpaceSize")
	if setkeepalive {
		keepsalivebuffer(t)
	}
	return uint(size), err
}

//DropoutForward performs the dropoutForward
func (d *DropOutD) DropoutForward(
	handle *Handle,
	xD *TensorD, //input
	x gocu.Mem, //input
	yD *TensorD, //input
	y gocu.Mem, //input/output
	reserveSpace gocu.Mem, //input/output
	reservesize uint,
) error {
	if setkeepalive {
		keepsalivebuffer(handle, d, xD, x, yD, y, reserveSpace)
	}
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

//DropoutBackward performs the dropoutForward
func (d *DropOutD) DropoutBackward(
	handle *Handle,
	dyD *TensorD, //input
	dy gocu.Mem, //input
	dxD *TensorD, //input
	dx gocu.Mem, //input/output
	reserveSpace gocu.Mem, //input/output
	reservesize uint,
) error {
	if setkeepalive {
		keepsalivebuffer(handle, d, dxD, dx, dyD, dy, reserveSpace)
	}
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
