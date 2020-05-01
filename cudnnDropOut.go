package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/gocudnn/gocu"
	"github.com/dereklstinson/cutil"
)

//DropOutD holds the dropout descriptor
type DropOutD struct {
	descriptor C.cudnnDropoutDescriptor_t
	gogc       bool
}

//CreateDropOutDescriptor creates a drop out descriptor to be set
func CreateDropOutDescriptor() (*DropOutD, error) {
	dod := new(DropOutD)

	err := Status(C.cudnnCreateDropoutDescriptor(&dod.descriptor)).error("CreateDropOutDescriptor()")
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
func (d *DropOutD) Set(handle *Handle, dropout float32, states cutil.Mem, bytes uint, seed uint64) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnSetDropoutDescriptor(
				d.descriptor,
				handle.x,
				C.float(dropout),
				states.Ptr(),
				C.size_t(bytes),
				C.ulonglong(seed),
			)).error("(d *DropOutD) Set")
		})
	}
	return Status(C.cudnnSetDropoutDescriptor(
		d.descriptor,
		handle.x,
		C.float(dropout),
		states.Ptr(),
		C.size_t(bytes),
		C.ulonglong(seed),
	)).error("(d *DropOutD) Set")
}

//SetUS is like Set but uses unsafe.Pointer instead of cutil.Mem
func (d *DropOutD) SetUS(handle *Handle, dropout float32, states unsafe.Pointer, bytes uint, seed uint64) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnSetDropoutDescriptor(
				d.descriptor,
				handle.x,
				C.float(dropout), states,
				C.size_t(bytes),
				C.ulonglong(seed),
			)).error("(d *DropOutD) SetUS")
		})
	}
	return Status(C.cudnnSetDropoutDescriptor(
		d.descriptor,
		handle.x,
		C.float(dropout), states,
		C.size_t(bytes),
		C.ulonglong(seed),
	)).error("(d *DropOutD) SetUS")
}

//Destroy destroys the dropout descriptor unless the the finalizer flag was set.
func (d *DropOutD) Destroy() error {
	if setfinalizer || d.gogc {
		return nil
	}
	return destroydropoutdescriptor(d)
}
func destroydropoutdescriptor(d *DropOutD) error {
	return Status(C.cudnnDestroyDropoutDescriptor(d.descriptor)).error("destroydropoutdescriptor(d *DropOutD)")
}

//Restore restores the descriptor to a previously saved-off state
func (d *DropOutD) Restore(
	handle *Handle,
	dropout float32, //probability that the input value is set to zero
	states cutil.Mem,
	bytes uint,
	seed uint64,
) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnRestoreDropoutDescriptor(
				d.descriptor,
				handle.x,
				C.float(dropout),
				states.Ptr(),
				C.size_t(bytes),
				C.ulonglong(seed),
			)).error("(d *DropOutD) Restore")
		})
	}
	return Status(C.cudnnRestoreDropoutDescriptor(
		d.descriptor,
		handle.x,
		C.float(dropout),
		states.Ptr(),
		C.size_t(bytes),
		C.ulonglong(seed),
	)).error("(d *DropOutD) Restore")
}

//RestoreUS is like Restore but uses unsafe.Pointer instead of cutil.Mem
func (d *DropOutD) RestoreUS(
	handle *Handle,
	dropout float32, //probability that the input value is set to zero
	states unsafe.Pointer,
	bytes uint,
	seed uint64,
) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnRestoreDropoutDescriptor(
				d.descriptor,
				handle.x,
				C.float(dropout),
				states,
				C.size_t(bytes),
				C.ulonglong(seed),
			)).error("(d *DropOutD) RestoreUS")
		})
	}

	return Status(C.cudnnRestoreDropoutDescriptor(
		d.descriptor,
		handle.x,
		C.float(dropout),
		states,
		C.size_t(bytes),
		C.ulonglong(seed),
	)).error("(d *DropOutD) RestoreUS")
}

//Get gets the descriptor to a previously saved-off state
func (d *DropOutD) Get(
	handle *Handle,
) (float32, cutil.Mem, uint64, error) {
	var seed C.ulonglong
	var dropout C.float
	var x unsafe.Pointer
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetDropoutDescriptor(
				d.descriptor,
				handle.x,
				&dropout,
				&x,
				&seed,
			)).error("(d *DropOutD) Get")
		})
	} else {
		err = Status(C.cudnnGetDropoutDescriptor(
			d.descriptor,
			handle.x,
			&dropout,
			&x,
			&seed,
		)).error("(d *DropOutD) Get")
	}

	return float32(dropout), gocu.WrapUnsafe(x), uint64(seed), err
}

//GetUS is like GetUS but uses unsafe.Pointer instead of cutil.Mem
func (d *DropOutD) GetUS(handle *Handle) (float32, unsafe.Pointer, uint64, error) {
	var seed C.ulonglong
	var dropout C.float
	var x unsafe.Pointer
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetDropoutDescriptor(
				d.descriptor,
				handle.x,
				&dropout,
				&x,
				&seed,
			)).error("(d *DropOutD) GetUS")
		})
	} else {
		err = Status(C.cudnnGetDropoutDescriptor(
			d.descriptor,
			handle.x,
			&dropout,
			&x,
			&seed,
		)).error("(d *DropOutD) GetUS")
	}

	return float32(dropout), x, uint64(seed), err
}

//GetStateSize returns the  state size in bytes
//Method calls a function that doesn't use DropOutD, but it is a dropout type function, and is
//used to get the size the cutil.Mem, or unsafe.Pointer needs to for state.
func (d *DropOutD) GetStateSize(handle *Handle) (uint, error) {
	var size C.size_t
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnDropoutGetStatesSize(handle.x, &size)).error("(d *DropOutD) GetStateSize")
		})
	} else {
		err = Status(C.cudnnDropoutGetStatesSize(handle.x, &size)).error("(d *DropOutD) GetStateSize")
	}

	return uint(size), err
}

//GetReserveSpaceSize returns the size of reserve space in bytes.  Method calls a function that doesn't
//use the DropOutD, but function is releveant to the DropOut operation
func (d *DropOutD) GetReserveSpaceSize(t *TensorD) (uint, error) {
	var size C.size_t
	err := Status(C.cudnnDropoutGetReserveSpaceSize(t.descriptor, &size)).error("(d *DropOutD) GetReserveSpaceSize")
	return uint(size), err
}

//Forward performs the dropoutForward
//
//	Input/Output: y,reserveSpace
func (d *DropOutD) Forward(
	handle *Handle,
	xD *TensorD, //input
	x cutil.Mem, //input
	yD *TensorD, //input
	y cutil.Mem, //input/output
	reserveSpace cutil.Mem, //input/output
	reservesize uint,
) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnDropoutForward(
				handle.x,
				d.descriptor,
				xD.descriptor,
				x.Ptr(),
				yD.descriptor,
				y.Ptr(),
				reserveSpace.Ptr(),
				C.size_t(reservesize),
			)).error("(d *DropOutD) Forward")

		})
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
	)).error("(d *DropOutD) Forward")
}

//ForwardUS is like Forward but uses unsafe.Pointer instead of cutil.Mem
func (d *DropOutD) ForwardUS(
	handle *Handle,
	xD *TensorD, x unsafe.Pointer, //input
	yD *TensorD, y unsafe.Pointer, //input/output
	reserveSpace unsafe.Pointer, reservesize uint,
) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnDropoutForward(
				handle.x,
				d.descriptor,
				xD.descriptor, x,
				yD.descriptor, y,
				reserveSpace,
				C.size_t(reservesize),
			)).error(" (d *DropOutD) ForwardUS")
		})
	}

	return Status(C.cudnnDropoutForward(
		handle.x,
		d.descriptor,
		xD.descriptor, x,
		yD.descriptor, y,
		reserveSpace,
		C.size_t(reservesize),
	)).error(" (d *DropOutD) ForwardUS")
}

//Backward performs the dropoutForward
//
//	Input/Output: dx,reserveSpace
func (d *DropOutD) Backward(
	handle *Handle,
	dyD *TensorD, //input
	dy cutil.Mem, //input
	dxD *TensorD, //input
	dx cutil.Mem, //input/output
	reserveSpace cutil.Mem, //input/output
	reservesize uint,
) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnDropoutBackward(
				handle.x,
				d.descriptor,
				dyD.descriptor,
				dy.Ptr(),
				dxD.descriptor,
				dx.Ptr(),
				reserveSpace.Ptr(),
				C.size_t(reservesize),
			)).error("(d *DropOutD) Backward")
		})
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
	)).error("(d *DropOutD) Backward")
}

//BackwardUS is like Backward but uses unsafe.Pointer instead of cutil.Mem
func (d *DropOutD) BackwardUS(
	handle *Handle,
	dyD *TensorD, //input
	dy unsafe.Pointer, //input
	dxD *TensorD, //input
	dx unsafe.Pointer, //input/output
	reserveSpace unsafe.Pointer, //input/output
	reservesize uint,
) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnDropoutBackward(
				handle.x,
				d.descriptor,
				dyD.descriptor, dy,
				dxD.descriptor, dx,
				reserveSpace, C.size_t(reservesize),
			)).error("(d *DropOutD) BackwardUS")
		})
	}

	return Status(C.cudnnDropoutBackward(
		handle.x,
		d.descriptor,
		dyD.descriptor, dy,
		dxD.descriptor, dx,
		reserveSpace, C.size_t(reservesize),
	)).error("(d *DropOutD) BackwardUS")
}
