package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import "unsafe"

//DropOutD holds the dropout descriptor
type DropOutD struct {
	descriptor C.cudnnDropoutDescriptor_t
}

//CreateDropoutDescriptor creates a dropout descriptor
func CreateDropoutDescriptor() (*DropOutD, error) {
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

//RestorDropoutDescriptor restors the descriptor to a previously saved-off state
func (d *DropOutD) RestorDropoutDescriptor(
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
	)).error("RestorDropoutDescriptor")
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
