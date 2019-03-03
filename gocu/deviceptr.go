package gocu

import "unsafe"

//DevicePtr can be used to allocate mem to the device.
//I didn't want to do this but it is unintuitive in the means of making it otherwise.
//The reason being is that cgo will yell at you if you try to use an unsafe pointer using a go type.
//Just make one by the built in x:=new(gocu.DevicePtr)
type DevicePtr struct {
	d unsafe.Pointer
}

//Ptr returns the unsafepointer
func (d *DevicePtr) Ptr() unsafe.Pointer {
	return d.d
}

//DPtr returns the *unsafe.Pointer
func (d *DevicePtr) DPtr() *unsafe.Pointer {
	return &d.d
}
