package gocu

import "unsafe"

//CudaPtr can be used to allocate mem that is used by cuda.
//I didn't want to do this but it is unintuitive in the means of making it otherwise.
//The reason being is that cgo will yell at you if you try to use an unsafe pointer using a go type.
//Just make one by the built in x:=new(gocu.DevicePtr)
type CudaPtr struct {
	d unsafe.Pointer
}

//WrapUnsafe wraps an unsafe.Pointer around a *CudaPtr
func WrapUnsafe(d unsafe.Pointer) *CudaPtr {
	return &CudaPtr{
		d: d,
	}
}

//Ptr returns the unsafepointer
func (d *CudaPtr) Ptr() unsafe.Pointer {
	return d.d
}

//DPtr returns the *unsafe.Pointer
func (d *CudaPtr) DPtr() *unsafe.Pointer {
	return &d.d
}

//Allocator allocates memory for cuda.  //Example can be seen in cudart.
type Allocator interface {
	Malloc(size uint) (*CudaPtr, error)
}
