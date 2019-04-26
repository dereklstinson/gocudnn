package npp

/*
#include <npps_support_functions.h>
#include <nppdefs.h>


*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

//Malloc8u is an allocator of *Uint8
func Malloc8u(nSize int32) (x *Uint8) {
	x = new(Uint8)
	//y := ()
	x.wrap(C.nppsMalloc_8u((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc8s is an allocator of *Int8
func Malloc8s(nSize int32) (x *Int8) {
	x = new(Int8)
	y := (C.nppsMalloc_8s((C.int)(nSize)))
	x.wrap(y)
	runtime.SetFinalizer(x, nppsFree)
	return x

}

//Malloc16u is an allocator of *Uint16
func Malloc16u(nSize int32) *Uint16 {
	x := (*Uint16)(C.nppsMalloc_16u((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc16s is an allocator of *Int16
func Malloc16s(nSize int32) (x *Int16) {
	x = new(Int16)
	y := (C.nppsMalloc_16s((C.int)(nSize)))
	x.wrap(y)
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc16sc is an allocator of *Int16Complex
func Malloc16sc(nSize int32) *Int16Complex {
	x := (*Int16Complex)(C.nppsMalloc_16sc((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc32u is an allocator of *Uint32
func Malloc32u(nSize int32) *Uint32 {
	x := (*Uint32)(C.nppsMalloc_32u((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc32s is an allocator of *Int32
func Malloc32s(nSize int32) *Int32 {
	x := (*Int32)(C.nppsMalloc_32s((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc32sc is an allocator of *Int32Complex
func Malloc32sc(nSize int32) *Int32Complex {
	x := (*Int32Complex)(C.nppsMalloc_32sc((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc32f is an allocator of *Float32
func Malloc32f(nSize int32) *Float32 {
	x := (*Float32)(C.nppsMalloc_32f((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc32fc is an allocator of *Float32Complex
func Malloc32fc(nSize int32) *Float32Complex {
	x := (*Float32Complex)(C.nppsMalloc_32fc((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc64s is an allocator of *Int64
func Malloc64s(nSize int32) *Int64 {
	x := (*Int64)(C.nppsMalloc_64s((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc64sc is an allocator of *Int64Complex
func Malloc64sc(nSize int32) *Int64Complex {
	x := (*Int64Complex)(C.nppsMalloc_64sc((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc64f is an allocator of *Float64
func Malloc64f(nSize int32) *Float64 {
	x := (*Float64)(C.nppsMalloc_64f((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

//Malloc64fc is an allocator of *Float64Complex
func Malloc64fc(nSize int32) *Float64Complex {
	x := (*Float64Complex)(C.nppsMalloc_64fc((C.int)(nSize)))
	runtime.SetFinalizer(x, nppsFree)
	return x
}

func nppsFree(x interface{}) error {
	switch y := x.(type) {
	case *Uint8:
		C.nppsFree(y.p)
		y = nil
		return nil
	case *Int8:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Int16:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Uint16:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Uint32:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Int32:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Int64:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Float32:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Float64:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Float32Complex:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Int32Complex:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Int64Complex:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Float64Complex:
		C.nppsFree(unsafe.Pointer(y))
		return nil
	case *Int16Complex:
		C.nppsFree(unsafe.Pointer(y))
		return nil

	}
	return errors.New("Unsupported type")

}
