package cudart

/*
#include<cuda_runtime_api.h>
typedef struct cudaPointerAttributes cudaPointerAttributes;
typedef enum cudaMemoryType cudaMemoryType;
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//MallocManagedHost uses the Unified memory mangement system and starts it off in the host
//It will also set a finalizer on the memory for GC
func MallocManagedHost(mem cutil.Mem, size uint) error {
	err := newErrorRuntime("MallocManaged", C.cudaMallocManaged(mem.DPtr(), C.size_t(size), C.cudaMemAttachHost))
	if err != nil {
		return err
	}
	err = Memset(mem, 0, (size))
	if err != nil {
		return err
	}
	runtime.SetFinalizer(mem, hostfreemem)

	return nil
}

/*
//MallocManagedHostUS is like MallocManaged but using unsafe.Pointer
func MallocManagedHostUS(mem unsafe.Pointer, size uint) error {
	err := newErrorRuntime("MallocManaged", C.cudaMallocManaged(&mem, C.size_t(size), C.uint(2)))
	runtime.SetFinalizer(mem, hostfreememUS)
	return err

}
*/

//MallocManagedGlobal uses the Unified memory mangement system and starts it off in the Device
//It will also set a finalizer on the memory for GC
func MallocManagedGlobal(mem cutil.Mem, size uint) error {
	err := newErrorRuntime("MallocManaged", C.cudaMallocManaged(mem.DPtr(), C.size_t(size), C.cudaMemAttachGlobal))
	runtime.SetFinalizer(mem, devicefreemem)
	return err
}

/*
//MallocManagedGlobalUS is like MallocManagedGlobal but uses unsafe.Pointer
func MallocManagedGlobalUS(mem unsafe.Pointer, size uint) error {
	err := newErrorRuntime("MallocManaged", C.cudaMallocManaged(&mem, C.size_t(size), C.uint(1)))
	runtime.SetFinalizer(mem, devicefreememUS)
	return err
}
*/

//Malloc will allocate memory to the device the size that was passed.
//It will also set the finalizer for GC
func Malloc(mem cutil.Mem, sizet uint) error {
	err := newErrorRuntime("Malloc", C.cudaMalloc(mem.DPtr(), C.size_t(sizet)))
	if err != nil {
		return err
	}
	err = Memset(mem, 0, (sizet))
	if err != nil {
		return err
	}
	runtime.SetFinalizer(mem, devicefreemem)
	return nil
}

//MallocHost will allocate memory on the host for cuda use.
//
func MallocHost(mem cutil.Mem, sizet uint) error {
	err := newErrorRuntime("Malloc", C.cudaMalloc(mem.DPtr(), C.size_t(sizet)))
	if err != nil {
		return err
	}
	err = Memset(mem, 0, (sizet))
	if err != nil {
		return err
	}
	runtime.SetFinalizer(mem, hostfreemem)
	return err
}

//PointerGetAttributes returns the atributes
func PointerGetAttributes(mem cutil.Mem) (Atribs, error) {
	var x C.cudaPointerAttributes
	cuerr := C.cudaPointerGetAttributes(&x, mem.Ptr())
	err := newErrorRuntime("Attributes", cuerr)
	if err != nil {
		return Atribs{}, err
	}
	var managed bool
	if x.isManaged > C.int(0) {
		managed = true
	}

	return Atribs{
		Type:    MemType(x.memoryType),
		Device:  int32(x.device),
		DPtr:    unsafe.Pointer(x.devicePointer),
		HPtr:    unsafe.Pointer(x.hostPointer),
		Managed: managed,
	}, nil
}

//MemsetUS is like Memset but with unsafe.pointer
func MemsetUS(mem unsafe.Pointer, value int32, count uint) error {
	err := C.cudaMemset(mem, C.int(value), C.size_t(count))

	return newErrorRuntime("cudaMemset", err)
}

//Memset sets the value for each byte in device memory
func Memset(mem cutil.Mem, value int32, count uint) error {
	err := C.cudaMemset(mem.Ptr(), C.int(value), C.size_t(count))

	return newErrorRuntime("cudaMemset", err)
}

//Atribs are a memories attributes on the device side
type Atribs struct {
	Type    MemType
	Device  int32
	DPtr    unsafe.Pointer
	HPtr    unsafe.Pointer
	Managed bool
}

//MemType is a typedefed C.cudaMemoryType
type MemType C.cudaMemoryType

/*

finalizer functions

*/

func devicefreemem(mem cutil.Mem) error {

	err := newErrorRuntime("devicefree", C.cudaFree(mem.Ptr()))
	if err != nil {
		return nil
	}
	mem = nil
	return nil
}
func devicefreememUS(mem unsafe.Pointer) error {

	err := newErrorRuntime("devicefree", C.cudaFree(mem))
	if err != nil {
		return nil
	}
	mem = nil
	return nil
}
func hostfreememUS(mem unsafe.Pointer) error {
	err := newErrorRuntime("hostfree", C.cudaFreeHost(mem))
	if err != nil {
		return err
	}
	mem = nil
	return nil
}
func hostfreemem(mem cutil.Mem) error {
	err := newErrorRuntime("hostfree", C.cudaFreeHost(mem.Ptr()))
	if err != nil {
		return err
	}
	mem = nil
	return nil
}
