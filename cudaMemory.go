package gocudnn

/*
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <driver_types.h>
*/
import "C"

import (
	"errors"
	"unsafe"
)

//Malloced is a non garbage collection varient to what was used in buffer.
//It gives the user more control over the memory for cases of building neural networks
//on the GPU.  Just remember to free the memory or the gpu will fill up fast.
//Malloced contains info on a chunk of memory on the device
type Malloced struct {
	ptr       unsafe.Pointer
	size      SizeT
	typevalue string
	onhost    bool
	onmanaged bool
}

//Ptr returns an unsafe.Pointer
func (mem *Malloced) Ptr() unsafe.Pointer {
	return mem.ptr
}

//Stored returns the Location Flag of the memory
func (mem *Malloced) Stored() Location {
	if mem.ptr == nil {
		return 0
	}
	if mem.onhost == true {
		return 3
	}
	if mem.onmanaged == true {
		return 4
	}
	return 2

}

//GoPointer holds a pointer to a slice
type GoPointer struct {
	ptr       unsafe.Pointer
	size      SizeT
	typevalue string
}

//ByteSize returns the size of the memory chunck
func (mem *Malloced) ByteSize() SizeT {
	if mem.ptr == nil {
		return SizeT(0)
	}
	return mem.size
}

//ByteSize returns the size of the memory chunck
func (mem *GoPointer) ByteSize() SizeT {
	if mem.ptr == nil {
		return SizeT(0)
	}
	return mem.size
}

//Ptr returns an unsafe.Pointer
func (mem *GoPointer) Ptr() unsafe.Pointer {
	return mem.ptr
}

//Stored returns an Location which can be used to by other programs
func (mem *GoPointer) Stored() Location {
	if mem.ptr == nil {
		return 0
	}
	return 1
}

//Free unassignes the pointers and does the garbage collection
func (mem *GoPointer) Free() error {
	mem.size = 0
	mem.ptr = nil
	mem.typevalue = ""
	return nil
}

//CudaMemCopy copies some memory from src to dest.  If default is selected and if the system supports unified virtual addressing then the transfer is inferred.
func CudaMemCopy(dest Memer, src Memer, count SizeT, kind MemcpyKind) error {
	err := C.cudaMemcpy(dest.Ptr(), src.Ptr(), count.c(), kind.c())
	return newErrorRuntime("cudaMemcpy", err)
}

/*
//CudaMemCopy copies some memory from src to dest.  If default is selected and if the system supports unified virtual addressing then the transfer is inferred.
func (mem *Malloced) CudaMemCopy(dest Memer, src Memer, count SizeT, kind MemcpyKind) error {
	err := C.cudaMemcpy(dest.Ptr(), src.Ptr(), count.c(), kind.c())
	return newErrorRuntime("cudaMemcpy", err)
}
*/

//Free Frees the memory on the device
func (mem *Malloced) Free() error {
	if mem.onhost == true {
		err := C.cudaFreeHost(mem.ptr)
		mem.size = 0
		mem.typevalue = ""
		return newErrorRuntime("Free", err)
	}
	err := C.cudaFree(mem.ptr)
	mem.ptr = nil
	mem.size = 0
	mem.typevalue = ""
	return newErrorRuntime("Free", err)
}

//Malloc returns struct Malloced that has a pointer memory that is now allocated to the device
func Malloc(size SizeT) (*Malloced, error) {
	var gpu Malloced
	gpu.size = size
	err := C.cudaMalloc(&gpu.ptr, gpu.size.c())
	return &gpu, newErrorRuntime("Malloc", err)
}

//MallocHost - Allocates page-locked memory on the host. used specifically for fast calls from the host.
func MallocHost(size SizeT) (*Malloced, error) {
	var mem Malloced
	mem.size = size
	x := C.cudaMallocHost(&mem.ptr, mem.size.c())
	err := newErrorRuntime("MallocHost", x)
	if err != nil {
		return nil, err
	}
	mem.onhost = true
	return &mem, nil
}

//ManagedMemFlag used to pass ManagedMem flags through methods
type ManagedMemFlag struct {
}

//ManagedMem is used for mem management flags
type ManagedMem C.uint

func (mem ManagedMem) c() C.uint { return C.uint(mem) }

//Global returns ManagedMem(1)
func (mem ManagedMemFlag) Global() ManagedMem {
	return ManagedMem(1)
}

//Host returns ManagedMem(2)
func (mem ManagedMemFlag) Host() ManagedMem {
	return ManagedMem(2)
}

//MallocManaged is useful if devices support unified virtual memory.
func MallocManaged(size SizeT, management ManagedMem) (*Malloced, error) {
	var mem Malloced
	mem.onmanaged = true
	mem.size = size
	err := C.cudaMallocManaged(&mem.ptr, size.c(), management.c())
	return &mem, newErrorRuntime("MallocManaged", err)
}

//MakeGoPointer takes a slice and gives a GoPointer for that slice.  I wouldn't use that slice anylonger
func MakeGoPointer(input interface{}) (*GoPointer, error) {
	var ptr GoPointer

	var err error
	switch val := input.(type) {
	case []int:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "int"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case []int8:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "int8"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil

	case []byte:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "byte"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case []float64:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "float64"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case []float32:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "float32"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case []int32:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "int32"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case []uint32:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "uint32"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	default:
		return nil, errors.New("Unsupported Type")
	}
}

//FindSizeT finds the SizeT of the array
func FindSizeT(input interface{}) (SizeT, error) {
	switch val := input.(type) {
	case []int:
		return SizeT(len(val) * 8), nil
	case []byte:
		return SizeT(len(val)), nil
	case []int8:
		return SizeT(len(val)), nil
	case []float64:
		return SizeT(len(val) * 8), nil
	case []float32:
		return SizeT(len(val) * 4), nil
	case []int32:
		return SizeT(len(val) * 4), nil
	case []uint32:
		return SizeT(len(val) * 4), nil
	default:
		return SizeT(0), errors.New("Unsupported Type")
	}
}

/*

Flags


*/

//Memory holds Memory flags will do more later
type Memory struct {
	Flgs MemcpyKindFlag
}

//MemcpyKindFlag used to pass flags for MemcpyKind through methods
type MemcpyKindFlag struct {
}

//MemcpyKind are enum flags for mem copy
type MemcpyKind C.enum_cudaMemcpyKind

//HostToHost return MemcpyKind(C.cudaMemcpyHostToHost )
func (m MemcpyKindFlag) HostToHost() MemcpyKind {
	return MemcpyKind(C.cudaMemcpyHostToHost)
}

//HostToDevice 	return MemcpyKind(C.cudaMemcpyHostToDevice )
func (m MemcpyKindFlag) HostToDevice() MemcpyKind {
	return MemcpyKind(C.cudaMemcpyHostToDevice)
}

//DeviceToHost return MemcpyKind(C.cudaMemcpyDeviceToHost )
func (m MemcpyKindFlag) DeviceToHost() MemcpyKind {
	return MemcpyKind(C.cudaMemcpyDeviceToHost)
}

//DeviceToDevice return MemcpyKind(C.cudaMemcpyDeviceToDevice )
func (m MemcpyKindFlag) DeviceToDevice() MemcpyKind {
	return MemcpyKind(C.cudaMemcpyDeviceToDevice)
}

//Default return MemcpyKind(C.cudaMemcpyDefault )
func (m MemcpyKindFlag) Default() MemcpyKind {
	return MemcpyKind(C.cudaMemcpyDefault)
}
func (m MemcpyKind) c() C.enum_cudaMemcpyKind { return C.enum_cudaMemcpyKind(m) }

func MemCpyDeterminer(src, dest Memer) (MemcpyKind, error) {
	var L LocationFlag
	var M MemcpyKindFlag
	if dest.Stored() == L.NotAllocated() {
		return M.HostToHost(), errors.New("Destination Memory Not Allocated")
	}
	if src.Stored() == L.NotAllocated() {
		return M.HostToHost(), errors.New("Source Memory Not Allocated")
	}
	if dest.Stored() == L.Unified() {
		return M.Default(), nil
	}
	if dest.Stored() == L.GoSideHost() {
		if src.Stored() == dest.Stored() {
			return M.HostToHost(), errors.New("No Cuda Allocation Needed")
		}
		if src.Stored() == L.Device() {
			return M.DeviceToHost(), nil
		}
		if src.Stored() == L.Unified() {
			return M.Default(), nil
		}
		if src.Stored() == L.CudaHost() {
			return M.HostToHost(), nil
		}
		return M.HostToHost(), errors.New("not supported for gocudnn")
	}
	if dest.Stored() == L.Device() {
		if src.Stored() == dest.Stored() {
			return M.DeviceToDevice(), nil
		}
		if src.Stored() == L.GoSideHost() {
			return M.HostToDevice(), nil
			if src.Stored() == L.CudaHost() {
				return M.HostToDevice(), nil
			}
			return M.Default(), errors.New("not supported for gocudnn")
		}
		if dest.Stored() == L.CudaHost() {
			if src.Stored() == dest.Stored() {
				return M.HostToHost(), nil
			}
			if src.Stored() == L.GoSideHost() {
				return M.HostToHost(), nil
			}
			if src.Stored() == L.Device() {
				return M.DeviceToHost(), nil
			}
			return M.Default(), errors.New("not supported for gocudnn")
		}

	}
	if dest.Stored() == L.Unified() {
		if src.Stored() == dest.Stored() {
			return M.Default(), nil
		}
		if src.Stored() == L.GoSideHost() {
			return M.Default(), nil
		}
		return M.Default(), errors.New("not Supported for gocudnn")
	}
	return M.HostToHost(), errors.New("not supported for gocudnn")
}
