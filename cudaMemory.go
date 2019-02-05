package gocudnn

/*
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <driver_types.h>

typedef struct cudaPointerAttributes cudaPointerAttributes;
typedef enum cudaMemoryType cudaMemoryType;
*/
import "C"

import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/half"
	"github.com/pkg/errors"
)

//Memer is an interface for memory
type Memer interface {
	Ptr() unsafe.Pointer
	ByteSize() SizeT
	Free() error
	Stored() Location
	FillSlice(interface{}) error
	keepsalive()
}

//Malloced is a non garbage collection memory that is stored on the device.  When done with it be sure to destroy it.
type Malloced struct {
	ptr       unsafe.Pointer
	size      SizeT
	typevalue string
	onhost    bool
	onmanaged bool
	devptr    C.CUdeviceptr
}

//Offset returns the offset pointer
func (mem *Malloced) Offset(offset uint) unsafe.Pointer {
	return unsafe.Pointer(uintptr(mem.ptr) + uintptr(offset))
}

//OffSet will return the offset address from the pointer passed
func OffSet(point unsafe.Pointer, unitsize int, offset int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(point) + uintptr(unitsize*offset))
}
func (mem *Malloced) keepsalive() {
	runtime.KeepAlive(mem)
}

//KeepAlive keeps the mem alive
func (mem *Malloced) KeepAlive() {
	runtime.KeepAlive(mem)
}

//Atribs are a memories attributes on the device side
type Atribs struct {
	Type    MemType
	Device  int32
	DPtr    unsafe.Pointer
	HPtr    unsafe.Pointer
	Managed bool
}

func cfloattofloat32(input []C.float) []float32 {
	slice := make([]float32, len(input))
	for i := 0; i < len(input); i++ {
		slice[i] = float32(input[i])
	}
	return slice
}

//Atributes returns the atributes
func (mem *Malloced) Atributes() (Atribs, error) {
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
	if setkeepalive {
		mem.keepsalive()
	}
	return Atribs{
		Type:    MemType(x.memoryType),
		Device:  int32(x.device),
		DPtr:    unsafe.Pointer(x.devicePointer),
		HPtr:    unsafe.Pointer(x.hostPointer),
		Managed: managed,
	}, nil
}

//Set sets the value for each byte in device memory
func (mem *Malloced) Set(value int32) error {
	err := C.cudaMemset(mem.ptr, C.int(value), mem.size.c())
	if setkeepalive {
		mem.keepsalive()
	}
	return newErrorRuntime("cudaMemset", err)
}

//FillSlice will fill a slice array that is passed to cuda
func (mem *Malloced) FillSlice(input interface{}) error {
	var kind MemcpyKindFlag
	bsize, err := FindSizeT(input)
	if err != nil {
		return err
	}
	ptr, err := MakeGoPointer(input)
	if err != nil {
		return err
	}
	if setkeepalive {
		mem.keepsalive()
	}
	if mem.onmanaged == true {

		return CudaMemCopy(ptr, mem, bsize, kind.Default())
	}
	if mem.onhost == true {
		return CudaMemCopy(ptr, mem, bsize, kind.HostToHost())
	}

	return CudaMemCopy(ptr, mem, bsize, kind.DeviceToHost())
}

//MemType is a typedefed C.cudaMemoryType
type MemType C.cudaMemoryType

//Ptr returns an unsafe.Pointer
func (mem *Malloced) Ptr() unsafe.Pointer {
	if mem == nil {
		return nil
	}
	return mem.ptr
}

//Stored returns the Location Flag of the memory
func (mem *Malloced) Stored() Location {
	if mem.Ptr() == nil {
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

//ByteSize returns the size of the memory chunck
func (mem *Malloced) ByteSize() SizeT {
	if mem == nil {
		return SizeT(0)
	}
	if mem.Ptr() == nil {
		return SizeT(0)
	}
	return mem.size
}

//Free Frees the memory on the device
func (mem *Malloced) Free() error {
	if mem.onhost == true {
		err := C.cudaFreeHost(mem.Ptr())
		mem.size = 0
		mem.typevalue = ""
		return newErrorRuntime("Free", err)
	}
	err := C.cudaFree(mem.Ptr())
	mem.ptr = nil
	mem.size = 0
	mem.typevalue = ""
	mem = nil
	return newErrorRuntime("Free", err)
}
func freecudamallocedmemory(mem *Malloced) error {
	return mem.Free()
}

func checkinterface(input interface{}) (string, error) {
	switch input.(type) {
	case []float32:
		return "[]float32", nil
	case []float64:
		return "[]float64", nil
	case []int:
		return "[]int", nil
	case []uint:
		return "[]uint", nil
	case []int32:
		return "[]int32", nil
	case []uint32:
		return "[]uint32", nil
	case []byte:
		return "[]byte", nil
	case []int8:
		return "[]int8", nil
	case []half.Float16:
		return "[]half.Float16", nil
	default:
		return "", errors.New("No Support")
	}

}

func tofloat64array(input interface{}) []float64 {
	switch x := input.(type) {
	case []float64:
		return x
	default:
		return nil
	}

}
func tofloat32array(input interface{}) []float32 {
	switch x := input.(type) {
	case []float32:
		return x
	default:
		return nil
	}

}
func tointarray(input interface{}) []int {
	switch x := input.(type) {
	case []int:
		return x
	default:
		return nil
	}

}
func touintarray(input interface{}) []uint {
	switch x := input.(type) {
	case []uint:
		return x
	default:
		return nil
	}

}
func toint32array(input interface{}) []int32 {
	switch x := input.(type) {
	case []int32:
		return x
	default:
		return nil
	}

}
func touint32array(input interface{}) []uint32 {
	switch x := input.(type) {
	case []uint32:
		return x
	default:
		return nil
	}

}
func tobytearray(input interface{}) []byte {
	switch x := input.(type) {
	case []byte:
		return x
	default:
		return nil
	}

}
func tohalfarray(input interface{}) []half.Float16 {
	switch x := input.(type) {
	case []half.Float16:
		return x

	default:
		return nil
	}
}
func toint8array(input interface{}) []int8 {
	switch x := input.(type) {
	case []int8:
		return x
	default:
		return nil
	}
}

//CudaMemCopy copies some memory from src to dest.  If default is selected and if the system supports unified virtual addressing then the transfer is inferred.
func CudaMemCopy(dest Memer, src Memer, count SizeT, kind MemcpyKind) error {
	err := C.cudaMemcpy(dest.Ptr(), src.Ptr(), count.c(), kind.c())
	if setkeepalive {
		keepsalivebuffer(dest, src)
	}
	return newErrorRuntime("cudaMemcpy", err)
}

//CudaMemCopyUnsafe takes unsafe Pointers and does the cuda mem copy
func CudaMemCopyUnsafe(dest unsafe.Pointer, src unsafe.Pointer, count SizeT, kind MemcpyKind) error {
	err := C.cudaMemcpy(dest, src, count.c(), kind.c())
	if setkeepalive {
		keepsalivebuffer(dest, src)
	}
	return newErrorRuntime("cudaMemcpy", err)
}

//UnifiedMemCopy does a mem copy based on the default
func UnifiedMemCopy(dest Memer, src Memer) error {
	if dest.ByteSize() != src.ByteSize() {
		return errors.New("Dest and Src not same size")
	}
	if setkeepalive {
		keepsalivebuffer(dest, src)
	}
	return CudaMemCopy(dest, src, dest.ByteSize(), MemcpyKind(C.cudaMemcpyDefault))
}

//UnifiedMemCopyUnsafe  does a memcopy but with unsafe pointers
func UnifiedMemCopyUnsafe(dest, src unsafe.Pointer, size SizeT) error {
	return CudaMemCopyUnsafe(dest, src, size, MemcpyKind(C.cudaMemcpyDefault))
}

//CudaMallocUnsafe returns an unsafe pointer that points to cuda gpu allocated mem.
func CudaMallocUnsafe(totalbytes SizeT) (unsafe.Pointer, error) {
	var gpumem unsafe.Pointer
	err := newErrorRuntime("Malloc", C.cudaMalloc(&gpumem, totalbytes.c()))
	if err != nil {
		return nil, err
	}
	err = newErrorRuntime("CudaMallocUnsafe-MemSet", C.cudaMemset(gpumem, C.int(0), totalbytes.c()))
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		runtime.SetFinalizer(gpumem, C.cudaFree)
	}
	return gpumem, nil
}

//Malloc returns struct Malloced that has a pointer memory that is now allocated to the device. Values are set to 0.
func Malloc(totalbytes SizeT) (gpumem *Malloced, err error) {
	var gpu Malloced

	gpu.size = totalbytes
	err = newErrorRuntime("Malloc", C.cudaMalloc(&gpu.ptr, gpu.size.c()))
	gpu.Set(0)
	gpumem = &gpu
	if setfinalizer {
		runtime.SetFinalizer(gpumem, freecudamallocedmemory)
	}

	return gpumem, err
}

//MallocHost - Allocates page-locked memory on the host. used specifically for fast calls from the host.
func MallocHost(totalbytes SizeT) (cudamem *Malloced, err error) {
	var mem Malloced
	mem.size = totalbytes

	err = newErrorRuntime("MallocHost", C.cudaMallocHost(&mem.ptr, mem.size.c()))
	if err != nil {
		return nil, err
	}
	mem.onhost = true
	mem.Set(0)
	cudamem = &mem
	if setfinalizer {
		runtime.SetFinalizer(cudamem, freecudamallocedmemory)
	}
	return cudamem, nil
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

//UnifiedMangedHost uses the Unified memory mangement system and starts it off in the host
func UnifiedMangedHost(size SizeT) (*Malloced, error) {
	return MallocManaged(size, ManagedMem(2))
}

//UnifiedMangedGlobal uses the Unified memory mangement system and starts it off in the Device
func UnifiedMangedGlobal(size SizeT) (*Malloced, error) {
	return MallocManaged(size, ManagedMem(1))
}

//MallocManaged is useful if devices support unified virtual memory.
func MallocManaged(size SizeT, management ManagedMem) (cudamem *Malloced, err error) {
	var mem Malloced
	mem.onmanaged = true

	mem.size = size
	err = newErrorRuntime("MallocManaged", C.cudaMallocManaged(&mem.ptr, size.c(), management.c()))
	cudamem = &mem
	if setfinalizer {
		runtime.SetFinalizer(cudamem, freecudamallocedmemory)
	}
	return cudamem, err
}
func prependerror(info string, err error) error {
	return errors.New(info + ": " + err.Error())
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
	case []CHalf:
		return SizeT(len(val) * 2), nil
	case []half.Float16:
		return SizeT(len(val) * 2), nil
	case int:
		return SizeT(4), nil
	case byte:
		return SizeT(1), nil
	case int8:
		return SizeT(1), nil
	case float64:
		return SizeT(8), nil
	case float32:
		return SizeT(4), nil
	case int32:
		return SizeT(4), nil
	case uint32:
		return SizeT(4), nil
	case half.Float16:
		return SizeT(2), nil
	case CHalf:
		return SizeT(2), nil
	default:
		return SizeT(0), errors.New("FindSizeT: Unsupported Type")
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

//MemCpyDeterminer is a helper function and has not been fully tested
func MemCpyDeterminer(src, dest Memer) (MemcpyKind, error) {
	var L LocationFlag
	var M MemcpyKindFlag
	if src.Stored() == L.NotAllocated() {
		return M.HostToHost(), errors.New("Source Memory Not Allocated")
	}
	switch dest.Stored() {

	case L.NotAllocated():
		return M.HostToHost(), errors.New("Destination Memory Not Allocated")

	case L.Unified():
		switch src.Stored() {
		case L.Unified():
			return M.Default(), nil
		case L.GoSideHost():
			return M.Default(), nil
		default:
			return M.Default(), errors.New("not Supported for gocudnn")
		}

	case L.GoSideHost():
		switch src.Stored() {
		case L.GoSideHost():
			return M.HostToHost(), errors.New("No Cuda Allocation Needed")
		case L.Device():
			return M.DeviceToHost(), nil

		case L.CudaHost():
			return M.HostToHost(), nil
		default:
			return M.HostToHost(), errors.New("not supported for gocudnn")
		}
	case L.Device():
		switch src.Stored() {
		case L.Device():
			return M.DeviceToDevice(), nil
		case L.GoSideHost():
			return M.HostToDevice(), nil
		case L.CudaHost():
			return M.HostToDevice(), nil
		default:
			return M.Default(), errors.New("not supported for gocudnn")

		}
	case L.CudaHost():
		switch src.Stored() {
		case L.CudaHost():
			return M.HostToHost(), nil
		case L.GoSideHost():
			return M.HostToHost(), nil
		case L.Device():
			return M.DeviceToHost(), nil
		default:
			return M.Default(), errors.New("not supported for gocudnn")

		}
	default:
		return M.HostToHost(), errors.New("not supported for gocudnn")
	}

}

//Readable makes the Location readable readable
func (l Location) Readable() string {
	switch l {
	case 0:
		return "no memory allocated"
	case 1:
		return "on Go Side Host"
	case 2:
		return "on Cuda Side Device"
	case 3:
		return "on Cuda Side Host"
	case 4:
		return "on Cuda Unified Memory"
	}
	return "UH OH"
}

//Location is used for flags.  It will be used for mem copies between host and device.
type Location int

//LocationFlag struct is nil and used to pass Location in a more readable format
type LocationFlag struct {
}

//NotAllocated would be that the pointer stored for Memer is pointing to nil
func (l LocationFlag) NotAllocated() Location {
	return Location(0)
}

//GoSideHost would be some sort of slice that is not in cuda's paged memory /domain.
func (l LocationFlag) GoSideHost() Location {
	return Location(1)
}

//Device would mean that the memory is on the device
func (l LocationFlag) Device() Location {
	return Location(2)
}

//CudaHost would be cuda paged memory on the host
func (l LocationFlag) CudaHost() Location {
	return Location(3)
}

//Unified would mean cuda virtual unified memory
func (l LocationFlag) Unified() Location {
	return Location(4)
}
