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
	"errors"
	"fmt"
	"strconv"
	"unsafe"
)

//Memer is an interface for memory
type Memer interface {
	Ptr() unsafe.Pointer
	ByteSize() SizeT
	Free() error
	Stored() Location
	FillSlice(interface{}) error
	IsMalloced() *Malloced
	IsGoPtr() *GoPointer

	//Atributes()Atribs
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

//OffSet will return the offset address from the pointer passed
func OffSet(point unsafe.Pointer, unitsize int, offset int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(point) + uintptr(unitsize*offset))
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

//Is Malloced will return the malloced
func (mem *Malloced) IsMalloced() *Malloced {
	return mem
}

//IsGoPtr will return nil
func (mem *Malloced) IsGoPtr() *GoPointer {
	return nil
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
	return newErrorRuntime("Free", err)
}

//GoPointer holds a pointer to a slice
type GoPointer struct {
	ptr       unsafe.Pointer
	slice     interface{}
	size      SizeT
	typevalue string
	array     bool
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
	default:
		return "", errors.New("No Support")
	}

}

//FillSlice will fill the slice, but if the GoPointer has been coppied to device memory. Then the memory will not be up to date.
func (mem *GoPointer) FillSlice(input interface{}) error {
	bsizein, err := FindSizeT(input)
	if err != nil {
		return err
	}
	if bsizein != mem.ByteSize() {
		return errors.New("FillSlice: Sizes Don't Match " + strconv.Itoa(int(bsizein)) + " and " + strconv.Itoa(int(mem.ByteSize())))
	}
	inputtype, err := checkinterface(input)
	if err != nil {
		return err
	}
	memtype, err := checkinterface(mem.slice)
	if err != nil {
		return err
	}
	if inputtype != memtype {
		return errors.New("Fill SLice: Types Don't Match")

	}

	switch x := input.(type) {
	case []float32:
		y := tofloat32array(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []int32:
		y := toint32array(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []int:
		y := tointarray(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []float64:
		y := tofloat64array(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []uint32:
		y := touint32array(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []uint:
		y := touintarray(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []byte:
		y := tobytearray(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	}

	return nil
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
func toint8array(input interface{}) []int8 {
	switch x := input.(type) {
	case []int8:
		return x
	default:
		return nil
	}
}

//ByteSize returns the size of the memory chunk
func (mem *GoPointer) ByteSize() SizeT {
	if mem.ptr == nil {
		return SizeT(0)
	}
	if mem == nil {
		return SizeT(0)
	}
	return mem.size
}

//Ptr returns an unsafe.Pointer
func (mem *GoPointer) Ptr() unsafe.Pointer {
	if mem == nil {
		return nil
	}
	return mem.ptr
}

//Is Malloced will return nil.
func (mem *GoPointer) IsMalloced() *Malloced {
	return nil
}

//IsGoPtr will return the go pointer
func (mem *GoPointer) IsGoPtr() *GoPointer {
	return mem
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

//MakeGoPointer takes a slice and gives a GoPointer for that slice.  I wouldn't use that slice anylonger
func MakeGoPointer(input interface{}) (*GoPointer, error) {
	//fname:="MakeGoPointer"
	var ptr GoPointer
	ptr.slice = input
	var err error
	switch val := input.(type) {
	case []int:

		ptr.ptr = unsafe.Pointer(&val[0])

		ptr.typevalue = "int"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case []int8:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "int8"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}
		return &ptr, nil

	case []byte:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "byte"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case []float64:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "float64"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case []float32:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "float32"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case []int32:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "int32"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case []uint32:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "uint32"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}
		return &ptr, nil

	case int:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "int"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case int8:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "int8"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil

	case byte:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "byte"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case float64:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "float64"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case float32:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "float32"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case int32:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "int32"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case uint32:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "uint32"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case CInt:

		ptr.ptr = val.CPtr()
		ptr.typevalue = "CInt"
		ptr.size = SizeT(val.Bytes())
		return &ptr, nil
	case CDouble:

		ptr.ptr = val.CPtr()
		ptr.typevalue = "CDouble"
		ptr.size = SizeT(val.Bytes())
		return &ptr, nil
	case CFloat:

		ptr.ptr = val.CPtr()
		ptr.typevalue = "CFloat"
		ptr.size = SizeT(val.Bytes())
		return &ptr, nil
	case CUInt:

		ptr.ptr = val.CPtr()
		ptr.typevalue = "CUInt"
		ptr.size = SizeT(val.Bytes())
		return &ptr, nil
	default:
		thetype := fmt.Errorf("Type %T", val)
		return nil, errors.New("MakeGoPointer: Unsupported Type -- Type: " + thetype.Error())
	}
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

//Malloc returns struct Malloced that has a pointer memory that is now allocated to the device
func Malloc(totalbytes SizeT) (*Malloced, error) {
	var gpu Malloced
	gpu.ptr = unsafe.Pointer(&gpu.devptr)
	gpu.size = totalbytes
	err := C.cudaMalloc(&gpu.ptr, gpu.size.c())
	gpu.Set(0)
	return &gpu, newErrorRuntime("Malloc", err)
}

//MallocHost - Allocates page-locked memory on the host. used specifically for fast calls from the host.
func MallocHost(totalbytes SizeT) (*Malloced, error) {
	var mem Malloced
	mem.size = totalbytes
	x := C.cudaMallocHost(&mem.ptr, mem.size.c())
	err := newErrorRuntime("MallocHost", x)
	if err != nil {
		return nil, err
	}
	mem.onhost = true
	mem.Set(0)
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
