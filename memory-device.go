package gocudnn

/*
#include <cuda_runtime_api.h>
#include <cuda.h>
*/
import "C"

import (
	"errors"
	"unsafe"
)

//DMalloced is a non garbage collection varient to what was used in buffer.
//It gives the user more control over the memory for cases of building neural networks
//on the GPU.  Just remember to free the memory or the gpu will fill up fast.
//DMalloced contains info on a chunk of memory on the device
type DMalloced struct {
	ptr        unsafe.Pointer
	sizeintype int
	typevalue  string
}

//SizeinBytes returns the size of the memory chunck
func (mem *DMalloced) SizeinBytes() int {
	return bytesize(mem.typevalue) * mem.sizeintype
}

//SizeinType returns the size of memory held by this struct by type
func (mem *DMalloced) SizeinType() int {
	return mem.sizeintype
}

//Ptr returns an unsafe.Pointer
func (mem *DMalloced) Ptr() unsafe.Pointer {
	return mem.ptr
}

//CudaMemCopyHostToDevice takes slice and applies it to the device returns an error
func (mem *DMalloced) CudaMemCopyHostToDevice(val interface{}) error {
	var res C.cudaError_t
	switch val := val.(type) {
	case []byte:
		if len(val) != mem.SizeinType() {
			return errors.New("Host: Memory Sizes Don't Match")
		}
		res = C.cudaMemcpy(mem.ptr, unsafe.Pointer(&val[0]), C.size_t(mem.SizeinBytes()),
			C.cudaMemcpyHostToDevice)
	case []float64:
		if len(val) != mem.SizeinType() {
			return errors.New("Host: Memory Sizes Don't Match")
		}
		res = C.cudaMemcpy(mem.ptr, unsafe.Pointer(&val[0]), C.size_t(mem.SizeinBytes()),
			C.cudaMemcpyHostToDevice)
	case []float32:
		if len(val) != mem.SizeinType() {
			return errors.New("Host: Memory Sizes Don't Match")
		}
		res = C.cudaMemcpy(mem.ptr, unsafe.Pointer(&val[0]), C.size_t(mem.SizeinBytes()),
			C.cudaMemcpyHostToDevice)
	case []int:
		if len(val) != mem.SizeinType() {
			return errors.New("Host: Memory Sizes Don't Match")
		}
		res = C.cudaMemcpy(mem.ptr, unsafe.Pointer(&val[0]), C.size_t(mem.SizeinBytes()),
			C.cudaMemcpyHostToDevice)

	case []uint32:
		if len(val) != mem.SizeinType() {
			return errors.New("Host: Memory Sizes Don't Match")
		}
		res = C.cudaMemcpy(mem.ptr, unsafe.Pointer(&val[0]), C.size_t(mem.SizeinBytes()),
			C.cudaMemcpyHostToDevice)

	}

	return newErrorRuntime("cudaMemcpy", res)
}

//CloneCudaMem clones a block of memory on the gpu and returns the pointer.
func (mem *DMalloced) CloneCudaMem() (DMalloced, error) {
	var res C.cudaError_t
	newgpumem := Malloc(mem.SizeinBytes(), mem.typevalue)
	res = C.cudaMemcpy(newgpumem.ptr, mem.ptr, C.size_t(mem.SizeinBytes()), C.cudaMemcpyDeviceToDevice)
	return newgpumem, newErrorRuntime("cudaMemcpyDtoD", res)
}

//CudaMemCopyDeviceToHost takes slice and applies it to the device returns an error
func (mem *DMalloced) CudaMemCopyDeviceToHost(val interface{}) error {
	var res C.cudaError_t
	switch val := val.(type) {
	case []byte:
		if len(val) != mem.SizeinType() {
			return errors.New("Host: Memory Sizes Don't Match")
		}
		res = C.cudaMemcpy(unsafe.Pointer(&val[0]), mem.ptr, C.size_t(mem.SizeinBytes()),
			C.cudaMemcpyDeviceToHost)

	case []float64:
		if len(val) != mem.SizeinType() {
			return errors.New("Host: Memory Sizes Don't Match")
		}
		res = C.cudaMemcpy(unsafe.Pointer(&val[0]), mem.ptr, C.size_t(mem.SizeinBytes()),
			C.cudaMemcpyDeviceToHost)
	case []float32:

		if len(val) != mem.SizeinType() {
			return errors.New("Host: Memory Sizes Don't Match")
		}
		res = C.cudaMemcpy(unsafe.Pointer(&val[0]), mem.ptr, C.size_t(mem.SizeinBytes()),
			C.cudaMemcpyDeviceToHost)
	case []int:
		if len(val) != mem.SizeinType() {
			return errors.New("Host: Memory Sizes Don't Match")
		}
		res = C.cudaMemcpy(unsafe.Pointer(&val[0]), mem.ptr, C.size_t(mem.SizeinBytes()),
			C.cudaMemcpyDeviceToHost)
	case []uint32:
		if len(val) != mem.SizeinType() {
			return errors.New("Host: Memory Sizes Don't Match")
		}
		res = C.cudaMemcpy(unsafe.Pointer(&val[0]), mem.ptr, C.size_t(mem.SizeinBytes()),
			C.cudaMemcpyDeviceToHost)
	}

	return newErrorRuntime("cudaMemcpy", res)
}

//CudaFree Frees the memory on the device
func (mem *DMalloced) CudaFree() {
	C.cudaFree(mem.ptr)
	mem.sizeintype = 0
	mem.typevalue = ""
}

//Malloc returns struct DeviceMem which holds device memory info.
func Malloc(size int, typevalue string) DMalloced {
	var gpu DMalloced
	var some unsafe.Pointer
	gpu.ptr = some
	gpu.typevalue = typevalue
	gpu.sizeintype = size
	inbytes := bytesize(typevalue)
	C.cudaMalloc(&gpu.ptr, C.size_t(size*inbytes))
	return gpu
}

//LoadToDevice an array to the device and returns a DMalloced or device malloc
func LoadToDevice(val interface{}) (DMalloced, error) {

	switch val := val.(type) {
	case []byte:
		x := Malloc(len(val), "byte")
		err := x.CudaMemCopyHostToDevice(val)
		return x, err
	case []float64:
		x := Malloc(len(val), "float64")
		err := x.CudaMemCopyHostToDevice(val)
		return x, err
	case []float32:
		x := Malloc(len(val), "float32")
		err := x.CudaMemCopyHostToDevice(val)
		return x, err
	case []int:
		x := Malloc(len(val), "int")
		err := x.CudaMemCopyHostToDevice(val)
		return x, err
	case []uint32:
		x := Malloc(len(val), "uint32")
		err := x.CudaMemCopyHostToDevice(val)
		return x, err
	default:
		return DMalloced{}, errors.New("Unsupported Type")
	}

}

func bytesize(typevalue string) int {
	switch typevalue {
	case "float32":
		return 4
	case "int":
		return 4
	case "float64":
		return 8
	case "byte":
		return 1
	default:
		panic("wrong format picked")

	}

}
