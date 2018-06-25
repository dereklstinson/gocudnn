package gocudnn

/*
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <driver_types.h>
*/
import "C"

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
