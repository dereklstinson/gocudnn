package cudart

/*
#include <cuda_runtime_api.h>
*/
import "C"
import (
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//MemcpyKind are enum flags for mem copy can be passed using methdos
type MemcpyKind C.enum_cudaMemcpyKind

//HostToHost return MemcpyKind(C.cudaMemcpyHostToHost )
func (m MemcpyKind) HostToHost() MemcpyKind {
	return MemcpyKind(C.cudaMemcpyHostToHost)
}

//HostToDevice 	return MemcpyKind(C.cudaMemcpyHostToDevice )
func (m MemcpyKind) HostToDevice() MemcpyKind {
	return MemcpyKind(C.cudaMemcpyHostToDevice)
}

//DeviceToHost return MemcpyKind(C.cudaMemcpyDeviceToHost )
func (m MemcpyKind) DeviceToHost() MemcpyKind {
	return MemcpyKind(C.cudaMemcpyDeviceToHost)
}

//DeviceToDevice return MemcpyKind(C.cudaMemcpyDeviceToDevice )
func (m MemcpyKind) DeviceToDevice() MemcpyKind {
	return MemcpyKind(C.cudaMemcpyDeviceToDevice)
}

//Default return MemcpyKind(C.cudaMemcpyDefault )
func (m MemcpyKind) Default() MemcpyKind {
	return MemcpyKind(C.cudaMemcpyDefault)
}

func (m MemcpyKind) c() C.enum_cudaMemcpyKind { return C.enum_cudaMemcpyKind(m) }

//MemCpy copies some memory from src to dest.  If default is selected and if the system supports unified virtual addressing then the transfer is inferred.
func MemCpy(dest gocu.Mem, src gocu.Mem, sizet uint, kind MemcpyKind) error {
	err := C.cudaMemcpy(dest.Ptr(), src.Ptr(), C.size_t(sizet), kind.c())

	return newErrorRuntime("cudaMemcpy", err)
}

//MemCpyUS will do a memcopy using unsafe pointers. It's a little lower level than the regular MemCpy
func MemcpyUS(dest, src unsafe.Pointer, sizet uint, kind MemcpyKind) error {
	err := C.cudaMemcpy(dest, src, C.size_t(sizet), kind.c())

	return newErrorRuntime("cudaMemcpy-MemcpyUnsafe", err)
}

//MemcpyAsync Copies data between host and device.
func MemcpyAsync(dest gocu.Mem, src gocu.Mem, sizet uint, kind MemcpyKind, stream gocu.Streamer) error {
	err := C.cudaMemcpyAsync(dest.Ptr(), src.Ptr(), C.size_t(sizet), kind.c(), C.cudaStream_t(stream.Ptr()))

	return newErrorRuntime("cudaMemcpy", err)
}

//Memcpy2D copies some memory from src to dest.  If default is selected and if the system supports unified virtual addressing then the transfer is inferred.
func Memcpy2D(dest gocu.Mem, dpitch uint, src gocu.Mem, spitch uint, width, height uint, kind MemcpyKind) error {
	err := C.cudaMemcpy2D(dest.Ptr(), C.size_t(dpitch), src.Ptr(), C.size_t(spitch), C.size_t(width), C.size_t(height), kind.c())

	return newErrorRuntime("cudaMemcpy", err)
}

//MemcpyPeer Copies memory between two devices.
func MemcpyPeer(dest gocu.Mem, ddev Device, src gocu.Mem, sdev Device, sizet uint) error {
	err := C.cudaMemcpyPeer(dest.Ptr(), ddev.c(), src.Ptr(), sdev.c(), C.size_t(sizet))
	return newErrorRuntime("cudaMemcpyPeer", err)
}

/*
func FillSliceFromManaged(mem gocu.Mem, slice interface{}) error {

}
func FillSliceFromHost(mem gocu.Mem, slice interface{}) error {

}
func FillSliceFromDevice(mem gocu.Mem, slice interface{}) error {

}
*/
