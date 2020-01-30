package cudart

/*
#include <cuda_runtime_api.h>
*/
import "C"
import (
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/cutil"
)

//MemcpyKind are enum flags for mem copy can be passed using methdos
type MemcpyKind C.enum_cudaMemcpyKind

//HostToHost return MemcpyKind(C.cudaMemcpyHostToHost )
func (m *MemcpyKind) HostToHost() MemcpyKind {
	*m = MemcpyKind(C.cudaMemcpyHostToHost)
	return *m
}

//HostToDevice 	return MemcpyKind(C.cudaMemcpyHostToDevice )
func (m *MemcpyKind) HostToDevice() MemcpyKind {
	*m = MemcpyKind(C.cudaMemcpyHostToDevice)
	return *m
}

//DeviceToHost return MemcpyKind(C.cudaMemcpyDeviceToHost )
func (m *MemcpyKind) DeviceToHost() MemcpyKind {
	*m = MemcpyKind(C.cudaMemcpyDeviceToHost)
	return *m
}

//DeviceToDevice return MemcpyKind(C.cudaMemcpyDeviceToDevice )
func (m *MemcpyKind) DeviceToDevice() MemcpyKind {
	*m = MemcpyKind(C.cudaMemcpyDeviceToDevice)
	return *m
}

//Default return MemcpyKind(C.cudaMemcpyDefault )
func (m *MemcpyKind) Default() MemcpyKind {
	*m = MemcpyKind(C.cudaMemcpyDefault)
	return *m
}

func (m MemcpyKind) c() C.enum_cudaMemcpyKind { return C.enum_cudaMemcpyKind(m) }

//MemCpy copies some memory from src to dest.  If default is selected and if the system supports unified virtual addressing then the transfer is inferred.
func MemCpy(dest, src cutil.Pointer, sizet uint, kind MemcpyKind) error {
	err := C.cudaMemcpy(dest.Ptr(), src.Ptr(), C.size_t(sizet), kind.c())

	return newErrorRuntime("cudaMemcpy", err)
}

//MemcpyUS will do a memcopy using unsafe pointers. It's a little lower level than the regular MemCpy
func MemcpyUS(dest, src unsafe.Pointer, sizet uint, kind MemcpyKind) error {
	err := C.cudaMemcpy(dest, src, C.size_t(sizet), kind.c())

	return newErrorRuntime("cudaMemcpy-MemcpyUnsafe", err)
}

//MemcpyAsync Copies data between host and device.
func MemcpyAsync(dest, src cutil.Pointer, sizet uint, kind MemcpyKind, stream gocu.Streamer) error {
	err := C.cudaMemcpyAsync(dest.Ptr(), src.Ptr(), C.size_t(sizet), kind.c(), C.cudaStream_t(stream.Ptr()))

	return newErrorRuntime("cudaMemcpy", err)
}

//Memcpy2D copies some memory from src to dest.  If default is selected and if the system supports unified virtual addressing then the transfer is inferred.
func Memcpy2D(dest cutil.Pointer, dpitch uint, src cutil.Pointer, spitch uint, width, height uint, kind MemcpyKind) error {
	err := C.cudaMemcpy2D(dest.Ptr(), C.size_t(dpitch), src.Ptr(), C.size_t(spitch), C.size_t(width), C.size_t(height), kind.c())

	return newErrorRuntime("cudaMemcpy", err)
}

//MemcpyPeer Copies memory between two devices.
func MemcpyPeer(dest cutil.Pointer, ddev Device, src cutil.Pointer, sdev Device, sizet uint) error {
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
