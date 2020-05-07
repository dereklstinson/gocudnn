package cudart

/*
#include <cuda_runtime_api.h>
*/
import "C"
import (
	"unsafe"

	"github.com/dereklstinson/cutil"
	"github.com/dereklstinson/gocudnn/gocu"
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

//Memcpy copies some memory from src to dest.  If default is selected and if the system supports unified virtual addressing then the transfer is inferred.
func Memcpy(dest, src cutil.Pointer, sizet uint, kind MemcpyKind) error {
	return status(C.cudaMemcpy(dest.Ptr(), src.Ptr(), C.size_t(sizet), kind.c())).error("cudaMemcpy")

}

//MemcpyUS will do a memcopy using unsafe pointers. It's a little lower level than the regular MemCpy
func MemcpyUS(dest, src unsafe.Pointer, sizet uint, kind MemcpyKind) error {
	return status(C.cudaMemcpy(dest, src, C.size_t(sizet), kind.c())).error("MemcpyUS")

}

//MemcpyAsync Copies data between host and device.
func MemcpyAsync(dest, src cutil.Pointer, sizet uint, kind MemcpyKind, stream gocu.Streamer) error {
	return status(C.cudaMemcpyAsync(dest.Ptr(), src.Ptr(), C.size_t(sizet), kind.c(), C.cudaStream_t(stream.Ptr()))).error("MemcpyAsync")
}

//Memcpy2D copies some memory from src to dest.  If default is selected and if the system supports unified virtual addressing then the transfer is inferred.
func Memcpy2D(dest cutil.Pointer, dpitch uint, src cutil.Pointer, spitch uint, width, height uint, kind MemcpyKind) error {
	err := C.cudaMemcpy2D(dest.Ptr(), C.size_t(dpitch), src.Ptr(), C.size_t(spitch), C.size_t(width), C.size_t(height), kind.c())

	return newErrorRuntime("Memcpy2D", err)
}

//MemcpyPeer Copies memory between two devices.
func MemcpyPeer(dest cutil.Pointer, ddev Device, src cutil.Pointer, sdev Device, sizet uint) error {
	err := C.cudaMemcpyPeer(dest.Ptr(), ddev.c(), src.Ptr(), sdev.c(), C.size_t(sizet))
	return newErrorRuntime("cudaMemcpyPeer", err)
}

//MemcpyPeerAsync copies memory between two devices async.
func MemcpyPeerAsync(dest cutil.Pointer, ddev Device, src cutil.Pointer, sdev Device, sizet uint, stream gocu.Streamer) error {
	err := C.cudaMemcpyPeerAsync(dest.Ptr(), ddev.c(), src.Ptr(), sdev.c(), C.size_t(sizet), (C.cudaStream_t)(stream.Ptr()))
	return newErrorRuntime("MemcpyPeerAsync", err)
}

//Memcpy3DParams is used for Memcpy3d
type Memcpy3DParams C.struct_cudaMemcpy3DParms

func (m *Memcpy3DParams) cptr() *C.struct_cudaMemcpy3DParms {
	return (*C.struct_cudaMemcpy3DParms)(m)
}

//CreateMemcpy3DParams srcpp and destpp are optional and can be zero
func CreateMemcpy3DParams(srcArray *Array, srcPos Pos, srcPtr PitchedPtr, dstArray *Array, dstPos Pos, dstPtr PitchedPtr, ext Extent, kind MemcpyKind) (m *Memcpy3DParams) {
	m = new(Memcpy3DParams)
	m.srcArray = srcArray.c
	m.srcPos = srcPos.c()
	m.srcPtr = srcPtr.c()
	m.dstArray = dstArray.c
	m.dstPos = dstPos.c()
	m.dstPtr = dstPtr.c()
	m.extent = ext.c()
	m.kind = kind.c()
	return m
}

//Memcpy3D -Copies a matrix (height rows of width bytes each) from the memory area pointed
//to by src to the CUDA array dst starting at the upper left corner (wOffset, hOffset)
// where kind specifies the direction of the copy.  m is created using CreateMemcpy3DParams
func Memcpy3D(m *Memcpy3DParams) error {
	return newErrorRuntime("Memcpy3D()", C.cudaMemcpy3D(m.cptr()))
}

//Memcpy3DAsync -Copies a matrix (height rows of width bytes each) from the memory area pointed
//to by src to the CUDA array dst starting at the upper left corner (wOffset, hOffset)
// where kind specifies the direction of the copy.  m is created using CreateMemcpy3DParams
func Memcpy3DAsync(m *Memcpy3DParams, s gocu.Streamer) error {

	return newErrorRuntime("Memcpy3DAsync()", C.cudaMemcpy3DAsync(m.cptr(), ExternalWrapper(s.Ptr()).c()))
}

/*
func FillSliceFromManaged(mem gocu.Mem, slice interface{}) error {

}
func FillSliceFromHost(mem gocu.Mem, slice interface{}) error {

}
func FillSliceFromDevice(mem gocu.Mem, slice interface{}) error {

}
*/
