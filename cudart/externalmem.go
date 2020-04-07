package cudart

/*

#include <cuda_runtime_api.h>
#include <cuda.h>
//#include <driver_types.h>

    typedef enum cudaExternalMemoryHandleType_enum {
         //         cudaExternalMemoryHandleTypeOpaqueFd         = 1,
         //         cudaExternalMemoryHandleTypeOpaqueWin32      = 2,
         //         cudaExternalMemoryHandleTypeOpaqueWin32Kmt   = 3,
         //         cudaExternalMemoryHandleTypeD3D12Heap        = 4,
         //         cudaExternalMemoryHandleTypeD3D12Resource    = 5,
                  cudaExternalMemoryHandleTypeD3D11Resource    = 6,
                  cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
                  cudaExternalMemoryHandleTypeNvSciBuf         = 8
              } cudaExternalMemoryHandleType;
*/
import "C"

/*
//ExternalMemory is a handle to an external memory object
type ExternalMemory struct {
	x C.cudaExternalMemory_t
}

type ExternalMemoryHandleDesc C.struct_cudaExternalMemoryHandleDesc

func CreateExternalMemoryHandleDesc() (e *ExternalMemoryHandleDesc) {
	e = new(ExternalMemoryHandleDesc)
	return e

}
func (e *ExternalMemoryHandleDesc)Set(eMemType ExternalMemoryHandleType,)
//func ImportExternalMemory()
//C.cudaImportExternalMemory(cudaExternalMemory_t *extMem_out, const struct cudaExternalMemoryHandleDesc *memHandleDesc);
//C.cudaExternalMemoryGetMappedBuffer(void **devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc *bufferDesc);
//C.cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc);
//C.cudaDestroyExternalMemory(cudaExternalMemory_t extMem);
//C.cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc);
//C.cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));
//C.cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));
//C.cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem);

//ExternalMemoryHandleType is the type of memory the handle has
type ExternalMemoryHandleType C.enum_cudaExternalMemoryHandleType

func (e ExternalMemoryHandleType) c() C.enum_cudaExternalMemoryHandleType {
	return (C.enum_cudaExternalMemoryHandleType)(e)
}

//OpaqueFileDescriptor - Handle is an opaque file descriptor
func (e *ExternalMemoryHandleType) OpaqueFileDescriptor() ExternalMemoryHandleType {
	*e = ExternalMemoryHandleType(C.cudaExternalMemoryHandleTypeOpaqueFd)
	return *e
}

//OpaqueWin32 Handle is an opaque shared NT handle
func (e *ExternalMemoryHandleType) OpaqueWin32() ExternalMemoryHandleType {
	*e = ExternalMemoryHandleType(C.cudaExternalMemoryHandleTypeOpaqueWin32)
	return *e
}

//OpaqueWin32Kmt - Handle is an opaque, globally shared handle
func (e *ExternalMemoryHandleType) OpaqueWin32Kmt() ExternalMemoryHandleType {
	*e = ExternalMemoryHandleType(C.cudaExternalMemoryHandleTypeOpaqueWin32Kmt)
	return *e
}

//D3D12Heap - Handle is a D3D12 heap object
func (e *ExternalMemoryHandleType) D3D12Heap() ExternalMemoryHandleType {
	*e = ExternalMemoryHandleType(C.cudaExternalMemoryHandleTypeD3D12Heap)
	return *e
}

//D3D12Resource -  Handle is a D3D12 committed resource
func (e *ExternalMemoryHandleType) D3D12Resource() ExternalMemoryHandleType {
	*e = ExternalMemoryHandleType(C.cudaExternalMemoryHandleTypeD3D12Resource)
	return *e
}

//D3D11Resource  Handle is a D3D11 committed resource
func (e *ExternalMemoryHandleType) D3D11Resource() ExternalMemoryHandleType {
	*e = ExternalMemoryHandleType(C.cudaExternalMemoryHandleTypeD3D11Resource)
	return *e
}

//D3D11ResourceKmt - Handle is oquaue globally shared
func (e *ExternalMemoryHandleType) D3D11ResourceKmt() ExternalMemoryHandleType {
	*e = ExternalMemoryHandleType(C.cudaExternalMemoryHandleTypeD3D11ResourceKmt)
	return *e
}

//NvSciBuf - I don't know what this is
func (e *ExternalMemoryHandleType) NvSciBuf() ExternalMemoryHandleType {
	*e = ExternalMemoryHandleType(C.cudaExternalMemoryHandleTypeNvSciBuf)
	return *e
}
*/
