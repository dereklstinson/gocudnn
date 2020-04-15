package nvjpeg

/*
#include <nvjpeg.h>
#include<cuda_runtime_api.h>

int dev_malloc_unified(void **p,size_t s) {
	return (int)cudaMallocManaged(p,s,cudaMemAttachGlobal);
}

int dev_free(void *p) { return (int)cudaFree(p); }

int host_malloc_unified(void **p,size_t s,unsigned int flags){
	return (int)cudaMallocManaged(p,s,cudaMemAttachHost);
}

int host_free_unified(void *p) { return (int)cudaFree(p); }



void fillhostvales_unified(nvjpegPinnedAllocator_t *devalloc){
	devalloc->pinned_malloc=&host_malloc_unified;
	devalloc->pinned_free=&host_free_unified;

}
void filldevvales_unified(nvjpegDevAllocator_t *devalloc){
	devalloc->dev_malloc=&dev_malloc_unified;
	devalloc->dev_free=&dev_free;

}
//regular HOST and Device allocators
int host_free_regular(void *p){return (int)cudaFreeHost(p);}

int dev_malloc_regular(void **p,size_t s){
	return (int)cudaMalloc(p,s);
}
int host_malloc_regular(void **p,size_t s,unsigned int flags){
	return (int)cudaMallocHost(p,s);
}

void filldevvales_regular(nvjpegDevAllocator_t *devalloc){
	devalloc->dev_malloc=&dev_malloc_regular;
	devalloc->dev_free=&dev_free;

}
void fillhostvales_regular(nvjpegPinnedAllocator_t *devalloc){
	devalloc->pinned_malloc=&host_malloc_regular;
	devalloc->pinned_free=&host_free_regular;
}
*/
import "C"

import (
	"runtime"
)

//CreateEx uses cudaMallocManaged. The handle is used for all consecutive nvjpeg calls
// IN         backend       : Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.
// INT/OUT    handle        : Codec instance, use for other calls
func CreateEx(backend Backend) (h *Handle, err error) {
	h = new(Handle)
	h.exdevallocator = new(C.nvjpegDevAllocator_t)
	h.expinnedallocator = new(C.nvjpegPinnedAllocator_t)
	C.filldevvales_unified(h.exdevallocator)
	C.fillhostvales_unified(h.expinnedallocator)
	//deval.dev_malloc = C.dev_malloc
	//deval.dev_free = C.dev_free
	err = status(C.nvjpegCreateEx(backend.c(), h.exdevallocator, h.expinnedallocator, 0, &h.h)).error()
	return h, err
}

//PinnedBuffer buffer for pinned host memory
type PinnedBuffer struct {
	b         C.nvjpegBufferPinned_t
	allocator *C.nvjpegPinnedAllocator_t
}

func CreateUnifiedPinnedBuffer(h *Handle) (pbuff *PinnedBuffer, err error) {
	pbuff = new(PinnedBuffer)
	pbuff.allocator = new(C.nvjpegPinnedAllocator_t)
	C.fillhostvales_unified(pbuff.allocator)
	err = status(C.nvjpegBufferPinnedCreate(h.h, pbuff.allocator, &pbuff.b)).error()
	runtime.SetFinalizer(pbuff, nvjpegBufferPinnedDestroy)
	return pbuff, err
}
func CreateNonUnifiedPinnedBuffer(h *Handle) (pbuff *PinnedBuffer, err error) {
	pbuff = new(PinnedBuffer)
	pbuff.allocator = new(C.nvjpegPinnedAllocator_t)
	C.fillhostvales_regular(pbuff.allocator)
	err = status(C.nvjpegBufferPinnedCreate(h.h, pbuff.allocator, &pbuff.b)).error()
	runtime.SetFinalizer(pbuff, nvjpegBufferPinnedDestroy)
	return pbuff, err
}
func nvjpegBufferPinnedDestroy(pbuff *PinnedBuffer) (err error) {
	return status(C.nvjpegBufferPinnedDestroy(pbuff.b)).error()
}

//DeviceBuffer buffer for device memory
type DeviceBuffer struct {
	b         C.nvjpegBufferDevice_t
	allocator *C.nvjpegDevAllocator_t
}

func CreateUnifiedDeviceBuffer(h *Handle) (dbuff *DeviceBuffer, err error) {
	dbuff = new(DeviceBuffer)
	dbuff.allocator = new(C.nvjpegDevAllocator_t)
	C.filldevvales_unified(dbuff.allocator)
	err = status(C.nvjpegBufferDeviceCreate(h.h, dbuff.allocator, &dbuff.b)).error()
	runtime.SetFinalizer(dbuff, nvjpegBufferDeviceDestroy)
	return dbuff, err
}
func CreateNonUnifiedDeviceBuffer(h *Handle) (dbuff *DeviceBuffer, err error) {
	dbuff = new(DeviceBuffer)
	dbuff.allocator = new(C.nvjpegDevAllocator_t)
	C.filldevvales_regular(dbuff.allocator)
	err = status(C.nvjpegBufferDeviceCreate(h.h, dbuff.allocator, &dbuff.b)).error()
	runtime.SetFinalizer(dbuff, nvjpegBufferDeviceDestroy)
	return dbuff, err
}
func nvjpegBufferDeviceDestroy(dbuff *DeviceBuffer) (err error) {
	return status(C.nvjpegBufferDeviceDestroy(dbuff.b)).error()
}

/*

These are not working on one of my machines.  For now I will comment them out


func (p *PinnedBuffer) Retrieve() (ptr cutil.Mem, sib uint, err error) {
	var sizet C.size_t
	ptr = new(gocu.CudaPtr)
	err = status(C.nvjpegBufferPinnedRetrieve(p.b, &sizet, ptr.DPtr())).error()
	return ptr, (uint)(sizet), err

}
func (d *DeviceBuffer) Retrieve() (ptr cutil.Mem, sib uint, err error) {
	var sizet C.size_t
	ptr = new(gocu.CudaPtr)
	err = status(C.nvjpegBufferDeviceRetrieve(d.b, &sizet, ptr.DPtr())).error()
	return ptr, (uint)(sizet), err
}

*/
//Backend are flags that are used to set the implimentation.
type Backend C.nvjpegBackend_t

func (b Backend) c() C.nvjpegBackend_t {
	return C.nvjpegBackend_t(b)
}

//Default defaults to GPUHybrid ---  Backend(C.NVJPEG_BACKEND_DEFAULT)  //binder note. I don't know why this is here.
//Method changes the underlying value of the flag and also returns that value.
func (b *Backend) Default() Backend {
	*b = Backend(C.NVJPEG_BACKEND_DEFAULT)
	return *b
}

//Hybrid uses CPU for Huffman decode ---  Backend(C.NVJPEG_BACKEND_HYBRID)
//Method changes the underlying value of the flag and also returns that value.
func (b *Backend) Hybrid() Backend {
	*b = Backend(C.NVJPEG_BACKEND_HYBRID)
	return *b
}

//GPUHybrid nvjpegDecodeBatched will use GPU decoding for baseline JPEG images
//with interleaved scan when batch size is bigger than 100,  it uses CPU for other JPEG types.
//Other decode APIs will continue to use CPU for Huffman decode
//Method changes the underlying value of the flag and also returns that value.
func (b *Backend) GPUHybrid() Backend {
	*b = Backend(C.NVJPEG_BACKEND_GPU_HYBRID)
	return *b
}
