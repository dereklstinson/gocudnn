package cudart

/*

#include <cuda_runtime_api.h>
#include <cuda.h>
*/
import "C"
import (
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/cutil"
)

//Device is a struct that holds a device info.
type Device C.int

func (d Device) c() C.int {
	return (C.int)(d)
}
func (d *Device) cptr() *C.int {
	return (*C.int)(d)
}

//DeviceSync Blocks until the device has completed all preceding requested tasks.
//DeviceSync() returns an error if one of the preceding tasks has failed.
//If the cudaDeviceScheduleBlockingSync flag was set for this device,
// the host thread will block until the device has finished its work.
//Will Set Device
func (d Device) DeviceSync() error {
	err := d.Set()
	if err != nil {
		return err
	}
	return newErrorRuntime("cudaDeviceSynchronize: ", C.cudaDeviceSynchronize())
}

//MemPrefetchAsync - Prefetches memory to the specified destination device.
//
//From Cuda Documentation:
//
//Prefetches memory to the specified destination device. devPtr is the base device pointer of the memory to be prefetched and dstDevice is the destination device. count specifies the number of bytes to copy. stream is the stream in which the operation is enqueued. The memory range must refer to managed memory allocated via cudaMallocManaged or declared via __managed__ variables.
//
//Passing in cudaCpuDeviceId for dstDevice will prefetch the data to host memory. If dstDevice is a GPU, then the device attribute cudaDevAttrConcurrentManagedAccess must be non-zero. Additionally, stream must be associated with a device that has a non-zero value for the device attribute cudaDevAttrConcurrentManagedAccess.
//
//The start address and end address of the memory range will be rounded down and rounded up respectively to be aligned to CPU page size before the prefetch operation is enqueued in the stream.
//
//If no physical memory has been allocated for this region, then this memory region will be populated and mapped on the destination device. If there's insufficient memory to prefetch the desired region, the Unified Memory driver may evict pages from other cudaMallocManaged allocations to host memory in order to make room. Device memory allocated using cudaMalloc or cudaMallocArray will not be evicted.
//
//By default, any mappings to the previous location of the migrated pages are removed and mappings for the new location are only setup on dstDevice. The exact behavior however also depends on the settings applied to this memory range via cudaMemAdvise as described below:
//
//If cudaMemAdviseSetReadMostly was set on any subset of this memory range, then that subset will create a read-only copy of the pages on dstDevice.
//
//If cudaMemAdviseSetPreferredLocation was called on any subset of this memory range, then the pages will be migrated to dstDevice even if dstDevice is not the preferred location of any pages in the memory range.
//
//If cudaMemAdviseSetAccessedBy was called on any subset of this memory range, then mappings to those pages from all the appropriate processors are updated to refer to the new location if establishing such a mapping is possible. Otherwise, those mappings are cleared.
//
//Note that this API is not required for functionality and only serves to improve performance by allowing the application to migrate data to a suitable location before it is accessed. Memory accesses to this range are always coherent and are allowed even when the data is actively being migrated.
//
//Note that this function is asynchronous with respect to the host and all work on other devices.
func (d Device) MemPrefetchAsync(mem cutil.Mem, size uint, s gocu.Streamer) error {
	stream := ExternalWrapper(s.Ptr())
	return newErrorRuntime("(d Device)MemPrefetchAsync: ", C.cudaMemPrefetchAsync(mem.Ptr(), (C.size_t)(size), d.c(), stream.c()))
}

//GetDevice gets the currently set device being used
func GetDevice() (Device, error) {
	var d C.int
	err := newErrorRuntime("GetDevice(): ", C.cudaGetDevice(&d))
	return (Device)(d), err
}

//GetDeviceCount returns the number of devices.
func GetDeviceCount() (n int32, err error) {
	var num C.int
	err = newErrorRuntime("Cudart-GetDeviceCount", C.cudaGetDeviceCount(&num))
	n = (int32)(num)
	return n, err

}

//MemGetInfo returns the free and total memory for device called
//Will Set Device
func (d Device) MemGetInfo() (free, total int, err error) {
	var (
		x C.size_t
		y C.size_t
	)

	d.Set()
	err = newErrorRuntime("GetMemInfo", C.cudaMemGetInfo(&x, &y))
	return int(x), int(y), err
}

//CanAccessPeer checks to see if peer's memory can be accessed by device called by method.
//Deivce calling method doesn't get set.
func (d Device) CanAccessPeer(peer Device) (bool, error) {
	var x C.int
	rte := newErrorRuntime("CanAccessPeer", C.cudaDeviceCanAccessPeer(&x, d.c(), peer.c()))
	if x > 0 {
		return true, rte
	}
	return false, rte
}

//DisablePeerAccess check cudaDeviceDisablePeerAccess
//Device calling method will be set
func (d Device) DisablePeerAccess(peer Device) error {
	err := d.Set()
	if err != nil {
		return err
	}
	return newErrorRuntime("DisablePeerAccess", C.cudaDeviceDisablePeerAccess(peer.c()))
}

//EnablePeerAccess enables memory access between device
//Device calling method will be set
func (d Device) EnablePeerAccess(peer Device) error {
	err := d.Set()
	if err != nil {
		return err
	}
	return newErrorRuntime("EnablePeerAccess", C.cudaDeviceEnablePeerAccess(peer.c(), 0))

}

//Set sets the device to use. This will change the device that is residing on the current host thread.  There is no sychronization, with the previous or new device on the host thread.
func (d Device) Set() error {
	return newErrorRuntime("Set", C.cudaSetDevice(d.c()))
}

//CreateDevice just creates a device it doesn't set it
func CreateDevice(device int32) Device {
	return (Device)(device)

}

//SetValidDevices takes a list of devices in terms of user priority for cuda execution
func SetValidDevices(devices []Device) error {

	return newErrorRuntime("SetValidDevices", C.cudaSetValidDevices(devices[0].cptr(), C.int(len(devices))))
}

//Reset resets the device. If device isn't set on current host thread. This function will auto set it. Make sure that the device that was currently using the host thread is set back onto host
func (d Device) Reset() error {

	err := d.Set()
	if err != nil {
		return err
	}
	return newErrorRuntime("Reset", C.cudaDeviceReset())

}

//CudaDeviceAttribute is used to pass device attributes It is not flushed out yet
//TODO actually build
type cudadeviceattribute C.uint

func (d cudadeviceattribute) c() uint32 {
	return uint32(d)
}
func (d Device) getattribute(attr cudadeviceattribute) (int32, error) {
	var val C.int
	err := d.Set()
	if err != nil {
		return 0, err
	}
	err = newErrorRuntime("getattribute", C.cudaDeviceGetAttribute(&val, attr.c(), d.c()))
	return int32(val), err
}

//MaxBlockDimXYZ returns an array of the values of blocks xyz in that order and an error
//Will set device
func (d Device) MaxBlockDimXYZ() ([]int32, error) {
	var err error
	xyz := make([]int32, 3)
	xyz[0], err = d.getattribute(C.cudaDevAttrMaxBlockDimX)
	if err != nil {
		return nil, err
	}
	xyz[1], err = d.getattribute(C.cudaDevAttrMaxBlockDimY)
	if err != nil {
		return nil, err
	}
	xyz[2], err = d.getattribute(C.cudaDevAttrMaxBlockDimZ)
	if err != nil {
		return nil, err
	}
	return xyz, err
}

//MaxGridDimXYZ returns an array of the values of blocks xyz in that order and an error
//Will set device
func (d Device) MaxGridDimXYZ() ([]int32, error) {
	var err error
	xyz := make([]int32, 3)
	xyz[0], err = d.getattribute(C.cudaDevAttrMaxGridDimX)
	if err != nil {
		return nil, err
	}
	xyz[1], err = d.getattribute(C.cudaDevAttrMaxGridDimY)
	if err != nil {
		return nil, err
	}
	xyz[2], err = d.getattribute(C.cudaDevAttrMaxGridDimZ)
	if err != nil {
		return nil, err
	}
	return xyz, err
}

//MaxThreadsPerBlock returns the max number of threads per block and the rutime error
//will set deivce
func (d Device) MaxThreadsPerBlock() (int32, error) {
	return d.getattribute(C.cudaDevAttrMaxThreadsPerBlock)
}

//MultiProcessorCount returns the number of multiproccessors on device and the runtime error
//will set device
func (d Device) MultiProcessorCount() (int32, error) {
	return d.getattribute(C.cudaDevAttrMultiProcessorCount)
}

//MaxThreadsPerMultiProcessor returns the number of threads that run a multiprocessor on device and the runtime error
//Will set device
func (d Device) MaxThreadsPerMultiProcessor() (int32, error) {

	return d.getattribute(C.cudaDevAttrMaxThreadsPerMultiProcessor)
}

//Major returns the major compute capability of device
func (d Device) Major() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrComputeCapabilityMajor)
	return (int)(x), err
}

//Minor returns the minor comnute capability of device
func (d Device) Minor() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrComputeCapabilityMinor)
	return (int)(x), err
}

//AttrMaxThreadsPerBlock - Maximum number of threads per block
func (d Device) AttrMaxThreadsPerBlock() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxThreadsPerBlock)
	return (int)(x), err
}

//AttrMaxBlockDimX - Maximum block dimension X
func (d Device) AttrMaxBlockDimX() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxBlockDimX)
	return (int)(x), err
}

//AttrMaxBlockDimY - Maximum block dimension Y
func (d Device) AttrMaxBlockDimY() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxBlockDimY)
	return (int)(x), err
}

//AttrMaxBlockDimZ - Maximum block dimension Z
func (d Device) AttrMaxBlockDimZ() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxBlockDimZ)
	return (int)(x), err
}

//AttrMaxGridDimX - Maximum grid dimension X
func (d Device) AttrMaxGridDimX() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxGridDimX)
	return (int)(x), err
}

//AttrMaxGridDimY - Maximum grid dimension Y
func (d Device) AttrMaxGridDimY() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxGridDimY)
	return (int)(x), err
}

//AttrMaxGridDimZ - Maximum grid dimension Z
func (d Device) AttrMaxGridDimZ() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxGridDimZ)
	return (int)(x), err
}

//AttrMaxSharedMemoryPerBlock - Maximum shared memory available per block in bytes
func (d Device) AttrMaxSharedMemoryPerBlock() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSharedMemoryPerBlock)
	return (int)(x), err
}

//AttrTotalConstantMemory - Memory available on device for __constant__ variables in a CUDA C kernel in bytes
func (d Device) AttrTotalConstantMemory() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrTotalConstantMemory)
	return (int)(x), err
}

//AttrWarpSize - Warp size in threads
func (d Device) AttrWarpSize() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrWarpSize)
	return (int)(x), err
}

//AttrMaxPitch - Maximum pitch in bytes allowed by memory copies
func (d Device) AttrMaxPitch() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxPitch)
	return (int)(x), err
}

//AttrMaxRegistersPerBlock - Maximum number of 32-bit registers available per block
func (d Device) AttrMaxRegistersPerBlock() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxRegistersPerBlock)
	return (int)(x), err
}

//AttrClockRate - Peak clock frequency in kilohertz
func (d Device) AttrClockRate() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrClockRate)
	return (int)(x), err
}

//AttrTextureAlignment - Alignment requirement for textures
func (d Device) AttrTextureAlignment() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrTextureAlignment)
	return (int)(x), err
}

//AttrGpuOverlap - Device can possibly copy memory and execute a kernel concurrently
func (d Device) AttrGpuOverlap() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrGpuOverlap)
	return (int)(x), err
}

//AttrMultiProcessorCount - Number of multiprocessors on device
func (d Device) AttrMultiProcessorCount() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMultiProcessorCount)
	return (int)(x), err
}

//AttrKernelExecTimeout - Specifies whether there is a run time limit on kernels
func (d Device) AttrKernelExecTimeout() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrKernelExecTimeout)
	return (int)(x), err
}

//AttrIntegrated - Device is integrated with host memory
func (d Device) AttrIntegrated() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrIntegrated)
	return (int)(x), err
}

//AttrCanMapHostMemory - Device can map host memory into CUDA address space
func (d Device) AttrCanMapHostMemory() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrCanMapHostMemory)
	return (int)(x), err
}

//AttrComputeMode - Compute mode (See cudaComputeMode for details)
func (d Device) AttrComputeMode() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrComputeMode)
	return (int)(x), err
}

//AttrMaxTexture1DWidth - Maximum 1D texture width
func (d Device) AttrMaxTexture1DWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture1DWidth)
	return (int)(x), err
}

//AttrMaxTexture2DWidth - Maximum 2D texture width
func (d Device) AttrMaxTexture2DWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DWidth)
	return (int)(x), err
}

//AttrMaxTexture2DHeight - Maximum 2D texture height
func (d Device) AttrMaxTexture2DHeight() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DHeight)
	return (int)(x), err
}

//AttrMaxTexture3DWidth - Maximum 3D texture width
func (d Device) AttrMaxTexture3DWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture3DWidth)
	return (int)(x), err
}

//AttrMaxTexture3DHeight - Maximum 3D texture height
func (d Device) AttrMaxTexture3DHeight() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture3DHeight)
	return (int)(x), err
}

//AttrMaxTexture3DDepth - Maximum 3D texture depth
func (d Device) AttrMaxTexture3DDepth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture3DDepth)
	return (int)(x), err
}

//AttrMaxTexture2DLayeredWidth - Maximum 2D layered texture width
func (d Device) AttrMaxTexture2DLayeredWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DLayeredWidth)
	return (int)(x), err
}

//AttrMaxTexture2DLayeredHeight - Maximum 2D layered texture height
func (d Device) AttrMaxTexture2DLayeredHeight() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DLayeredHeight)
	return (int)(x), err
}

//AttrMaxTexture2DLayeredLayers - Maximum layers in a 2D layered texture
func (d Device) AttrMaxTexture2DLayeredLayers() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DLayeredLayers)
	return (int)(x), err
}

//AttrSurfaceAlignment - Alignment requirement for surfaces
func (d Device) AttrSurfaceAlignment() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrSurfaceAlignment)
	return (int)(x), err
}

//AttrConcurrentKernels - Device can possibly execute multiple kernels concurrently
func (d Device) AttrConcurrentKernels() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrConcurrentKernels)
	return (int)(x), err
}

//AttrEccEnabled - Device has ECC support enabled
func (d Device) AttrEccEnabled() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrEccEnabled)
	return (int)(x), err
}

//AttrPciBusID - PCI bus ID of the device
func (d Device) AttrPciBusID() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrPciBusId)
	return (int)(x), err
}

//AttrPciDeviceID - PCI device ID of the device
func (d Device) AttrPciDeviceID() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrPciDeviceId)
	return (int)(x), err
}

//AttrTccDriver - Device is using TCC driver model
func (d Device) AttrTccDriver() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrTccDriver)
	return (int)(x), err
}

//AttrMemoryClockRate - Peak memory clock frequency in kilohertz
func (d Device) AttrMemoryClockRate() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMemoryClockRate)
	return (int)(x), err
}

//AttrGlobalMemoryBusWidth - Global memory bus width in bits
func (d Device) AttrGlobalMemoryBusWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrGlobalMemoryBusWidth)
	return (int)(x), err
}

//AttrL2CacheSize - Size of L2 cache in bytes
func (d Device) AttrL2CacheSize() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrL2CacheSize)
	return (int)(x), err
}

//AttrMaxThreadsPerMultiProcessor - Maximum resident threads per multiprocessor
func (d Device) AttrMaxThreadsPerMultiProcessor() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxThreadsPerMultiProcessor)
	return (int)(x), err
}

//AttrAsyncEngineCount - Number of asynchronous engines
func (d Device) AttrAsyncEngineCount() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrAsyncEngineCount)
	return (int)(x), err
}

//AttrUnifiedAddressing - Device shares a unified address space with the host
func (d Device) AttrUnifiedAddressing() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrUnifiedAddressing)
	return (int)(x), err
}

//AttrMaxTexture1DLayeredWidth - Maximum 1D layered texture width
func (d Device) AttrMaxTexture1DLayeredWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture1DLayeredWidth)
	return (int)(x), err
}

//AttrMaxTexture1DLayeredLayers - Maximum layers in a 1D layered texture
func (d Device) AttrMaxTexture1DLayeredLayers() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture1DLayeredLayers)
	return (int)(x), err
}

//AttrMaxTexture2DGatherWidth - Maximum 2D texture width if cudaArrayTextureGather is set
func (d Device) AttrMaxTexture2DGatherWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DGatherWidth)
	return (int)(x), err
}

//AttrMaxTexture2DGatherHeight - Maximum 2D texture height if cudaArrayTextureGather is set
func (d Device) AttrMaxTexture2DGatherHeight() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DGatherHeight)
	return (int)(x), err
}

//AttrMaxTexture3DWidthAlt - Alternate maximum 3D texture width
func (d Device) AttrMaxTexture3DWidthAlt() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture3DWidthAlt)
	return (int)(x), err
}

//AttrMaxTexture3DHeightAlt - Alternate maximum 3D texture height
func (d Device) AttrMaxTexture3DHeightAlt() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture3DHeightAlt)
	return (int)(x), err
}

//AttrMaxTexture3DDepthAlt - Alternate maximum 3D texture depth
func (d Device) AttrMaxTexture3DDepthAlt() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture3DDepthAlt)
	return (int)(x), err
}

//AttrPciDomainID - PCI domain ID of the device
func (d Device) AttrPciDomainID() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrPciDomainId)
	return (int)(x), err
}

//AttrTexturePitchAlignment - Pitch alignment requirement for textures
func (d Device) AttrTexturePitchAlignment() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrTexturePitchAlignment)
	return (int)(x), err
}

//AttrMaxTextureCubemapWidth - Maximum cubemap texture width/height
func (d Device) AttrMaxTextureCubemapWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTextureCubemapWidth)
	return (int)(x), err
}

//AttrMaxTextureCubemapLayeredWidth - Maximum cubemap layered texture width/height
func (d Device) AttrMaxTextureCubemapLayeredWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTextureCubemapLayeredWidth)
	return (int)(x), err
}

//AttrMaxTextureCubemapLayeredLayers - Maximum layers in a cubemap layered texture
func (d Device) AttrMaxTextureCubemapLayeredLayers() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTextureCubemapLayeredLayers)
	return (int)(x), err
}

//AttrMaxSurface1DWidth - Maximum 1D surface width
func (d Device) AttrMaxSurface1DWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface1DWidth)
	return (int)(x), err
}

//AttrMaxSurface2DWidth - Maximum 2D surface width
func (d Device) AttrMaxSurface2DWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface2DWidth)
	return (int)(x), err
}

//AttrMaxSurface2DHeight - Maximum 2D surface height
func (d Device) AttrMaxSurface2DHeight() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface2DHeight)
	return (int)(x), err
}

//AttrMaxSurface3DWidth - Maximum 3D surface width
func (d Device) AttrMaxSurface3DWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface3DWidth)
	return (int)(x), err
}

//AttrMaxSurface3DHeight - Maximum 3D surface height
func (d Device) AttrMaxSurface3DHeight() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface3DHeight)
	return (int)(x), err
}

//AttrMaxSurface3DDepth - Maximum 3D surface depth
func (d Device) AttrMaxSurface3DDepth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface3DDepth)
	return (int)(x), err
}

//AttrMaxSurface1DLayeredWidth - Maximum 1D layered surface width
func (d Device) AttrMaxSurface1DLayeredWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface1DLayeredWidth)
	return (int)(x), err
}

//AttrMaxSurface1DLayeredLayers - Maximum layers in a 1D layered surface
func (d Device) AttrMaxSurface1DLayeredLayers() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface1DLayeredLayers)
	return (int)(x), err
}

//AttrMaxSurface2DLayeredWidth - Maximum 2D layered surface width
func (d Device) AttrMaxSurface2DLayeredWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface2DLayeredWidth)
	return (int)(x), err
}

//AttrMaxSurface2DLayeredHeight - Maximum 2D layered surface height
func (d Device) AttrMaxSurface2DLayeredHeight() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface2DLayeredHeight)
	return (int)(x), err
}

//AttrMaxSurface2DLayeredLayers - Maximum layers in a 2D layered surface
func (d Device) AttrMaxSurface2DLayeredLayers() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurface2DLayeredLayers)
	return (int)(x), err
}

//AttrMaxSurfaceCubemapWidth - Maximum cubemap surface width
func (d Device) AttrMaxSurfaceCubemapWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurfaceCubemapWidth)
	return (int)(x), err
}

//AttrMaxSurfaceCubemapLayeredWidth - Maximum cubemap layered surface width
func (d Device) AttrMaxSurfaceCubemapLayeredWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurfaceCubemapLayeredWidth)
	return (int)(x), err
}

//AttrMaxSurfaceCubemapLayeredLayers - Maximum layers in a cubemap layered surface
func (d Device) AttrMaxSurfaceCubemapLayeredLayers() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSurfaceCubemapLayeredLayers)
	return (int)(x), err
}

//AttrMaxTexture1DLinearWidth - Maximum 1D linear texture width
func (d Device) AttrMaxTexture1DLinearWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture1DLinearWidth)
	return (int)(x), err
}

//AttrMaxTexture2DLinearWidth - Maximum 2D linear texture width
func (d Device) AttrMaxTexture2DLinearWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DLinearWidth)
	return (int)(x), err
}

//AttrMaxTexture2DLinearHeight - Maximum 2D linear texture height
func (d Device) AttrMaxTexture2DLinearHeight() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DLinearHeight)
	return (int)(x), err
}

//AttrMaxTexture2DLinearPitch - Maximum 2D linear texture pitch in bytes
func (d Device) AttrMaxTexture2DLinearPitch() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DLinearPitch)
	return (int)(x), err
}

//AttrMaxTexture2DMipmappedWidth - Maximum mipmapped 2D texture width
func (d Device) AttrMaxTexture2DMipmappedWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DMipmappedWidth)
	return (int)(x), err
}

//AttrMaxTexture2DMipmappedHeight - Maximum mipmapped 2D texture height
func (d Device) AttrMaxTexture2DMipmappedHeight() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture2DMipmappedHeight)
	return (int)(x), err
}

//AttrComputeCapabilityMajor - Major compute capability version number
func (d Device) AttrComputeCapabilityMajor() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrComputeCapabilityMajor)
	return (int)(x), err
}

//AttrComputeCapabilityMinor - Minor compute capability version number
func (d Device) AttrComputeCapabilityMinor() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrComputeCapabilityMinor)
	return (int)(x), err
}

//AttrMaxTexture1DMipmappedWidth - Maximum mipmapped 1D texture width
func (d Device) AttrMaxTexture1DMipmappedWidth() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxTexture1DMipmappedWidth)
	return (int)(x), err
}

//AttrStreamPrioritiesSupported - Device supports stream priorities
func (d Device) AttrStreamPrioritiesSupported() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrStreamPrioritiesSupported)
	return (int)(x), err
}

//AttrGlobalL1CacheSupported - Device supports caching globals in L1
func (d Device) AttrGlobalL1CacheSupported() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrGlobalL1CacheSupported)
	return (int)(x), err
}

//AttrLocalL1CacheSupported - Device supports caching locals in L1
func (d Device) AttrLocalL1CacheSupported() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrLocalL1CacheSupported)
	return (int)(x), err
}

//AttrMaxSharedMemoryPerMultiprocessor - Maximum shared memory available per multiprocessor in bytes
func (d Device) AttrMaxSharedMemoryPerMultiprocessor() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSharedMemoryPerMultiprocessor)
	return (int)(x), err
}

//AttrMaxRegistersPerMultiprocessor - Maximum number of 32-bit registers available per multiprocessor
func (d Device) AttrMaxRegistersPerMultiprocessor() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxRegistersPerMultiprocessor)
	return (int)(x), err
}

//AttrManagedMemory - Device can allocate managed memory on this system
func (d Device) AttrManagedMemory() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrManagedMemory)
	return (int)(x), err
}

//AttrIsMultiGpuBoard - Device is on a multi-GPU board
func (d Device) AttrIsMultiGpuBoard() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrIsMultiGpuBoard)
	return (int)(x), err
}

//AttrMultiGpuBoardGroupID - Unique identifier for a group of devices on the same multi-GPU board
func (d Device) AttrMultiGpuBoardGroupID() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMultiGpuBoardGroupID)
	return (int)(x), err
}

//AttrHostNativeAtomicSupported - Link between the device and the host supports native atomic operations
func (d Device) AttrHostNativeAtomicSupported() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrHostNativeAtomicSupported)
	return (int)(x), err
}

//AttrSingleToDoublePrecisionPerfRatio - Ratio of single precision performance (in floating-point operations per second) to double precision performance
func (d Device) AttrSingleToDoublePrecisionPerfRatio() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrSingleToDoublePrecisionPerfRatio)
	return (int)(x), err
}

//AttrPageableMemoryAccess - Device supports coherently accessing pageable memory without calling cudaHostRegister on it
func (d Device) AttrPageableMemoryAccess() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrPageableMemoryAccess)
	return (int)(x), err
}

//AttrConcurrentManagedAccess - Device can coherently access managed memory concurrently with the CPU
func (d Device) AttrConcurrentManagedAccess() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrConcurrentManagedAccess)
	return (int)(x), err
}

//AttrComputePreemptionSupported - Device supports Compute Preemption
func (d Device) AttrComputePreemptionSupported() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrComputePreemptionSupported)
	return (int)(x), err
}

//AttrCanUseHostPointerForRegisteredMem - Device can access host registered memory at the same virtual address as the CPU
func (d Device) AttrCanUseHostPointerForRegisteredMem() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrCanUseHostPointerForRegisteredMem)
	return (int)(x), err
}

//AttrCooperativeLaunch - Device supports launching cooperative kernels via cudaLaunchCooperativeKernel
func (d Device) AttrCooperativeLaunch() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrCooperativeLaunch)
	return (int)(x), err
}

//AttrCooperativeMultiDeviceLaunch - Device can participate in cooperative kernels launched via cudaLaunchCooperativeKernelMultiDevice
func (d Device) AttrCooperativeMultiDeviceLaunch() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrCooperativeMultiDeviceLaunch)
	return (int)(x), err
}

//AttrMaxSharedMemoryPerBlockOptin - The maximum optin shared memory per block. This value may vary by chip. See cudaFuncSetAttribute
func (d Device) AttrMaxSharedMemoryPerBlockOptin() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrMaxSharedMemoryPerBlockOptin)
	return (int)(x), err
}

//AttrCanFlushRemoteWrites - Device supports flushing of outstanding remote writes.
func (d Device) AttrCanFlushRemoteWrites() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrCanFlushRemoteWrites)
	return (int)(x), err
}

//AttrHostRegisterSupported - Device supports host memory registration via cudaHostRegister.
func (d Device) AttrHostRegisterSupported() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrHostRegisterSupported)
	return (int)(x), err
}

//AttrPageableMemoryAccessUsesHostPageTables - Device accesses pageable memory via the host page tables.
func (d Device) AttrPageableMemoryAccessUsesHostPageTables() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrPageableMemoryAccessUsesHostPageTables)
	return (int)(x), err
}

//AttrDirectManagedMemAccessFromHost - Host can directly access managed memory on the device without migration.
func (d Device) AttrDirectManagedMemAccessFromHost() (int, error) {
	x, err := d.getattribute(C.cudaDevAttrDirectManagedMemAccessFromHost)
	return (int)(x), err
}
