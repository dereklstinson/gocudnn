package cudart

/*

#include <cuda_runtime_api.h>
#include <cuda.h>
*/
import "C"

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

//GetDevice gets the currently set device being used
func GetDevice() (Device, error) {
	var d C.int
	err := newErrorRuntime("cudaDeviceSynchronize: ", C.cudaGetDevice(&d))
	return (Device)(d), err
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

//CanAccessPeer checks to see if peer's memory can be accessed my device called by method.
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

//CreateDevice sets the device on the current host thread.
//Be sure when starting a goroutine lock the thread before calling this.
func CreateDevice(device int32) (Device, error) {
	var x Device
	x = (Device)(device)
	err := x.Set()
	return x, err

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
