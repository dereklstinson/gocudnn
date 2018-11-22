package gocudnn

//#include <cuda_runtime_api.h>
//#include <cuda.h>
import "C"
import (
	"runtime"
)

func init() {
	err := newErrorDriver("intit", C.cuInit(0))

	if err != nil {
		panic(err)
	}
}

//Cuda is a nil struct that is used to pass Cuda functions
type Cuda struct {
}

//Device is a struct that holds a device info.
type Device struct {
	id    C.CUdevice
	idinC CInt
	major int
	minor int
	//	thread int32
}

//DeviceSync Blocks until the device has completed all preceding requested tasks.
//DeviceSync() returns an error if one of the preceding tasks has failed.
//If the cudaDeviceScheduleBlockingSync flag was set for this device,
// the host thread will block until the device has finished its work.
func (cu Cuda) DeviceSync() error {
	return newErrorRuntime("cudaDeviceSynchronize: ", C.cudaDeviceSynchronize())
}

//GetDeviceList returns a list of *Devices that are not set yet.
func (cu Cuda) GetDeviceList() ([]*Device, error) {

	y, err := cu.devicecount()
	if err != nil {
		return nil, err
	}
	var x []*Device
	zero := CInt(0)
	for i := zero; i < y; i++ {
		var id C.CUdevice
		err = newErrorDriver("Getting DeviceID", C.cuDeviceGet(&id, i.c()))
		if err != nil {
			return nil, err
		}
		var major C.int
		var minor C.int
		err = newErrorDriver("cuDeviceGetAttribute - major", C.cuDeviceGetAttribute(&major, C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, id))
		if err != nil {
			return nil, err
		}
		err = newErrorDriver("cuDeviceGetAttribute - minor", C.cuDeviceGetAttribute(&minor, C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, id))
		if err != nil {
			return nil, err
		}

		x = append(x, &Device{id: id, major: int(major), minor: int(minor), idinC: CInt(id)})
	}
	return x, nil
}

//MemGetInfo returns the free and total memory for the currently set device.
func (cu Cuda) MemGetInfo() (free, total int, err error) {
	var (
		x C.size_t
		y C.size_t
	)
	err = newErrorRuntime("GetMemInfo", C.cudaMemGetInfo(&x, &y))
	return int(x), int(y), err
}

//DeviceCount returns the number of cuda devices
func (cu Cuda) devicecount() (CInt, error) {
	var x C.int
	rte := C.cuDeviceGetCount(&x)
	return CInt(x), newErrorDriver("DeviceCount", rte)
}

//EnablePeerAccess check cudaDeviceEnablePeerAccess
func (d *Device) EnablePeerAccess(peer *Device) error {
	err := d.Set()
	if err != nil {
		return err
	}
	return newErrorRuntime("EnablePeerAccess", C.cudaDeviceEnablePeerAccess(peer.idinC.c(), C.uint(0)))
}

//CudaDeviceAttribute is used to pass device attributes It is not flushed out yet
//TODO actually build
type cudadeviceattribute C.uint

func (d cudadeviceattribute) c() uint32 {
	return uint32(d)
}
func (d *Device) getattribute(attr cudadeviceattribute) (int32, error) {
	var val C.int
	err := newErrorRuntime("getattribute", C.cudaDeviceGetAttribute(&val, attr.c(), d.idinC.c()))
	return int32(val), err
}

//MaxBlockDimXYZ returns an array of the values of blocks xyz in that order and an error
func (d *Device) MaxBlockDimXYZ() ([]int32, error) {
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
func (d *Device) MaxGridDimXYZ() ([]int32, error) {
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
func (d *Device) MaxThreadsPerBlock() (int32, error) {
	return d.getattribute(C.cudaDevAttrMaxThreadsPerBlock)
}

//MultiProcessorCount returns the number of multiproccessors on device and the runtime error
func (d *Device) MultiProcessorCount() (int32, error) {
	return d.getattribute(C.cudaDevAttrMultiProcessorCount)
}

//MaxThreadsPerMultiProcessor returns the number of threads that run a multiprocessor on device and the runtime error
func (d *Device) MaxThreadsPerMultiProcessor() (int32, error) {
	return d.getattribute(C.cudaDevAttrMaxThreadsPerMultiProcessor)
}

//Reset resets the device. If device isn't set on current host thread. This function will auto set it. Make sure that the device that was currently using the host thread is set back onto host
func (d *Device) Reset() error {

	err := d.Set()
	if err != nil {
		return err
	}
	x := C.cudaDeviceReset()

	return newErrorRuntime("Reset", x)

}

//Set sets the device to use. This will change the device that is residing on the current host thread.  There is no sychronization, with the previous or new device on the host thread.
func (d *Device) Set() error {
	return newErrorRuntime("Set", C.cudaSetDevice(d.idinC.c()))
}

//SetValidDevices takes a list of devices in terms of user priority for cuda execution
func (cu Cuda) SetValidDevices(devices []*Device) error {
	x := make([]C.int, len(devices))
	for i := 0; i < len(devices); i++ {
		x[i] = C.int(devices[i].id)
	}
	return newErrorRuntime("SetValidDevices", C.cudaSetValidDevices(&x[0], C.int(len(devices))))
}

//CreateDevice sets the device on the current host thread.
//Be sure when starting a goroutine lock the thread before calling this.
func (cu Cuda) CreateDevice(device int32) (*Device, error) {
	d := C.int(device)
	x := C.cudaSetDevice(d)
	return &Device{
		id:    C.CUdevice(d),
		idinC: CInt(d),
	}, newErrorRuntime("SetDevice", x)
}

//Major returns the compute capability major value
func (d *Device) Major() int {
	return d.major
}

//Minor returns the compute capability minor value
func (d *Device) Minor() int {
	return d.minor
}

//LockHostThread locks the current host thread
func (cu Cuda) LockHostThread() {
	runtime.LockOSThread()
}

//UnLockHostThread unlocks the current host thread
func (cu Cuda) UnLockHostThread() {
	runtime.UnlockOSThread()
}
