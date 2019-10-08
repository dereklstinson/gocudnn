package cuda

//#include <cuda.h>
import "C"
import (
	"runtime"
)

//Device is a struct that holds a device info.
type Device C.CUdevice

func (d Device) c() C.CUdevice {
	return (C.CUdevice)(d)
}

//Major returns the compute capability major part of cuda device
func (d Device) Major() (int, error) {
	val, err := (d.getattribute(C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR))
	return (int)(val), err
}

//Minor returns the compute capability minor part
func (d Device) Minor() (int, error) {
	val, err := (d.getattribute(C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))
	return (int)(val), err

}

func (d Device) getattribute(atribute C.CUdevice_attribute) (int32, error) {
	var val C.int
	err := newErrorDriver("cuDeviceGetAttribute - major", C.cuDeviceGetAttribute(&val, atribute, d.c()))
	return (int32)(val), err
}

//GetDeviceList returns a list of *Devices that are not set yet.
func GetDeviceList() ([]Device, error) {

	y, err := devicecount()
	if err != nil {
		return nil, err
	}
	x := make([]Device, y)
	zero := C.int(0)
	for i := zero; i < y; i++ {
		var id C.CUdevice
		err = newErrorDriver("Getting DeviceID", C.cuDeviceGet(&id, i))
		if err != nil {
			return nil, err
		}
		x[i] = Device(id)

	}
	return x, nil
}

//DeviceCount returns the number of cuda devices
func devicecount() (C.int, error) {
	var x C.int
	rte := C.cuDeviceGetCount(&x)
	return x, newErrorDriver("DeviceCount", rte)
}

//LockHostThread locks the current host thread
func LockHostThread() {
	runtime.LockOSThread()
}

//UnLockHostThread unlocks the current host thread
func UnLockHostThread() {
	runtime.UnlockOSThread()
}

//MaxBlockDimXYZ returns an array of the values of blocks xyz in that order and an error
//Will set device
func (d Device) MaxBlockDimXYZ() ([]int32, error) {
	var err error
	xyz := make([]int32, 3)
	xyz[0], err = d.getattribute(C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
	if err != nil {
		return nil, err
	}
	xyz[1], err = d.getattribute(C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
	if err != nil {
		return nil, err
	}
	xyz[2], err = d.getattribute(C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
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
	xyz[0], err = d.getattribute(C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
	if err != nil {
		return nil, err
	}
	xyz[1], err = d.getattribute(C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
	if err != nil {
		return nil, err
	}
	xyz[2], err = d.getattribute(C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
	if err != nil {
		return nil, err
	}
	return xyz, err
}

//MaxThreadsPerBlock returns the max number of threads per block and the rutime error
//will set deivce
func (d Device) MaxThreadsPerBlock() (int32, error) {
	return d.getattribute(C.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
}

//MultiProcessorCount returns the number of multiproccessors on device and the runtime error
//will set device
func (d Device) MultiProcessorCount() (int32, error) {
	return d.getattribute(C.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
}

//MaxThreadsPerMultiProcessor returns the number of threads that run a multiprocessor on device and the runtime error
//Will set device
func (d Device) MaxThreadsPerMultiProcessor() (int32, error) {

	return d.getattribute(C.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
}
