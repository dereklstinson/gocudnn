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
	return d.getattribute(C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
}

//Minor returns the compute capability minor part
func (d Device) Minor() (int, error) {
	return d.getattribute(C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
}

func (d Device) getattribute(atribute C.CUdevice_attribute) (int, error) {
	var val C.int
	err := newErrorDriver("cuDeviceGetAttribute - major", C.cuDeviceGetAttribute(&val, atribute, d.c()))
	return (int)(val), err
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
