package gocudnn

//#include <cuda_runtime_api.h>
import "C"
import (
	"runtime"
)

//Cuda is a nil struct that is used to pass Cuda functions
type Cuda struct {
}

type Device struct {
	id     CInt
	thread int32
}

//GetDeviceList returns a list of *Devices that are not set yet.
func (cu Cuda) GetDeviceList() ([]*Device, error) {

	y, err := cu.devicecount()
	if err != nil {
		return nil, err
	}
	x := make([]*Device, y)
	zero := int32(0)
	for i := zero; i < y; i++ {
		x[i].id = CInt(i)
	}
	return x, nil
}

//DeviceCount returns the number of cuda devices
func (cu Cuda) devicecount() (int32, error) {
	var x C.int
	rte := C.cudaGetDeviceCount(&x)
	return int32(x), newErrorRuntime("DeviceCount", rte)
}

func (d *Device) EnablePeerAccess(peer *Device) error {
	err := d.Set()
	if err != nil {
		return err
	}
	return newErrorRuntime("EnablePeerAccess", C.cudaDeviceEnablePeerAccess(peer.id.c(), C.uint(0)))
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
	return newErrorRuntime("Set", C.cudaSetDevice(d.id.c()))
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
	x := C.cudaSetDevice(C.int(device))
	return &Device{
		id: CInt(device),
	}, newErrorRuntime("SetDevice", x)
}
func (cu Cuda) LockHostThread() {
	runtime.LockOSThread()
}
func (cu Cuda) UnLockHostThread() {
	runtime.UnlockOSThread()
}
