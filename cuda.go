package gocudnn

//#include <cuda_runtime_api.h>
import "C"
import (
	"errors"
	"runtime"
)

//Cuda is a nil struct that is used to pass Cuda functions
type Cuda struct {
}
type Device struct {
	id        int32
	beingused bool
	thread    int32
}

//GetDevice will get the cuda device
func (cu Cuda) GetDeviceList() {

}

//DeviceCount returns the number of cuda devices
func (cu Cuda) devicecount() (int32, error) {
	var x C.int
	rte := C.cudaGetDeviceCount(&x)
	return int32(x), newErrorRuntime("DeviceCount", rte)
}
func (d *Device) Reset(Override bool) error {
	if d.beingused == true {
		if Override == true {

			x := C.cudaDeviceReset()
			return newErrorRuntime("Reset", x)
		}
		return errors.New("Device in use.")
	}
	x := C.cudaDeviceReset()
	return newErrorRuntime("Reset", x)
}

//SetDevice sets the device on the current host thread.
//Be sure when starting a goroutine lock the thread before calling this.
func (cu Cuda) SetDevice(device int32) (*Device, error) {
	x := C.cudaSetDevice(C.int(device))
	return &Device{
		id: device,
	}, newErrorRuntime("SetDevice", x)
}
func (cu Cuda) LockHostThread() {
	runtime.LockOSThread()
}
func (cu Cuda) UnLockHostThread() {
	runtime.UnlockOSThread()
}
