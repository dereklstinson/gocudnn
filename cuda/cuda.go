package cuda

//#include <cuda_runtime_api.h>
//#include <cuda.h>
import "C"
import (
	"runtime"
	//There is an init here than I want to make sure that is used
	_ "github.com/dereklstinson/GoCudnn/gocu"
)

//Device is a struct that holds a device info.
type Device struct {
	id    C.CUdevice
	idinC C.int
	major int
	minor int
	//	thread int32
}

//GetDeviceList returns a list of *Devices that are not set yet.
func GetDeviceList() ([]*Device, error) {

	y, err := devicecount()
	if err != nil {
		return nil, err
	}
	var x []*Device
	zero := C.int(0)
	for i := zero; i < y; i++ {
		var id C.CUdevice
		err = newErrorDriver("Getting DeviceID", C.cuDeviceGet(&id, i))
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

		x = append(x, &Device{id: id, major: int(major), minor: int(minor), idinC: C.int(id)})
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
