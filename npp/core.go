package npp

//#include<nppcore.h>
import "C"
import (
	"errors"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
)

//GetLibVersion - return A struct containing separate values for major and minor revision and build number.
func GetLibVersion() (version LibraryVersion) {
	x := C.nppGetLibVersion()
	version = (LibraryVersion)(*x)
	return version
}

//GetGpuComputeCapability returns the CUDA compute model is supported by the active CUDA device?
//Before trying to call any NPP functions, the user should make a call
//this function to ensure that the current machine has a CUDA capable device.
func GetGpuComputeCapability() GpuComputeCapability {
	return (GpuComputeCapability)(C.nppGetGpuComputeCapability())
}

//GetGpuNumSMs returns the number of streaming multiprocessors the current set device has
func GetGpuNumSMs() int32 {
	return (int32)(C.nppGetGpuNumSMs())
}

//GetMaxThreadsPerBlock get max threads per sm block
func GetMaxThreadsPerBlock() int32 {
	return (int32)(C.nppGetMaxThreadsPerBlock())
}

//GetMaxThreadsPerSM gets the max threads per sm on currently set device
func GetMaxThreadsPerSM() int32 {
	return (int32)(C.nppGetMaxThreadsPerSM())
}

//GetGpuDeviceProperties returns the properties of the device
func GetGpuDeviceProperties() (pMaxThreadsPerSM, pMaxThreadsPerBlock, pNumberOfSMs int32, err error) {
	var (
		pmtpsm C.int
		pmtpb  C.int
		pnosm  C.int
	)
	x := C.nppGetGpuDeviceProperties(&pmtpsm, &pmtpb, &pnosm)
	if x < 0 {
		err = errors.New("Npp Device Error")
	}
	pMaxThreadsPerSM = (int32)(pmtpsm)
	pMaxThreadsPerBlock = (int32)(pmtpb)
	pNumberOfSMs = (int32)(pnosm)
	return pMaxThreadsPerSM, pMaxThreadsPerBlock, pNumberOfSMs, err

}

//SetStream sets the stream
func SetStream(hStream gocu.Streamer) error {
	return status(C.nppSetStream((C.cudaStream_t)(hStream.Ptr()))).error()
}

//GetStream returns the current gocu.Streamer
func GetStream() gocu.Streamer {
	return cudart.ExternalWrapper((unsafe.Pointer)(C.nppGetStream()))
}

/*
 const char * nppGetGpuName(void)
 unsigned int nppGetStreamNumSMs(void)
 unsigned int nppGetStreamMaxThreadsPerSM(void)
*/

/**
  int nppGetGpuNumSMs(void);
  * Get the number of Streaming Multiprocessors (SM) on the active CUDA device.
  *
  * \return Number of SMs of the default CUDA device.
*/

/**int nppGetMaxThreadsPerBlock(void);
 * Get the maximum number of threads per block on the active CUDA device.
 *
 * \return Maximum number of threads per block on the active CUDA device.
 */

/** int  nppGetMaxThreadsPerSM(void);
 * Get the maximum number of threads per SM for the active GPU
 *
 * \return Maximum number of threads per SM for the active GPU
 */

/** int nppGetGpuDeviceProperties(int * pMaxThreadsPerSM, int * pMaxThreadsPerBlock, int * pNumberOfSMs);
 * Get the maximum number of threads per SM, maximum threads per block, and number of SMs for the active GPU
 *
 * \return cudaSuccess for success, -1 for failure
 */

/**const char * nppGetGpuName(void);
 * Get the name of the active CUDA device.
 *
 * \return Name string of the active graphics-card/compute device in a system.
 */

/**  cudaStream_t nppGetStream(void);
 * Get the NPP CUDA stream.
 * NPP enables concurrent device tasks via a global stream state varible.
 * The NPP stream by default is set to stream 0, i.e. non-concurrent mode.
 * A user can set the NPP stream to any valid CUDA stream. All CUDA commands
 * issued by NPP (e.g. kernels launched by the NPP library) are then
 * issed to that NPP stream.
 */

/**  unsigned int nppGetStreamNumSMs(void);
 * Get the number of SMs on the device associated with the current NPP CUDA stream.
 * NPP enables concurrent device tasks via a global stream state varible.
 * The NPP stream by default is set to stream 0, i.e. non-concurrent mode.
 * A user can set the NPP stream to any valid CUDA stream. All CUDA commands
 * issued by NPP (e.g. kernels launched by the NPP library) are then
 * issed to that NPP stream.  This call avoids a cudaGetDeviceProperties() call.
 */

/**unsigned int nppGetStreamMaxThreadsPerSM(void);
 * Get the maximum number of threads per SM on the device associated with the current NPP CUDA stream.
 * NPP enables concurrent device tasks via a global stream state varible.
 * The NPP stream by default is set to stream 0, i.e. non-concurrent mode.
 * A user can set the NPP stream to any valid CUDA stream. All CUDA commands
 * issued by NPP (e.g. kernels launched by the NPP library) are then
 * issed to that NPP stream.  This call avoids a cudaGetDeviceProperties() call.
 */

/**NppStatus  nppSetStream(cudaStream_t hStream);
 * Set the NPP CUDA stream.  This function now returns an error if a problem occurs with Cuda stream management.
 *   This function should only be called if a call to nppGetStream() returns a stream number which is different from
 *   the desired stream since unnecessarily flushing the current stream can significantly affect performance.
 * \see nppGetStream()
 */
