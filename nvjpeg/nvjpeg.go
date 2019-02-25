package nvjpeg

/*
#include <nvjpeg.h>
#include <cuda_runtime_api.h>
*/
import "C"
import (
	"errors"
	"runtime"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

type status C.nvjpegStatus_t

func (n status) error() error {
	switch n {
	case status(C.NVJPEG_STATUS_SUCCESS):
		return nil
	case status(C.NVJPEG_STATUS_NOT_INITIALIZED):
		return n
	case status(C.NVJPEG_STATUS_INVALID_PARAMETER):
		return n
	case status(C.NVJPEG_STATUS_BAD_JPEG):
		return n
	case status(C.NVJPEG_STATUS_JPEG_NOT_SUPPORTED):
		return n
	case status(C.NVJPEG_STATUS_ALLOCATOR_FAILURE):
		return n
	case status(C.NVJPEG_STATUS_EXECUTION_FAILED):
		return n
	case status(C.NVJPEG_STATUS_ARCH_MISMATCH):
		return n
	case status(C.NVJPEG_STATUS_INTERNAL_ERROR):
		return n
	default:
		return errors.New("Unsupported Error")
	}

}
func (n status) Error() string {
	switch n {
	case status(C.NVJPEG_STATUS_NOT_INITIALIZED):
		return "NVJPEG_STATUS_NOT_INITIALIZED"
	case status(C.NVJPEG_STATUS_INVALID_PARAMETER):
		return "NVJPEG_STATUS_INVALID_PARAMETER"
	case status(C.NVJPEG_STATUS_BAD_JPEG):
		return "NVJPEG_STATUS_BAD_JPEG"
	case status(C.NVJPEG_STATUS_JPEG_NOT_SUPPORTED):
		return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED"
	case status(C.NVJPEG_STATUS_ALLOCATOR_FAILURE):
		return "NVJPEG_STATUS_ALLOCATOR_FAILURE"
	case status(C.NVJPEG_STATUS_EXECUTION_FAILED):
		return "NVJPEG_STATUS_EXECUTION_FAILED"
	case status(C.NVJPEG_STATUS_ARCH_MISMATCH):
		return "NVJPEG_STATUS_ARCH_MISMATCH"
	case status(C.NVJPEG_STATUS_INTERNAL_ERROR):
		return "NVJPEG_STATUS_INTERNAL_ERROR"
	default:
		return "Unsupported Error"

	}

}

// ChromaSubsampling returned by getImageInfo identifies image chroma subsampling stored inside JPEG input stream
// In the case of NVJPEG_CSS_GRAY only 1 luminance channel is encoded in JPEG input stream
// Otherwise both chroma planes are present
// Initial release support: 4:4:4, 4:2:0, 4:2:2, Grayscale
type ChromaSubsampling C.nvjpegChromaSubsampling_t

func (ch ChromaSubsampling) c() C.nvjpegChromaSubsampling_t {
	return C.nvjpegChromaSubsampling_t(ch)
}

//ChromaSubsamplingFlag returns  ChromaSAubsampling through methods
type ChromaSubsamplingFlag struct {
}

//CSS444 is a standard flag
func (c ChromaSubsamplingFlag) CSS444() ChromaSubsampling {
	return ChromaSubsampling(C.NVJPEG_CSS_444)
}

//CSS422 is a standard flag
func (c ChromaSubsamplingFlag) CSS422() ChromaSubsampling {
	return ChromaSubsampling(C.NVJPEG_CSS_422)
}

//CSS420 is a standard flag
func (c ChromaSubsamplingFlag) CSS420() ChromaSubsampling {
	return ChromaSubsampling(C.NVJPEG_CSS_420)
}

//CSS411 is a standard flag
func (c ChromaSubsamplingFlag) CSS411() ChromaSubsampling {
	return ChromaSubsampling(C.NVJPEG_CSS_411)
}

//CSS410 is a standard flag
func (c ChromaSubsamplingFlag) CSS410() ChromaSubsampling {
	return ChromaSubsampling(C.NVJPEG_CSS_410)
}

//CSSGRAY is a standard flag
func (c ChromaSubsamplingFlag) CSSGRAY() ChromaSubsampling {
	return ChromaSubsampling(C.NVJPEG_CSS_GRAY)
}

//CSSUNKNOWN is a standard flag
func (c ChromaSubsamplingFlag) CSSUNKNOWN() ChromaSubsampling {
	return ChromaSubsampling(C.NVJPEG_CSS_UNKNOWN)
}

//OutputFormat specifies what type of output user wants for image decoding
type OutputFormat C.nvjpegOutputFormat_t

func (o OutputFormat) c() C.nvjpegOutputFormat_t {
	return C.nvjpegOutputFormat_t(o)
}

// OutputFormatFlag passes OutputFormat Flags through Methods
type OutputFormatFlag struct {
}

//Unchanged returns decoded image as it is - write planar output
func (o OutputFormatFlag) Unchanged() OutputFormat {
	return OutputFormat(C.NVJPEG_OUTPUT_UNCHANGED)
}

//YUV returns planar luma and chroma
func (o OutputFormatFlag) YUV() OutputFormat {
	return OutputFormat(C.NVJPEG_OUTPUT_YUV)
}

//Y return luma component only, write to 1-st channel of nvjpegImage_t
func (o OutputFormatFlag) Y() OutputFormat {
	return OutputFormat(C.NVJPEG_OUTPUT_Y)

}

//RGB convert to planar RGB
func (o OutputFormatFlag) RGB() OutputFormat {
	return OutputFormat(C.NVJPEG_OUTPUT_RGB)

}

//BGR convert to planar BGR
func (o OutputFormatFlag) BGR() OutputFormat {
	return OutputFormat(C.NVJPEG_OUTPUT_BGR)
}

//RGBI convert to interleaved RGB and write to 1-st channel of nvjpegImage_t
func (o OutputFormatFlag) RGBI() OutputFormat {
	return OutputFormat(C.NVJPEG_OUTPUT_RGBI)
}

//BGRI convert to interleaved BGR and write to 1-st channel of nvjpegImage_t
func (o OutputFormatFlag) BGRI() OutputFormat {
	return OutputFormat(C.NVJPEG_OUTPUT_BGRI)
}

//Backend are flags that are used to set the implimentation.
type Backend C.nvjpegBackend_t

func (b Backend) c() C.nvjpegBackend_t {
	return C.nvjpegBackend_t(b)
}

//Backendflag returns Backend flags as methods
type Backendflag struct {
}

// Default returns Backend(C.NVJPEG_BACKEND_DEFAULT)
func (b Backendflag) Default() Backend {
	return Backend(C.NVJPEG_BACKEND_DEFAULT)
}

// Hybrid returns Backend(C.NVJPEG_BACKEND_HYBRID)
func (b Backendflag) Hybrid() Backend {
	return Backend(C.NVJPEG_BACKEND_HYBRID)
}

//GPU returns Backend(C.NVJPEG_BACKEND_GPU)
func (b Backendflag) GPU() Backend {
	return Backend(C.NVJPEG_BACKEND_GPU)
}

// Image is an Output descriptor.
// Data that is written to planes depends on output forman
type Image C.nvjpegImage_t

//Get gets the underlying values of image
func (im *Image) Get() (channel [C.NVJPEG_MAX_COMPONENT]*byte, pitch [C.NVJPEG_MAX_COMPONENT]uint32) {
	for i := 0; i < int(C.NVJPEG_MAX_COMPONENT); i++ {
		channel[i] = (*byte)(im.channel[i])
		pitch[i] = (uint32)(im.pitch[i])
	}
	return channel, pitch
}
func (im *Image) cptr() *C.nvjpegImage_t {
	return (*C.nvjpegImage_t)(im)
}
func (im *Image) c() C.nvjpegImage_t {
	return (C.nvjpegImage_t)(*im)
}

/*
typedef struct
{
    unsigned char * channel[NVJPEG_MAX_COMPONENT];
    unsigned int    pitch[NVJPEG_MAX_COMPONENT];
} nvjpegImage_t;
*/
/*
// Prototype for device memory allocation.
typedef int (*tDevMalloc)(void**, size_t);
// Prototype for device memory release
typedef int (*tDevFree)(void*);
*/

//DevAllocator - Memory allocator using mentioned prototypes, provided to nvjpegCreate
// This allocator will be used for all device memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
type DevAllocator C.nvjpegDevAllocator_t

func (d *DevAllocator) cptr() *C.nvjpegDevAllocator_t {
	return (*C.nvjpegDevAllocator_t)(d)
}
func (d DevAllocator) c() C.nvjpegDevAllocator_t {
	return (C.nvjpegDevAllocator_t)(d)
}

/*
typedef struct
{
    tDevMalloc dev_malloc;
    tDevFree dev_free;
} nvjpegDevAllocator_t;
*/

// Handle - Opaque library handle identifier.
type Handle struct {
	h C.nvjpegHandle_t
}

//JpegState is Opaque jpeg decoding state handle identifier - used to store intermediate information between deccding phases
type JpegState struct {
	j C.nvjpegJpegState_t
}

// Create  creates the nvjpeg handle. This handle is used for all consecutive calls
// IN         backend       : Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.
// IN         allocator     : Pointer to nvjpegDevAllocator. If NULL - use default cuda calls (cudaMalloc/cudaFree)
// OUT        handle        : Codec instance, use for other calls
func Create(b Backend, allocator *DevAllocator) (*Handle, error) {
	h := new(Handle)
	err := status(C.nvjpegCreate(b.c(), allocator.cptr(), &h.h)).error()
	if err != nil {
		return nil, err
	}

	runtime.SetFinalizer(h, nvjpegDestroy)
	return h, err
}
func nvjpegDestroy(h *Handle) error {
	err := status(C.nvjpegDestroy(h.h)).error()
	if err != nil {
		return err
	}
	h = nil
	return nil
}

//JpegStateCreate creates an initialized decode state
func JpegStateCreate(h *Handle) (*JpegState, error) {
	j := new(JpegState)
	err := status(C.nvjpegJpegStateCreate(h.h, &j.j)).error()
	if err != nil {
		return nil, err
	}

	runtime.SetFinalizer(j, nvjpegJpegStateDestroy)
	return j, nil
}

func nvjpegJpegStateDestroy(j *JpegState) error {
	err := status(C.nvjpegJpegStateDestroy(j.j)).error()
	if err != nil {
		return err
	}
	j = nil
	return nil
}

// GetImageInfo gets the image info
// Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
// If less than NVJPEG_MAX_COMPONENT channels are encoded, then zeros would be set to absent channels information
// If the image is 3-channel, all three groups are valid.
// This function is thread safe.
// IN         handle      : Library handle
// IN         data        : Pointer to the buffer containing the jpeg stream data to be decoded.
// IN         length      : Length of the jpeg image buffer.
// Return     nComponent  : Number of componenets of the image, currently only supports 1-channel (grayscale) or 3-channel.
// Return     subsampling : Chroma subsampling used in this JPEG, see nvjpegChromaSubsampling_t
// Return     widths      : pointer to NVJPEG_MAX_COMPONENT of ints, returns width of each channel. 0 if channel is not encoded
// Return     heights     : pointer to NVJPEG_MAX_COMPONENT of ints, returns height of each channel. 0 if channel is not encoded
func GetImageInfo(handle *Handle, data *byte, length uint) (nComponents int32, subsampling ChromaSubsampling, width int32, height int32, err error) {
	var ncomp C.int
	var sub C.nvjpegChromaSubsampling_t
	var w C.int
	var h C.int
	d := (*C.uchar)(data)
	err = status(C.nvjpegGetImageInfo(
		handle.h,
		d,
		C.size_t(length),
		&ncomp,
		&sub,
		&w,
		&h)).error()
	subsampling = ChromaSubsampling(sub)
	width = int32(w)
	height = int32(h)
	nComponents = int32(ncomp)
	return nComponents, subsampling, width, height, err
}

// Decode does the nvjpegDecode
// Decodes single image. Destination buffers should be large enough to be able to store
// output of specified format. For each color plane sizes could be retrieved for image using nvjpegGetImageInfo()
// and minimum required memory buffer for each plane is nPlaneHeight*nPlanePitch where nPlanePitch >= nPlaneWidth for
// planar output formats and nPlanePitch >= nPlaneWidth*nOutputComponents for interleaved output format.
//
// IN/OUT     handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
// IN         data          : Pointer to the buffer containing the jpeg image to be decoded.
// IN         length        : Length of the jpeg image buffer.
// IN         output_format : Output data format. See nvjpegOutputFormat_t for description
// IN/OUT     destination   : Pointer to structure with information about output buffers. See nvjpegImage_t description.
// IN/OUT     stream        : CUDA stream where to submit all GPU work
//
// \return NVJPEG_STATUS_SUCCESS if successful
func Decode(h *Handle, j *JpegState, data []byte, frmt OutputFormat, dest *Image, stream *gocudnn.Stream) error {
	d := (*C.uchar)(&data[0])
	length := len(data)
	return status(C.nvjpegDecode(h.h, j.j, d, C.size_t(length), frmt.c(), dest.cptr(), C.cudaStream_t(stream.Ptr()))).error()
}

//Decode in Phases

// Same functionality and parameters as for nvjpegDecodePlanar, but separated in steps:
// 1) CPU processing
// 2) Mixed processing that requires interaction of both GPU and CPU. Any previous call
// to nvjpegDecodeGPU() with same handle should be finished before this call, i.e. cudaStreamSycnhronize() could be used
// 3) GPU processing
// Actual amount of work done in each separate step depends on the selected backend. But in any way all
// of those functions must be called in this specific order. If one of the steps returns error - decode should be done from the beginning.

//DecodePhaseOne - CPU processing
func DecodePhaseOne(h *Handle, j *JpegState, data []byte, frmt OutputFormat, stream *gocudnn.Stream) error {
	d := (*C.uchar)(&data[0])
	length := len(data)
	return status(C.nvjpegDecodePhaseOne(h.h, j.j, d, C.size_t(length), frmt.c(), C.cudaStream_t(stream.Ptr()))).error()
}

//DecodePhaseTwo -  Mixed processing that requires interaction of both GPU and CPU. Any previous call
// to nvjpegDecodeGPU() with same handle should be finished before this call, i.e. cudaStreamSycnhronize() could be used
func DecodePhaseTwo(h *Handle, j *JpegState, stream *gocudnn.Stream) error {
	return status(C.nvjpegDecodePhaseTwo(h.h, j.j, C.cudaStream_t(stream.Ptr()))).error()
}

//DecodePhaseThree GPU processing
// Actual amount of work done in each separate step depends on the selected backend. But in any way all
// of those functions must be called in this specific order. If one of the steps returns error - decode should be done from the beginning.
func DecodePhaseThree(h *Handle, j *JpegState, dest *Image, stream *gocudnn.Stream) error {
	return status(C.nvjpegDecodePhaseThree(h.h, j.j, dest.cptr(), C.cudaStream_t(stream.Ptr()))).error()
}

//////////////////////////////////////////////
/////////////// Batch decoding ///////////////
//////////////////////////////////////////////

// DecodeBatchedInitialize - Resets and initizlizes batch decoder for working on the batches of specified size
// Should be called once for decoding bathes of this specific size, also use to reset failed batches
// IN/OUT     handle          : Library handle
// INT/OUT    jpeg_handle     : Decoded jpeg image state handle
// IN         batch_size      : Size of the batch
// IN         max_cpu_threads : Maximum number of CPU threads that will be processing this batch
// IN         output_format   : Output data format. Will be the same for every image in batch
//
// \return NVJPEG_STATUS_SUCCESS if successful
func DecodeBatchedInitialize(h *Handle, j *JpegState, batchsize, maxCPUthreads int, frmt OutputFormat) error {
	return status(C.nvjpegDecodeBatchedInitialize(h.h, j.j, C.int(batchsize), C.int(maxCPUthreads), frmt.c())).error()
}

//DecodeBatched - Decodes batch of images. Output buffers should be large enough to be able to store
// outputs of specified format, see single image decoding description for details. Call to
// nvjpegDecodeBatchedInitialize() is required prior to this call, batch size is expected to be the same as
// parameter to this batch initialization function.
//
// IN/OUT     handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
// IN         data          : Array of size batch_size of pointers to the input buffers containing the jpeg images to be decoded.
// IN         lengths       : Array of size batch_size with lengths of the jpeg images' buffers in the batch.
// IN/OUT     destinations  : Array of size batch_size with pointers to structure with information about output buffers,
// IN/OUT     stream        : CUDA stream where to submit all GPU work
//
// \return NVJPEG_STATUS_SUCCESS if successful
func DecodeBatched(h *Handle, j *JpegState, data [][]byte, dest []*Image, stream *gocudnn.Stream) error {
	x := make([]*C.uchar, len(data))
	y := make([]C.size_t, len(data))
	z := make([]*C.nvjpegImage_t, len(dest))
	var length int
	for i := range data {
		length = len(data[i])
		x[i] = (*C.uchar)(&data[i][0])
		y[i] = C.size_t(length)
		z[i] = dest[i].cptr()
	}

	return status(C.nvjpegDecodeBatched(h.h, j.j, &x[0], &y[0], z[0], C.cudaStream_t(stream.Ptr()))).error()
}

//Phased Decoding Batches.

// Same functionality as nvjpegDecodePlanarBatched but done in separate consecutive steps:
// 1) nvjpegDecodePlanarBatchedCPU should be called [batch_size] times for each image in batch.
// This function is thread safe and could be called by multiple threads simultaneously, by providing
// thread_idx (thread_idx should be less than max_cpu_threads from nvjpegDecodeBatchedInitialize())
// 2) nvjpegDecodePlanarBatchedMixed. Any previous call to nvjpegDecodeBatchedGPU() should be done by this point
// 3) nvjpegDecodePlanarBatchedGPU
// Actual amount of work done in each separate step depends on the selected backend. But in any way all
// of those functions must be called in this specific order. If one of the steps returns error -
// reset batch with nvjpegDecodeBatchedInitialize().

//DecodeBatchedPhaseOne - nvjpegDecodePlanarBatchedCPU should be called [batch_size] times for each image in batch.
// This function is thread safe and could be called by multiple threads simultaneously, by providing
// thread_idx (thread_idx should be less than max_cpu_threads from nvjpegDecodeBatchedInitialize())
func DecodeBatchedPhaseOne(h *Handle, j *JpegState, data []byte, length uint, imageidx, threadidx int, stream *gocudnn.Stream) error {

	return status(C.nvjpegDecodeBatchedPhaseOne(h.h, j.j, (*C.uchar)(&data[0]), C.size_t(len(data)), C.int(imageidx), C.int(threadidx), C.cudaStream_t(stream.Ptr()))).error()
}

//DecodeBatchedPhaseTwo - nvjpegDecodePlanarBatchedMixed. Any previous call to nvjpegDecodeBatchedGPU() should be done by this point
func DecodeBatchedPhaseTwo(h *Handle, j *JpegState, stream *gocudnn.Stream) error {
	return status(C.nvjpegDecodeBatchedPhaseTwo(h.h, j.j, C.cudaStream_t(stream.Ptr()))).error()
}

//DecodeBatchedPhaseThree - nvjpegDecodePlanarBatchedGPU
func DecodeBatchedPhaseThree(h *Handle, j *JpegState, dest []*Image, stream *gocudnn.Stream) error {
	z := make([]*C.nvjpegImage_t, len(dest))
	for i := range z {

		z[i] = dest[i].cptr()
	}
	return status(C.nvjpegDecodeBatchedPhaseThree(h.h, j.j, z[0], C.cudaStream_t(stream.Ptr()))).error()
}

//LibraryPropertyType are flags for finding the library major, minor,patch
type LibraryPropertyType C.libraryPropertyType

func (l LibraryPropertyType) c() C.libraryPropertyType {
	return C.libraryPropertyType(l)
}

//LibraryPropertyTypeFlag passes LibraryPropertyType flags through methods
type LibraryPropertyTypeFlag struct {
}

//Major passes the Major flag
func (l LibraryPropertyTypeFlag) Major() LibraryPropertyType {
	return LibraryPropertyType(C.MAJOR_VERSION)
}

//Minor passes the minor flag
func (l LibraryPropertyTypeFlag) Minor() LibraryPropertyType {
	return LibraryPropertyType(C.MINOR_VERSION)
}

//Patch passes the patch flag
func (l LibraryPropertyTypeFlag) Patch() LibraryPropertyType {
	return LibraryPropertyType(C.PATCH_LEVEL)
}

// GetProperty returns library's property values, such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
func GetProperty(ltype LibraryPropertyType) (int, error) {
	var x C.int
	err := status(C.nvjpegGetProperty(ltype.c(), &x)).error()
	return int(x), err
}
