package nvjpeg

/*
#include <nvjpeg.h>
#include <cuda_runtime_api.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//JpegState is Opaque jpeg decoding state handle identifier - used to store intermediate information between deccding phases
type JpegState struct {
	j C.nvjpegJpegState_t
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
func (j *JpegState) Decode(h *Handle, data []byte, frmt OutputFormat, dest *Image, s gocu.Streamer) error {
	d := (*C.uchar)(&data[0])
	length := len(data)
	return status(C.nvjpegDecode(h.h, j.j, d, C.size_t(length), frmt.c(), dest.cptr(), stream(s))).error()
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
func (j *JpegState) DecodePhaseOne(h *Handle, data []byte, frmt OutputFormat, s gocu.Streamer) error {
	d := (*C.uchar)(&data[0])
	length := len(data)
	return status(C.nvjpegDecodePhaseOne(h.h, j.j, d, C.size_t(length), frmt.c(), stream(s))).error()
}

//DecodePhaseTwo -  Mixed processing that requires interaction of both GPU and CPU. Any previous call
// to nvjpegDecodeGPU() with same handle should be finished before this call, i.e. cudaStreamSycnhronize() could be used
func (j *JpegState) DecodePhaseTwo(h *Handle, s gocu.Streamer) error {
	return status(C.nvjpegDecodePhaseTwo(h.h, j.j, stream(s))).error()
}

//DecodePhaseThree GPU processing
// Actual amount of work done in each separate step depends on the selected backend. But in any way all
// of those functions must be called in this specific order. If one of the steps
// returns error - decode should be done from the beginning.
func (j *JpegState) DecodePhaseThree(h *Handle, dest *Image, s gocu.Streamer) error {
	return status(C.nvjpegDecodePhaseThree(h.h, j.j, dest.cptr(), stream(s))).error()
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
func (j *JpegState) DecodeBatchedInitialize(h *Handle, batchsize, maxCPUthreads int, frmt OutputFormat) error {
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
func (j *JpegState) DecodeBatched(h *Handle, data [][]byte, dest []*Image, s gocu.Streamer) error {
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

	return status(C.nvjpegDecodeBatched(h.h, j.j, &x[0], &y[0], z[0], stream(s))).error()
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
func (j *JpegState) DecodeBatchedPhaseOne(h *Handle, data []byte, length uint, imageidx, threadidx int, s gocu.Streamer) error {

	return status(C.nvjpegDecodeBatchedPhaseOne(h.h, j.j, (*C.uchar)(&data[0]), C.size_t(len(data)), C.int(imageidx), C.int(threadidx), stream(s))).error()
}

//DecodeBatchedPhaseTwo - nvjpegDecodePlanarBatchedMixed. Any previous call to nvjpegDecodeBatchedGPU() should be done by this point
func (j *JpegState) DecodeBatchedPhaseTwo(h *Handle, s gocu.Streamer) error {
	return status(C.nvjpegDecodeBatchedPhaseTwo(h.h, j.j, stream(s))).error()
}

//DecodeBatchedPhaseThree - nvjpegDecodePlanarBatchedGPU
func (j *JpegState) DecodeBatchedPhaseThree(h *Handle, dest []*Image, s gocu.Streamer) error {
	z := make([]*C.nvjpegImage_t, len(dest))
	for i := range z {

		z[i] = dest[i].cptr()
	}
	return status(C.nvjpegDecodeBatchedPhaseThree(h.h, j.j, z[0], stream(s))).error()
}
