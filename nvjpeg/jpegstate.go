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

//CreateJpegState creates an initialized decode state
func CreateJpegState(h *Handle) (*JpegState, error) {
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

/*Decode does the nvjpegDecode

  Decodes single image. Destination buffers should be large enough to be able to store
  output of specified format. For each color plane sizes could be retrieved for image using nvjpegGetImageInfo()
  and minimum required memory buffer for each plane is nPlaneHeight*nPlanePitch where nPlanePitch >= nPlaneWidth for
  planar output formats and nPlanePitch >= nPlaneWidth*nOutputComponents for interleaved output format.

  Function will perform an s.Sync() before returning.

  IN/OUT     h             : Library handle

  IN         data          : Pointer to the buffer containing the jpeg image to be decoded.

  IN         fmt           : Output data format. See nvjpegOutputFormat_t for description

  IN/OUT     dest	 	 : Pointer to structure with information about output buffers. See nvjpegImage_t description.

  IN/OUT     s             : gocu.Streamer where to submit all GPU work

*/
func (j *JpegState) Decode(h *Handle, data []byte, frmt OutputFormat, dest *Image, s gocu.Streamer) error {
	d := (*C.uchar)(&data[0])
	length := len(data)
	err := status(C.nvjpegDecode(h.h, j.j, d, C.size_t(length), frmt.c(), dest.cptr(), stream(s))).error()
	if err != nil {
		return err
	}
	return s.Sync()
}

/*DecodePhase1 - CPU processing

If error is received restart at phase one

Function will not call an s.Sync(). It will be advised to sync your stream before you go to DecodePhase2.
*/
func (j *JpegState) DecodePhase1(h *Handle, data []byte, frmt OutputFormat, s gocu.Streamer) error {
	d := (*C.uchar)(&data[0])
	length := len(data)
	return status(C.nvjpegDecodePhaseOne(h.h, j.j, d, C.size_t(length), frmt.c(), stream(s))).error()

}

/*DecodePhase2 -  Mixed processing that requires interaction of both GPU and CPU.

Any previous call DecodeGPU() with same handle should be finished before this call, i.e Sync() using a gocu.Streamer.

DecodePhase1 must be ran before DecodePhase2.

If error is received restart at phase one.

Function will not call an s.Sync(). It will be advised to sync your stream before you go to DecodePhase3.
*/
func (j *JpegState) DecodePhase2(h *Handle, s gocu.Streamer) error {
	return status(C.nvjpegDecodePhaseTwo(h.h, j.j, stream(s))).error()

}

/*DecodePhase3 - GPU processing.

DecodePhase2() must be ran before DecodePhase3().

If error is received restart at phase one.

Function will not call an s.Sync(). It will be advised to sync your stream before you do anything with the dest image.
*/
func (j *JpegState) DecodePhase3(h *Handle, dest *Image, s gocu.Streamer) error {
	return status(C.nvjpegDecodePhaseThree(h.h, j.j, dest.cptr(), stream(s))).error()

}

//////////////////////////////////////////////
/////////////// Batch decoding ///////////////
//////////////////////////////////////////////

/*DecodeBatchedInitialize - Resets and initizlizes batch decoder for working on the batches of specified size
Should be called once for decoding batches of this specific size, also use to reset failed batches

IN/OUT     h             : Library handle

IN         batchsize     : Size of the batch

IN         maxCPUthreads : Maximum number of CPU threads that will be processing this batch

IN         frmt          : Output data format. Will be the same for every image in batch
*/
func (j *JpegState) DecodeBatchedInitialize(h *Handle, batchsize, maxCPUthreads int, frmt OutputFormat) error {
	return status(C.nvjpegDecodeBatchedInitialize(h.h, j.j, C.int(batchsize), C.int(maxCPUthreads), frmt.c())).error()
}

/*DecodeBatched - Decodes batch of images.
Output buffers should be large enough to be able to store
outputs of specified format, see single image decoding description for details. Call to
nvjpegDecodeBatchedInitialize() is required prior to this call, batch size is expected to be the same as
parameter to this batch initialization function.

IN/OUT     h             : Library handle

IN         data          : Slice of byte slices of input buffers containing the jpeg images to be decoded.

IN/OUT     destinations  : Array of pointers to structure with information about output buffers. len(dest) == len(data)

IN/OUT     s             : gocu.Streamer where to submit all GPU work

Function will perform an s.Sync() before returning.
*/
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

	err := status(C.nvjpegDecodeBatched(h.h, j.j, &x[0], &y[0], z[0], stream(s))).error()
	if err != nil {
		return err
	}
	return s.Sync()
}

/*DecodeBatchedPhase1 - nvjpegDecodePlanarBatchedCPU should be called [batch_size] times for each image in batch.
This function is thread safe and could be called by multiple threads simultaneously, by providing
thread_idx (thread_idx should be less than max_cpu_threads from nvjpegDecodeBatchedInitialize())
If error is received restart at phase one.

Function will perform an s.Sync() before returning.
*/
func (j *JpegState) DecodeBatchedPhase1(h *Handle, data []byte, imageidx, threadidx int, s gocu.Streamer) error {
	err := status(C.nvjpegDecodeBatchedPhaseOne(h.h, j.j, (*C.uchar)(&data[0]), C.size_t(len(data)), C.int(imageidx), C.int(threadidx), stream(s))).error()
	if err != nil {
		return err
	}
	return s.Sync()
}

/*DecodeBatchedPhase2 - nvjpegDecodePlanarBatchedMixed. Any previous call to nvjpegDecodeBatchedGPU() should be done by this point

If error is received restart at phase one.

Function will perform an s.Sync() before returning.
*/
func (j *JpegState) DecodeBatchedPhase2(h *Handle, s gocu.Streamer) error {
	err := status(C.nvjpegDecodeBatchedPhaseTwo(h.h, j.j, stream(s))).error()
	if err != nil {
		return err
	}
	return s.Sync()
}

/*DecodeBatchedPhase3 - nvjpegDecodePlanarBatchedGPU

If error is received restart at phase one.

Function will perform an s.Sync() before returning.
*/
func (j *JpegState) DecodeBatchedPhase3(h *Handle, dest []*Image, s gocu.Streamer) error {
	z := make([]*C.nvjpegImage_t, len(dest))
	for i := range z {

		z[i] = dest[i].cptr()
	}
	err := status(C.nvjpegDecodeBatchedPhaseThree(h.h, j.j, z[0], stream(s))).error()
	if err != nil {
		return err
	}
	return s.Sync()
}
