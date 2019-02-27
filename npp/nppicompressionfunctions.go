package npp

//#include <nppi_compression_functions.h>
//#include <nppdefs.h>
import "C"
import (
	"runtime"
)

type DCTState struct {
	state *C.NppiDCTState
}
type DecodeHuffmanSpec C.NppiDecodeHuffmanSpec
type EncodeHuffmanSpec C.NppiEncodeHuffmanSpec
type JpegFrameDescr C.NppiJpegFrameDescr

//typedef struct {
//Npp8u nComponents; /**< Number of components in frame */
//NppiSize oSizeInBlocks; /**< Size of component with 1x1 subsampling (usually luma) in DCT blocks. */
//NppiSize aComponentSubsampling[4]; /**< Subsampling factors of component, as described in frame header */
//Npp16s * apComponentBuffer[4];
/**<
  Buffer containing DCT coefficients. Use \ref nppiJpegDecodeGetDCTBufferSize to
  determine size of this buffer. After decoding, coefficients will be stored in
  zig-zag order, block by block. So the c-th coeffient of block `(x, y)` will
  be stored at `buffer[64 * (y * interleavedComponentWidthInBlocks + x) + c]`.
*/
//} NppiJpegFrameDescr;

type JpegScanDescr C.struct_NppiJpegScanDescr

/*
/// JPEG scan descriptor
typedef struct {
    Npp8u nComponents;  /// Number of components present in scan
    Npp8u aComponentIdx[4]; /// Frame-indexes of components. These values will be used to index arrays in \ref NppiJpegFrameDescr
    Npp8u aComponentDcHtSel[4]; /// DC Huffman table selector per component
    Npp8u aComponentAcHtSel[4]; /// AC Huffman table selector per component
    const Npp8u * apRawDcHtTable[4]; /// Pointers to DC Huffman table description in the raw format (the same format as used in JPEG header). This array will be indexed by \ref aComponentDcHtSel. Pointers for tables unused in scan may be set to NULL.
    const Npp8u * apRawAcHtTable[4]; /// See \ref apRawDcHtTable
    Npp8u nSs; /// Start of spectral selection (index of first coefficient), 0-63
    Npp8u nSe; /// End of spectral selection (index of first coefficient), 0-63
    Npp8u nAh; /// Successive approximation bit position high
    Npp8u nAl; /// Successive approximation bit position low
    Npp32s restartInterval; /// Restart interval in MCUs. Use 0 or -1 when none
    Npp32s length; /// Length of compressed (encoded) scan data
} NppiJpegScanDescr;
*/

/**
 * JPEG decode job used by \ref nppiJpegDecodeJob (see that for more documentation)
 *
 * The job describes piece of computation to be done.
 */
type JpegDecodeJob C.struct_NppiJpegDecodeJob

/*
typedef struct {
    NppiJpegFrameDescr * pFrame; // This field and its contents are never written
    NppiJpegScanDescr * pScan;  //This field is never written. `*pScan` is written only by ...Create... functions
    enum NppiJpegDecodeJobKind eKind;
} NppiJpegDecodeJob;

*/

/*

enum NppiJpegDecodeJobKind {
   NPPI_JPEG_DECODE_SIMPLE, // Decode whole scan using a single job

   // SIMPLE can be split into:

   NPPI_JPEG_DECODE_PRE, // Preprocessing scan on GPU
   NPPI_JPEG_DECODE_CPU, // Part of decoding run on CPU
   NPPI_JPEG_DECODE_GPU, // Part of decoding run on GPU


   NPPI_JPEG_DECODE_MEMZERO, // Zeroing memory before decoding
   NPPI_JPEG_DECODE_FINALIZE // Change memory representation of DCT coefficients to final

};
*/

/** Number of additional buffers that may be used by JPEG decode jobs.
 * This number may change in the future, but it remain small.
 *
 * \sa NppiJpegDecodeJobMemory
 */
//#define NPPI_JPEG_DECODE_N_BUFFERS 3

/**
 * Memory buffers used by one decode job.
 *
 * \sa nppiJpegDecodeJobMemorySize
 * \sa nppiJpegDecodeJob
 */

type JpegDecodeJobMemory C.struct_NppiJpegDecodeJobMemory

/*
typedef struct {
   const Npp8u * pCpuScan;
   // < Pointer to host memory containing compressed scan data.
   // * Should be allocated with additional \ref nppiJpegDecodeGetScanDeadzoneSize
   // * bytes of usable memory after the end of compressed scan data.
   // * Should be filled by caller.

   Npp8u * pGpuScan;
   // Pointer to device memory used for compressed scan data.
   // * Should be allocated with additional \ref nppiJpegDecodeGetScanDeadzoneSize
   // * bytes of usable memory after the end of compressed scan data.
   // * Should be filled by caller.
   // * This buffer may be overwritten by the decoder.
   // * Could be NULL for \ref NPPI_JPEG_DECODE_CPU.

   void * apCpuBuffer[NPPI_JPEG_DECODE_N_BUFFERS];
   // Pointers to additional host buffers used by job. Call \ref nppiJpegDecodeJobMemorySize
   // * to query sizes of these buffers. `apCpuBuffer[i]` should point to
   // * at least `aSize[i]` bytes. If `aSize[i] == 0`, the pointer should be set to NULL.

   void * apGpuBuffer[NPPI_JPEG_DECODE_N_BUFFERS];
   // Pointers to additional device buffers used by job.
   // * Minimal sizes of buffers should be the same as the sizes of \ref apCpuBuffer.
} NppiJpegDecodeJobMemory;
*/

/**
 * Type of job to execute. Usually you will need just SIMPLE
 * for each scan, one MEMZERO at the beginning and FINALIZE at the end.
 * See the example in \ref nppiJpegDecodeJob
 *
 * SIMPLE can be split into multiple jobs: PRE, CPU & GPU.
 * Please note that if you don't use SIMPLE,
 * you man need to add some memcopies and synchronizes as
 * described in \ref nppiJpegDecodeJob.
 *
 * \sa nppiJpegDecodeJob
 */

/*QuantFwdRawTableInitJPEG8u Apply quality factor to raw 8-bit quantization table.
 *
 * This is effectively and in-place method that modifies a given raw
 * quantization table based on a quality factor.
 * Note that this method is a host method and that the pointer to the
 * raw quantization table is a host pointer.
 *
 * \param hpQuantRawTable Raw quantization table.
 * \param nQualityFactor Quality factor for the table. Range is [1:100].
 * \return Error code:
 *      ::NPP_NULL_POINTER_ERROR is returned if hpQuantRawTable is 0.
 */
func QuantFwdRawTableInitJPEG8u(hpQuantRawTable []Uint8, nQualityFactor int32) ([]Uint8, error) {
	//x := convertNpp8utoCNpp8uarray(hpQuantRawTable)
	err := status(C.nppiQuantFwdRawTableInit_JPEG_8u((*C.uchar)(&hpQuantRawTable[0]), (C.int)(nQualityFactor))).ToError()
	//	y := convertCNpp8utoNpp8uarray(x)
	return hpQuantRawTable, err
}

/* QuantFwdTableInitJPEG8u16u Initializes a quantization table for nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R().
 *    The method creates a 16-bit version of the raw table and converts the
 * data order from zigzag layout to original row-order layout since raw
 * quantization tables are typically stored in zigzag format.
 *
 * This method is a host method. It consumes and produces host data. I.e. the pointers
 * passed to this function must be host pointers. The resulting table needs to be
 * transferred to device memory in order to be used with nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R()
 * function.
 *
 * \param hpQuantRawTable Host pointer to raw quantization table as returned by
 *      nppiQuantFwdRawTableInit_JPEG_8u(). The raw quantization table is assumed to be in
 *      zigzag order.
 * \param hpQuantFwdRawTable Forward quantization table for use with nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R().
 * \return Error code:
 *      ::NPP_NULL_POINTER_ERROR pQuantRawTable is 0.
 */
/*
func sizetoCNppiSize(oSizeROI NppiSize) C.NppiSize {
	w, h := oSizeROI.WidthHeight()
	var x C.NppiSize
	x.width = C.int(w)
	x.height = C.int(h)
	return x
}
*/

//QuantFwdTableInitJPEG8u16u initializes a table for uint8 to uint16
func QuantFwdTableInitJPEG8u16u(hpQuantRawTable []Uint8) ([]Uint16, error) {
	y := make([]Uint16, len(hpQuantRawTable))
	err := status(C.nppiQuantFwdTableInit_JPEG_8u16u((*C.Npp8u)(&hpQuantRawTable[0]), (*C.ushort)((&y[0])))).ToError()
	return y, err
}

func DCTQuantFwd8x8LSJPEG8u16sC1R(pSrc *Uint8, nSrcStep int32, pDst *Int16, nDstStep int32, pQuantFwdTable *Uint16, oSizeROI Size) error {
	w, h := oSizeROI.Get()
	var x C.NppiSize
	x.width = C.int(w)
	x.height = C.int(h)

	return status(C.nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R((*C.Npp8u)(pSrc), (C.int)(nSrcStep), (*C.Npp16s)(pDst), (C.int)(nDstStep), (*C.Npp16u)(pQuantFwdTable), x)).ToError()
}

func DCTQuantInv8x8LSJPEG16s8uC1R(pSrc *Int16, nSrcStep int32, pDst *Uint8, nDstStep int32, pQuantInvTable *Uint16, oSizeROI Size) error {

	return status(C.nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R((*C.Npp16s)(pSrc), (C.int)(nSrcStep), (*C.Npp8u)(pDst), (C.int)(nDstStep), (*C.Npp16u)(pQuantInvTable), oSizeROI.c())).ToError()
}

func DCTInitAlloc() (*DCTState, error) {
	var x *C.NppiDCTState
	err := status(C.nppiDCTInitAlloc(&x)).ToError()
	y := &DCTState{
		state: x,
	}
	runtime.SetFinalizer(y, freeNPPIDCTstate)
	return y, err
}
func freeNPPIDCTstate(x *DCTState) error {
	return status(C.nppiDCTFree(x.state)).ToError()
}

/*

func NPPIDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW( Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep,  Npp8u * pQuantizationTable, NppiSize oSizeROI, NppiDCTState* pState) error{
    return status(C.nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW()).ToError()
}
func NPPIDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW( Npp16s * pSrc, int nSrcStep,Npp8u  * pDst, int nDstStep, Npp8u * pQuantizationTable, NppiSize oSizeROI,NppiDCTState* pState) error{
    return status(C.nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW()).ToError()
}
func NPPIDCTQuant16Fwd8x8LS_JPEG_8u16s_C1R_NEW( Npp8u  * pSrc, int nSrcStep,Npp16s * pDst, int nDstStep, Npp16u * pQuantizationTable, NppiSize oSizeROI,NppiDCTState* pState) error{
    return status(C.nppiDCTQuant16Fwd8x8LS_JPEG_8u16s_C1R_NEW()).ToError()
}
func NPPIDCTQuant16Inv8x8LS_JPEG_16s8u_C1R_NEW( Npp16s * pSrc, int nSrcStep,Npp8u  * pDst, int nDstStep, Npp16u * pQuantizationTable, NppiSize oSizeROI,NppiDCTState* pState) error{
    return status(C.nppiDCTQuant16Inv8x8LS_JPEG_16s8u_C1R_NEW()).ToError()
}
func NPPIDecodeHuffmanSpecGetBufSize_JPEG(int* pSize) error{
    return status(C.nppiDecodeHuffmanSpecGetBufSize_JPEG()).ToError()
}
func NPPIDecodeHuffmanSpecInitHost_JPEG( Npp8u* pRawHuffmanTable, NppiHuffmanTableType eTableType, NppiDecodeHuffmanSpec  *pHuffmanSpec) error{
    return status(C.nppiDecodeHuffmanSpecInitHost_JPEG()).ToError()
}
func NPPIDecodeHuffmanSpecInitAllocHost_JPEG( Npp8u* pRawHuffmanTable, NppiHuffmanTableType eTableType, NppiDecodeHuffmanSpec  **ppHuffmanSpec) error{
    return status(C.nppiDecodeHuffmanSpecInitAllocHost_JPEG()).ToError()
}
func NPPIDecodeHuffmanSpecFreeHost_JPEG(NppiDecodeHuffmanSpec  *pHuffmanSpec) error{
    return status(C.nppiDecodeHuffmanSpecFreeHost_JPEG()).ToError()
}
func NPPIDecodeHuffmanScanHost_JPEG_8u16s_P1R( Npp8u  * pSrc, Npp32s nLength,Npp32s restartInterval, Npp32s Ss, Npp32s Se, Npp32s Ah, Npp32s Al,Npp16s * pDst, Npp32s nDstStep,NppiDecodeHuffmanSpec  * pHuffmanTableDC, NppiDecodeHuffmanSpec  * pHuffmanTableAC,NppiSize oSizeROI); error{
    return status(C.nppiDecodeHuffmanScanHost_JPEG_8u16s_P1R()).ToError()
}
func NPPIDecodeHuffmanScanHost_JPEG_8u16s_P3R( Npp8u * pSrc, Npp32s nLength,Npp32s nRestartInterval, Npp32s nSs, Npp32s nSe, Npp32s nAh, Npp32s nAl,Npp16s * apDst[3], Npp32s aDstStep[3],NppiDecodeHuffmanSpec * apHuffmanDCTable[3], NppiDecodeHuffmanSpec * apHuffmanACTable[3], NppiSize aSizeROI[3]) error{
    return status(C.nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R()).ToError()
}
func NPPIEncodeHuffmanSpecGetBufSize_JPEG(int* pSize) error{
    return status(C.nppiEncodeHuffmanSpecGetBufSize_JPEG()).ToError()
}
func NPPIEncodeHuffmanSpecInit_JPEG( Npp8u* pRawHuffmanTable, NppiHuffmanTableType eTableType, NppiEncodeHuffmanSpec  *pHuffmanSpec) error{
    return status(C.nppiEncodeHuffmanSpecInit_JPEG()).ToError()
}
func NPPIEncodeHuffmanSpecInitAlloc_JPEG( Npp8u* pRawHuffmanTable, NppiHuffmanTableType eTableType, NppiEncodeHuffmanSpec  **ppHuffmanSpec) error{
    return status(C.nppiEncodeHuffmanSpecInitAlloc_JPEG()).ToError()
}
func NPPIEncodeHuffmanSpecFree_JPEG(NppiEncodeHuffmanSpec  *pHuffmanSpec) error{
    return status(C.nppiEncodeHuffmanSpecFree_JPEG()).ToError()
}
func NPPIEncodeHuffmanScan_JPEG_8u16s_P1R( Npp16s * pSrc, Npp32s nSrcStep,Npp32s nRestartInterval, Npp32s nSs, Npp32s nSe, Npp32s nAh, Npp32s nAl,Npp8u  * pDst, Npp32s* nLength,NppiEncodeHuffmanSpec  * pHuffmanTableDC, NppiEncodeHuffmanSpec  * pHuffmanTableAC, NppiSize oSizeROI,Npp8u* pTempStorage); error{
    return status(C.nppiEncodeHuffmanScan_JPEG_8u16s_P1R()).ToError()
}
func NPPIEncodeHuffmanScan_JPEG_8u16s_P3R(Npp16s * apSrc[3], Npp32s aSrcStep[3],Npp32s nRestartInterval, Npp32s nSs, Npp32s nSe, Npp32s nAh, Npp32s nAl,Npp8u  * pDst, Npp32s* nLength,NppiEncodeHuffmanSpec * apHuffmanDCTable[3], NppiEncodeHuffmanSpec * apHuffmanACTable[3], NppiSize aSizeROI[3],Npp8u* pTempStorage) error{
    return status(C.nppiEncodeHuffmanScan_JPEG_8u16s_P3R()).ToError()
}
func NPPIEncodeOptimizeHuffmanScan_JPEG_8u16s_P1R( Npp16s * pSrc, Npp32s nSrcStep, Npp32s nRestartInterval, Npp32s nSs, Npp32s nSe, Npp32s nAh, Npp32s nAl, Npp8u * pDst, Npp32s * pLength, Npp8u * hpCodesDC, Npp8u * hpTableDC,Npp8u * hpCodesAC, Npp8u * hpTableAC, NppiEncodeHuffmanSpec * pHuffmanDCTable, NppiEncodeHuffmanSpec * pHuffmanACTable,NppiSize oSizeROI, Npp8u * pTempStorage) error{
    return status(C.nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P1R()).ToError()
}
func NPPIEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R(Npp16s * apSrc[3], Npp32s aSrcStep[3], Npp32s nRestartInterval, Npp32s nSs, Npp32s nSe, Npp32s nAh, Npp32s nAl, Npp8u * pDst, Npp32s * pLength, Npp8u * hpCodesDC[3], Npp8u * hpTableDC[3], Npp8u * hpCodesAC[3], Npp8u * hpTableAC[3], NppiEncodeHuffmanSpec * apHuffmanDCTable[3], NppiEncodeHuffmanSpec * apHuffmanACTable[3],  NppiSize oSizeROI[3], Npp8u * pTempStorage) error{
    return status(C.nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R()).ToError()
}
func NPPIEncodeHuffmanGetSize(NppiSize oSize, int nChannels, size_t * pBufSize) error{
    return status(C.nppiEncodeHuffmanGetSize()).ToError()
}
func NPPIEncodeOptimizeHuffmanGetSize(NppiSize oSize, int nChannels, int * pBufSize) error{
    return status(C.nppiEncodeOptimizeHuffmanGetSize()).ToError()
}
func NPPIJpegDecodeJobMemorySize( NppiJpegDecodeJob * pJob, size_t * aSize) error{
    return status(C.nppiJpegDecodeJobMemorySize()).ToError()
}
func NPPIJpegDecodeJob( NppiJpegDecodeJob * pJob,  NppiJpegDecodeJobMemory * pMemory) error{
    return status(C.nppiJpegDecodeJob()).ToError()
}
func NPPIJpegDecodeJobCreateMemzero(NppiJpegDecodeJob * pJob) error{
    return status(C.nppiJpegDecodeJobCreateMemzero()).ToError()
}
func NPPIJpegDecodeJobCreateFinalize(NppiJpegDecodeJob * pJob) error{
    return status(C.nppiJpegDecodeJobCreateFinalize()).ToError()
}
func NPPIDCTInv4x4_WebP_16s_C1R( Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI) error{
    return status(C.nppiDCTInv4x4_WebP_16s_C1R()).ToError()
}


size_t NPPIJpegDecodeGetScanDeadzoneSize(void);


size_t NPPIJpegDecodeGetDCTBufferSize(NppiSize oBlocks);


*/
