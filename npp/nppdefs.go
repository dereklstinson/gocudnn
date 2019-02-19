package npp

//#include <nppdefs.h>
import "C"
import "unsafe"

type NppiInterpolationMode C.NppiInterpolationMode

const (
	NppiInterUNDEFINED        = NppiInterpolationMode(C.NPPI_INTER_UNDEFINED)
	NppiInterNN               = NppiInterpolationMode(C.NPPI_INTER_NN)                 /**<  Nearest neighbor filtering. */
	NppiInterLINEAR           = NppiInterpolationMode(C.NPPI_INTER_LINEAR)             /**<  Linear interpolation. */
	NppiInterCUBIC            = NppiInterpolationMode(C.NPPI_INTER_CUBIC)              /**<  Cubic interpolation. */
	NppiInterBSPLINE          = NppiInterpolationMode(C.NPPI_INTER_CUBIC2P_BSPLINE)    /**<  Two-parameter cubic filter (B=1, C=0) */
	NppiInterCATMULLROM       = NppiInterpolationMode(C.NPPI_INTER_CUBIC2P_CATMULLROM) /**<  Two-parameter cubic filter (B=0, C=1/2) */
	NppiInterB05C03           = NppiInterpolationMode(C.NPPI_INTER_CUBIC2P_B05C03)     /**<  Two-parameter cubic filter (B=1/2, C=3/10) */
	NppiInterSUPER            = NppiInterpolationMode(C.NPPI_INTER_SUPER)              /**<  Super sampling. */
	NppiInterLANCZOS          = NppiInterpolationMode(C.NPPI_INTER_LANCZOS)            /**<  Lanczos filtering. */
	NppiInterLANCZ0S3ADVANCED = NppiInterpolationMode(C.NPPI_INTER_LANCZOS3_ADVANCED)  /**<  Generic Lanczos filtering with order 3. */
	NppiInterSMOOTHEDGE       = NppiInterpolationMode(C.NPPI_SMOOTH_EDGE)              /**<  Smooth edge filtering. */
)

/**
 * Bayer Grid Position Registration.
 */
type NpiiBayerGridePosition C.NppiBayerGridPosition

const (
	NppiBayerBGGR = NpiiBayerGridePosition(C.NPPI_BAYER_BGGR) /**<  Default registration position. */
	NppiBayerRGGB = NpiiBayerGridePosition(C.NPPI_BAYER_RGGB)
	NppiBayerGBRG = NpiiBayerGridePosition(C.NPPI_BAYER_GBRG)
	NppiBayerGRBG = NpiiBayerGridePosition(C.NPPI_BAYER_GRBG)
)

/**
 * Fixed filter-kernel sizes.
 */
type NppiMaskSize C.NppiMaskSize

const (
	NppMaskSize1x3   = NppiMaskSize(C.NPP_MASK_SIZE_1_X_3)
	NppMaskSize1x5   = NppiMaskSize(C.NPP_MASK_SIZE_1_X_5)
	NppMaskSize3x1   = NppiMaskSize(C.NPP_MASK_SIZE_3_X_1) // leaving space for more 1 X N type enum values
	NppMaskSize5x1   = NppiMaskSize(C.NPP_MASK_SIZE_5_X_1)
	NppMaskSize3x3   = NppiMaskSize(C.NPP_MASK_SIZE_3_X_3) // leaving space for more N X 1 type enum values
	NppMaskSize5x5   = NppiMaskSize(C.NPP_MASK_SIZE_5_X_5)
	NppMaskSize7x7   = NppiMaskSize(C.NPP_MASK_SIZE_7_X_7)
	NppMaskSize9x9   = NppiMaskSize(C.NPP_MASK_SIZE_9_X_9)
	NppMaskSize11x11 = NppiMaskSize(C.NPP_MASK_SIZE_11_X_11)
	NppMaskSize13x13 = NppiMaskSize(C.NPP_MASK_SIZE_13_X_13)
	NppMaskSize15x15 = NppiMaskSize(C.NPP_MASK_SIZE_15_X_15)
)

/**
 * Differential Filter types
 */

type NppiDifferentialKernel C.NppiDifferentialKernel

const (
	NppFilterSOBEL  = NppiDifferentialKernel(C.NPP_FILTER_SOBEL)
	NppFilterSCHARR = NppiDifferentialKernel(C.NPP_FILTER_SCHARR)
)

/**
 * Error Status Codes
 *
 * Almost all NPP function return error-status information using
 * these return codes.
 * Negative return codes indicate errors, positive return codes indicate
 * warnings, a return code of 0 indicates success.
 */
type NppStatus C.NppStatus

func (n NppStatus) error() error {
	switch n {
	case NppStatus(C.NPP_NO_ERROR):
		return nil
	}
	return n
}

func (n NppStatus) ToError() error {
	return n.error()
}
func (n NppStatus) Error() string {
	switch n {
	case NppStatus(C.NPP_NOT_SUPPORTED_MODE_ERROR):
		return "NPP_NOT_SUPPORTED_MODE_ERROR"
	case NppStatus(C.NPP_INVALID_HOST_POINTER_ERROR):
		return "NPP_INVALID_HOST_POINTER_ERROR"
	case NppStatus(C.NPP_INVALID_DEVICE_POINTER_ERROR):
		return "NPP_INVALID_DEVICE_POINTER_ERROR"
	case NppStatus(C.NPP_LUT_PALETTE_BITSIZE_ERROR):
		return "NPP_LUT_PALETTE_BITSIZE_ERROR"
	case NppStatus(C.NPP_ZC_MODE_NOT_SUPPORTED_ERROR):
		return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR" /**<  ZeroCrossing mode not supported  */
	case NppStatus(C.NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY):
		return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY"
	case NppStatus(C.NPP_TEXTURE_BIND_ERROR):
		return "NPP_TEXTURE_BIND_ERROR"
	case NppStatus(C.NPP_WRONG_INTERSECTION_ROI_ERROR):
		return "NPP_WRONG_INTERSECTION_ROI_ERROR"
	case NppStatus(C.NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR):
		return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR"
	case NppStatus(C.NPP_MEMFREE_ERROR):
		return "NPP_MEMFREE_ERROR"
	case NppStatus(C.NPP_MEMSET_ERROR):
		return "NPP_MEMSET_ERROR"
	case NppStatus(C.NPP_MEMCPY_ERROR):
		return "NPP_MEMCPY_ERROR"
	case NppStatus(C.NPP_ALIGNMENT_ERROR):
		return "NPP_ALIGNMENT_ERROR"
	case NppStatus(C.NPP_CUDA_KERNEL_EXECUTION_ERROR):
		return "NPP_CUDA_KERNEL_EXECUTION_ERROR"
	case NppStatus(C.NPP_ROUND_MODE_NOT_SUPPORTED_ERROR):
		return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR" /**< Unsupported round mode*/
	case NppStatus(C.NPP_QUALITY_INDEX_ERROR):
		return "NPP_QUALITY_INDEX_ERROR" /**< Image pixels are constant for quality index */
	case NppStatus(C.NPP_RESIZE_NO_OPERATION_ERROR):
		return "NPP_RESIZE_NO_OPERATION_ERROR" /**< One of the output image dimensions is less than 1 pixel */
	case NppStatus(C.NPP_OVERFLOW_ERROR):
		return "NPP_OVERFLOW_ERROR" /**< Number overflows the upper or lower limit of the data type */
	case NppStatus(C.NPP_NOT_EVEN_STEP_ERROR):
		return "NPP_NOT_EVEN_STEP_ERROR" /**< Step value is not pixel multiple */
	case NppStatus(C.NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR):
		return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR" /**< Number of levels for histogram is less than 2 */
	case NppStatus(C.NPP_LUT_NUMBER_OF_LEVELS_ERROR):
		return "NPP_LUT_NUMBER_OF_LEVELS_ERROR" /**< Number of levels for LUT is less than 2 */
	case NppStatus(C.NPP_CORRUPTED_DATA_ERROR):
		return "NPP_CORRUPTED_DATA_ERROR" /**< Processed data is corrupted */
	case NppStatus(C.NPP_CHANNEL_ORDER_ERROR):
		return "NPP_CHANNEL_ORDER_ERROR" /**< Wrong order of the destination channels */
	case NppStatus(C.NPP_ZERO_MASK_VALUE_ERROR):
		return "NPP_ZERO_MASK_VALUE_ERROR" /**< All values of the mask are zero */
	case NppStatus(C.NPP_QUADRANGLE_ERROR):
		return "NPP_QUADRANGLE_ERROR" /**< The quadrangle is nonconvex or degenerates into triangle, line or point */
	case NppStatus(C.NPP_RECTANGLE_ERROR):
		return "NPP_RECTANGLE_ERROR" /**< Size of the rectangle region is less than or equal to 1 */
	case NppStatus(C.NPP_COEFFICIENT_ERROR):
		return "NPP_COEFFICIENT_ERROR" /**< Unallowable values of the transformation coefficients   */
	case NppStatus(C.NPP_NUMBER_OF_CHANNELS_ERROR):
		return "NPP_NUMBER_OF_CHANNELS_ERROR" /**< Bad or unsupported number of channels */
	case NppStatus(C.NPP_COI_ERROR):
		return "NPP_COI_ERROR" /**< Channel of interest is not 1, 2, or 3 */
	case NppStatus(C.NPP_DIVISOR_ERROR):
		return "NPP_DIVISOR_ERROR" /**< Divisor is equal to zero */
	case NppStatus(C.NPP_CHANNEL_ERROR):
		return "NPP_CHANNEL_ERROR" /**< Illegal channel index */
	case NppStatus(C.NPP_STRIDE_ERROR):
		return "NPP_STRIDE_ERROR" /**< Stride is less than the row length */
	case NppStatus(C.NPP_ANCHOR_ERROR):
		return "NPP_ANCHOR_ERROR" /**< Anchor point is outside mask */
	case NppStatus(C.NPP_MASK_SIZE_ERROR):
		return "NPP_MASK_SIZE_ERROR" /**< Lower bound is larger than upper bound */
	case NppStatus(C.NPP_RESIZE_FACTOR_ERROR):
		return "NPP_RESIZE_FACTOR_ERROR"
	case NppStatus(C.NPP_INTERPOLATION_ERROR):
		return "NPP_INTERPOLATION_ERROR"
	case NppStatus(C.NPP_MIRROR_FLIP_ERROR):
		return "NPP_MIRROR_FLIP_ERROR"
	case NppStatus(C.NPP_MOMENT_00_ZERO_ERROR):
		return "NPP_MOMENT_00_ZERO_ERROR"
	case NppStatus(C.NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR):
		return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR"
	case NppStatus(C.NPP_THRESHOLD_ERROR):
		return "NPP_THRESHOLD_ERROR"
	case NppStatus(C.NPP_CONTEXT_MATCH_ERROR):
		return "NPP_CONTEXT_MATCH_ERROR"
	case NppStatus(C.NPP_FFT_FLAG_ERROR):
		return "NPP_FFT_FLAG_ERROR"
	case NppStatus(C.NPP_FFT_ORDER_ERROR):
		return "NPP_FFT_ORDER_ERROR"
	case NppStatus(C.NPP_STEP_ERROR):
		return "NPP_STEP_ERROR" /**<  Step is less or equal zero */
	case NppStatus(C.NPP_SCALE_RANGE_ERROR):
		return "NPP_SCALE_RANGE_ERROR"
	case NppStatus(C.NPP_DATA_TYPE_ERROR):
		return "NPP_DATA_TYPE_ERROR"
	case NppStatus(C.NPP_OUT_OFF_RANGE_ERROR):
		return "NPP_OUT_OFF_RANGE_ERROR"
	case NppStatus(C.NPP_DIVIDE_BY_ZERO_ERROR):
		return "NPP_DIVIDE_BY_ZERO_ERROR"
	case NppStatus(C.NPP_MEMORY_ALLOCATION_ERR):
		return "NPP_MEMORY_ALLOCATION_ERR"
	case NppStatus(C.NPP_NULL_POINTER_ERROR):
		return "NPP_NULL_POINTER_ERROR"
	case NppStatus(C.NPP_RANGE_ERROR):
		return "NPP_RANGE_ERROR"
	case NppStatus(C.NPP_SIZE_ERROR):
		return "NPP_SIZE_ERROR"
	case NppStatus(C.NPP_BAD_ARGUMENT_ERROR):
		return "NPP_BAD_ARGUMENT_ERROR"
	case NppStatus(C.NPP_NO_MEMORY_ERROR):
		return "NPP_NO_MEMORY_ERROR"
	case NppStatus(C.NPP_NOT_IMPLEMENTED_ERROR):
		return "NPP_NOT_IMPLEMENTED_ERROR"
	case NppStatus(C.NPP_ERROR):
		return "NPP_ERROR"
	case NppStatus(C.NPP_ERROR_RESERVED):
		return "NPP_ERROR_RESERVED"
	case NppStatus(C.NPP_NO_ERROR):
		return "NPP_NO_ERROR" /**<  Error free operation */
	case NppStatus(C.NPP_NO_OPERATION_WARNING):
		return "NPP_NO_OPERATION_WARNING" /**<  Indicates that no operation was performed */
	case NppStatus(C.NPP_DIVIDE_BY_ZERO_WARNING):
		return "NPP_DIVIDE_BY_ZERO_WARNING" /**<  Divisor is zero however does not terminate the execution */
	case NppStatus(C.NPP_AFFINE_QUAD_INCORRECT_WARNING):
		return "NPP_AFFINE_QUAD_INCORRECT_WARNING" /**<  Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded. */
	case NppStatus(C.NPP_WRONG_INTERSECTION_ROI_WARNING):
		return "NPP_WRONG_INTERSECTION_ROI_WARNING" /**<  The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed. */
	case NppStatus(C.NPP_WRONG_INTERSECTION_QUAD_WARNING):
		return "NPP_WRONG_INTERSECTION_QUAD_WARNING" /**<  The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed. */
	case NppStatus(C.NPP_DOUBLE_SIZE_WARNING):
		return "NPP_DOUBLE_SIZE_WARNING" /**<  Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing. */
	case NppStatus(C.NPP_MISALIGNED_DST_ROI_WARNING):
		return "NPP_MISALIGNED_DST_ROI_WARNING" /**<  Speed reduction due to uncoalesced memory accesses warning. */
	}
	return "UNSUPPORTED STATUS FLAG ON GO SIDE OF BINDING"
}

/* negative return-codes indicate errors */

type NppGpuComputeCapability C.NppGpuComputeCapability

const (
	NppCudaUnknownVersion = NppGpuComputeCapability(C.NPP_CUDA_UNKNOWN_VERSION) /**<  Indicates that the compute-capability query failed */
	NppCudaNotCapable     = NppGpuComputeCapability(C.NPP_CUDA_NOT_CAPABLE)     /**<  Indicates that no CUDA capable device was found */
	NppCuda0100           = NppGpuComputeCapability(C.NPP_CUDA_1_0)             /**<  Indicates that CUDA 1.0 capable device is machine's default device */
	NppCuda0101           = NppGpuComputeCapability(C.NPP_CUDA_1_1)             /**<  Indicates that CUDA 1.1 capable device is machine's default device */
	NppCuda0102           = NppGpuComputeCapability(C.NPP_CUDA_1_2)             /**<  Indicates that CUDA 1.2 capable device is machine's default device */
	NppCuda0103           = NppGpuComputeCapability(C.NPP_CUDA_1_3)             /**<  Indicates that CUDA 1.3 capable device is machine's default device */
	NppCuda0200           = NppGpuComputeCapability(C.NPP_CUDA_2_0)             /**<  Indicates that CUDA 2.0 capable device is machine's default device */
	NppCuda0201           = NppGpuComputeCapability(C.NPP_CUDA_2_1)             /**<  Indicates that CUDA 2.1 capable device is machine's default device */
	NppCuda0300           = NppGpuComputeCapability(C.NPP_CUDA_3_0)             /**<  Indicates that CUDA 3.0 capable device is machine's default device */
	NppCuda0302           = NppGpuComputeCapability(C.NPP_CUDA_3_2)             /**<  Indicates that CUDA 3.2 capable device is machine's default device */
	NppCuda0305           = NppGpuComputeCapability(C.NPP_CUDA_3_5)             /**<  Indicates that CUDA 3.5 capable device is machine's default device */
	NppCuda0307           = NppGpuComputeCapability(C.NPP_CUDA_3_7)             /**<  Indicates that CUDA 3.7 capable device is machine's default device */
	NppCuda0500           = NppGpuComputeCapability(C.NPP_CUDA_5_0)             /**<  Indicates that CUDA 5.0 capable device is machine's default device */
	NppCuda0502           = NppGpuComputeCapability(C.NPP_CUDA_5_2)             /**<  Indicates that CUDA 5.2 capable device is machine's default device */
	NppCuda0503           = NppGpuComputeCapability(C.NPP_CUDA_5_3)             /**<  Indicates that CUDA 5.3 capable device is machine's default device */
	NppCuda0600           = NppGpuComputeCapability(C.NPP_CUDA_6_0)             /**<  Indicates that CUDA 6.0 capable device is machine's default device */
	NppCuda0601           = NppGpuComputeCapability(C.NPP_CUDA_6_1)             /**<  Indicates that CUDA 6.1 capable device is machine's default device */
	NppCuda0602           = NppGpuComputeCapability(C.NPP_CUDA_6_2)             /**<  Indicates that CUDA 6.2 capable device is machine's default device */
	NppCuda0603           = NppGpuComputeCapability(C.NPP_CUDA_6_3)             /**<  Indicates that CUDA 6.3 capable device is machine's default device */
	NppCuda0700           = NppGpuComputeCapability(C.NPP_CUDA_7_0)             /**<  Indicates that CUDA 7.0 capable device is machine's default device */
	NppCuda0702           = NppGpuComputeCapability(C.NPP_CUDA_7_2)             /**<  Indicates that CUDA 7.2 capable device is machine's default device */
	NppCuda0703           = NppGpuComputeCapability(C.NPP_CUDA_7_3)             /**<  Indicates that CUDA 7.3 capable device is machine's default device */
	NppCuda0705           = NppGpuComputeCapability(C.NPP_CUDA_7_5)             /**<  Indicates that CUDA 7.5 or better is machine's default device */
)

type NppLibraryVersion C.NppLibraryVersion

var (
	nppmajor NppLibraryVersion /**<  Major version number */
	nppminor NppLibraryVersion /**<  Minor version number */
	nppbuild NppLibraryVersion /**<  Build number. This reflects the nightly build this release was made from. */
)

/** \defgroup npp_basic_types Basic NPP Data Types
 * @{
 */
//Npp8u
type Npp8u C.Npp8u   /**<  8-bit unsigned chars */
type Npp8s C.Npp8s   /**<  8-bit signed chars */
type Npp16u C.Npp16u /**<  16-bit unsigned integers */
type Npp16s C.Npp16s /**<  16-bit signed integers */
type Npp32u C.Npp32u /**<  32-bit unsigned integers */
type Npp32s C.Npp32s /**<  32-bit signed integers */
type Npp64u C.Npp64u /**<  64-bit unsigned integers */
type Npp64s C.Npp64s /**<  64-bit signed integers */
type Npp32f C.Npp32f /**<  32-bit (IEEE) floating-point numbers */
type Npp64f C.Npp64f /**<  64-bit floating-point numbers */

func convertNpp64utoCNpp64uarray(x []Npp64u) []C.Npp64u {
	y := make([]C.Npp64u, len(x))
	for i := range x {
		y[i] = C.Npp64u(x[i])
	}
	return y
}
func convertCNpp64utoNpp64uarray(x []C.Npp64u) []Npp64u {
	y := make([]Npp64u, len(x))
	for i := range x {
		y[i] = Npp64u(x[i])
	}
	return y
}
func convertNpp32utoCNpp32uarray(x []Npp32u) []C.Npp32u {
	y := make([]C.Npp32u, len(x))
	for i := range x {
		y[i] = C.Npp32u(x[i])
	}
	return y
}
func convertCNpp32utoNpp32uarray(x []C.Npp32u) []Npp32u {
	y := make([]Npp32u, len(x))
	for i := range x {
		y[i] = Npp32u(x[i])
	}
	return y
}

func convertNpp16utoCNpp16uarray(x []Npp16u) []C.Npp16u {
	y := make([]C.Npp16u, len(x))
	for i := range x {
		y[i] = C.Npp16u(x[i])
	}
	return y
}
func convertCNpp16utoNpp16uarray(x []C.Npp16u) []Npp16u {
	y := make([]Npp16u, len(x))
	for i := range x {
		y[i] = Npp16u(x[i])
	}
	return y
}

func convertNpp8utoCNpp8uarray(x []Npp8u) []C.Npp8u {
	y := make([]C.Npp8u, len(x))
	for i := range x {
		y[i] = C.Npp8u(x[i])
	}
	return y
}
func convertCNpp8utoNpp8uarray(x []C.Npp8u) []Npp8u {
	y := make([]Npp8u, len(x))
	for i := range x {
		y[i] = Npp8u(x[i])
	}
	return y
}

/*Npp8uc  Complex Number
 * This struct represents an unsigned char complex number.
 */
type Npp8uc struct {
	Val *C.Npp8uc
	cpu bool
}

//MakeNpp8ucFromUnsafe makes an Npp16uc from an unsafe Pointer
func MakeNpp8ucFromUnsafe(ptr unsafe.Pointer, cpu bool) *Npp8uc {
	return &Npp8uc{
		Val: (*C.Npp8uc)(ptr),
		cpu: cpu,
	}
}

//type Npp8uc C.Npp8uc

/*
func (c Npp8uc)Real()Npp8u{
    return Npp8u(c.re)
}
func (c Npp8uc)Imaginary()Npp8u{
    return Npp8u(c.im)
}
func CreateNpp8uc(real,imaginary Npp8u)Npp8uc{
    return Npp8uc{
        re:real,
        im:imaginary,
    }
}
*/

/*Npp16uc - See below
 * Complex Number
 * This struct represents an unsigned short complex number.
 */
type Npp16uc struct {
	Val *C.Npp16uc
	cpu bool
}

//MakeNpp16ucFromUnsafe makes an Npp16uc from an unsafe Pointer
func MakeNpp16ucFromUnsafe(ptr unsafe.Pointer, cpu bool) *Npp16uc {
	return &Npp16uc{
		Val: (*C.Npp16uc)(ptr),
		cpu: cpu,
	}
}

//type Npp16uc C.Npp16uc

/*
//  /Real returns real
func (c Npp16uc)Real()uint16{
    return uint16(c.re)
}
//Imaginary Returns Imaginary
func (c Npp16uc)Imaginary()uint16{
    return uint16(c.im)
}

func Npp16uc(real,imaginary uint16)Npp16uc{
    return Npp16uc{
        re:Npp16u(real),
        im:Npp16u(imaginary),
    }
}
*/

/* Npp16sc - See below
 * * Complex Number
 * This struct represents a short complex number.
 */
type Npp16sc struct {
	Val *C.Npp16sc
	cpu bool
}

//MakeNpp16scFromUnsafe makes an Npp32sc from an unsafe Pointer
func MakeNpp16scFromUnsafe(ptr unsafe.Pointer, cpu bool) *Npp16sc {
	return &Npp16sc{
		Val: (*C.Npp16sc)(ptr),
		cpu: cpu,
	}
}

//type Npp16sc C.Npp16sc

/*
//  /Real returns real
func (c Npp16sc)Real()Npp16s{
    return Npp16s(c.re)
}
//Imaginary Returns Imaginary
func (c Npp16sc)Imaginary()Npp16sc{
    return Npp16s(c.im)
}

func Npp16sc(real,imaginary int16)Npp16sc{
    return Npp16sc{
        re:Npp16s(real),
        im:Npp16s(imaginary),
    }
}
*/
/**
Npp32uc - See below
 * * Complex Number
 * This struct represents an unsigned int complex number.
*/
type Npp32uc struct {
	Val *C.Npp32uc
	cpu bool
}

//MakeNpp32ucFromUnsafe makes an Npp32sc from an unsafe Pointer
func MakeNpp32ucFromUnsafe(ptr unsafe.Pointer, cpu bool) *Npp32uc {
	return &Npp32uc{
		Val: (*C.Npp32uc)(ptr),
		cpu: cpu,
	}
}

//type Npp32uc C.Npp32uc

/*
//  /Real returns real
func (c Npp32uc)Real()Npp32u{
    return Npp32u(c.re)
}
//Imaginary Returns Imaginary
func (c Npp32uc)Imaginary()Npp32u{
    return Npp32u(c.im)
}

func Npp32uc(real,imaginary Npp32u)Npp32uc{
    return Npp32uc{
        re:real,
        im:imaginary,
    }
}
*/
/**Npp32sc - Complex Number
 * This struct represents a signed int complex number.
 */
type Npp32sc struct {
	Val *C.Npp32sc
	cpu bool
}

//MakeNpp32scFromUnsafe makes an Npp32sc from an unsafe Pointer
func MakeNpp32scFromUnsafe(ptr unsafe.Pointer, cpu bool) *Npp32sc {
	return &Npp32sc{
		Val: (*C.Npp32sc)(ptr),
		cpu: cpu,
	}
}

/*
//  /Real returns real
func (c Npp32sc)Real()Npp32s{
    return Npp32s(c.re)
}
//Imaginary Returns Imaginary
func (c Npp32sc)Imaginary()Npp32s{
    return Npp32s(c.im)
}

func Npp32uc(real,imaginary Npp32s)Npp32sc{
    return Npp32sc{
        re:real,
        im:imaginary,
    }
}

*/

/*
Npp32fc This struct represents a single floating-point complex number.
*/
type Npp32fc struct {
	Val *C.Npp32fc
	cpu bool
}

//MakeNpp32fcFromUnsafe makes an Npp32fc from an unsafe Pointer
func MakeNpp32fcFromUnsafe(ptr unsafe.Pointer, cpu bool) *Npp32fc {
	return &Npp32fc{
		Val: (*C.Npp32fc)(ptr),
		cpu: cpu,
	}
}

/*
//  /Real returns real
func (c Npp32fc)Real()Npp32f{
    return Npp32f(c.re)
}
//Imaginary Returns Imaginary
func (c Npp32fc)Imaginary()Npp32f{
    return Npp32f(c.im)
}

func Npp32fc(real,imaginary Npp16u)Npp32fc{
    return Npp32fc{
        re:real,
        im:imaginary,
    }
}
*/

// Npp64sc struct represents a long long complex number.
type Npp64sc struct {
	Val *C.Npp64sc
	cpu bool
}

//MakeNpp64scFromUnsafe makes an Npp64sc from an unsafe pointer.
func MakeNpp64scFromUnsafe(ptr unsafe.Pointer, cpu bool) *Npp64sc {
	return &Npp64sc{
		Val: (*C.Npp64sc)(ptr),
		cpu: cpu,
	}
}

/**
 * Complex Number
 * This struct represents a long long complex number.
 */
/*
typedef struct NPP_ALIGN_16
{
    Npp64s  re;
    Npp64s  im;
} Npp64sc;
*/

//Npp64fc struct represents a double floating-point complex number.
type Npp64fc struct {
	Val *C.Npp64fc
	cpu bool
}

//MakeNpp64fcFromUnsafe makes an Npp64fc from an unsafe pointer.
func MakeNpp64fcFromUnsafe(ptr unsafe.Pointer, cpu bool) *Npp64fc {
	return &Npp64fc{
		Val: (*C.Npp64fc)(ptr),
		cpu: cpu,
	}
}

/*
typedef struct NPP_ALIGN_16
{
    Npp64f  re;
    Npp64f  im;
} Npp64fc;
*/

//#define NPP_MIN_8U      ( 0 )                        /**<  Minimum 8-bit unsigned integer */
//#define NPP_MAX_8U      ( 255 )                      /**<  Maximum 8-bit unsigned integer */
//#define NPP_MIN_16U     ( 0 )                        /**<  Minimum 16-bit unsigned integer */
//#define NPP_MAX_16U     ( 65535 )                    /**<  Maximum 16-bit unsigned integer */
//#define NPP_MIN_32U     ( 0 )                        /**<  Minimum 32-bit unsigned integer */
//#define NPP_MAX_32U     ( 4294967295U )              /**<  Maximum 32-bit unsigned integer */
//#define NPP_MIN_64U     ( 0 )                        /**<  Minimum 64-bit unsigned integer */
//#define NPP_MAX_64U     ( 18446744073709551615ULL )  /**<  Maximum 64-bit unsigned integer */

//#define NPP_MIN_8S      (-127 - 1 )                  /**<  Minimum 8-bit signed integer */
//#define NPP_MAX_8S      ( 127 )                      /**<  Maximum 8-bit signed integer */
//#define NPP_MIN_16S     (-32767 - 1 )                /**<  Minimum 16-bit signed integer */
//#define NPP_MAX_16S     ( 32767 )                    /**<  Maximum 16-bit signed integer */
//#define NPP_MIN_32S     (-2147483647 - 1 )           /**<  Minimum 32-bit signed integer */
//#define NPP_MAX_32S     ( 2147483647 )               /**<  Maximum 32-bit signed integer */
//#define NPP_MAX_64S     ( 9223372036854775807LL )    /**<  Maximum 64-bit signed integer */
//#define NPP_MIN_64S     (-9223372036854775807LL - 1) /**<  Minimum 64-bit signed integer */

//#define NPP_MINABS_32F  ( 1.175494351e-38f )         /**<  Smallest positive 32-bit floating point value */
//#define NPP_MAXABS_32F  ( 3.402823466e+38f )         /**<  Largest  positive 32-bit floating point value */
//#define NPP_MINABS_64F  ( 2.2250738585072014e-308 )  /**<  Smallest positive 64-bit floating point value */
//#define NPP_MAXABS_64F  ( 1.7976931348623158e+308 )  /**<  Largest  positive 64-bit floating point value */

//NppiPoint is a 2d point
type NppiPoint C.NppiPoint

/*
typedef struct
{
    int x;
    int y;
} NppiPoint;
*/
//NppPointPolar is a 2D Polar Point
type NppPointPolar C.NppPointPolar

/*

typedef struct {
    Npp32f rho;
    Npp32f theta;
} NppPointPolar;
*/

//NppiSize -2D Size represents the size of a a rectangular region in two space.
type NppiSize C.NppiSize

/*
typedef struct
{
    int width;
    int height;
} NppiSize;
*/

//WidthHeight returns the width and Height
func (n *NppiSize) WidthHeight() (w, h int32) {
	w = int32(n.width)
	h = int32(n.height)
	return w, h
}

/* NppiRect
 * 2D Rectangle
 * This struct contains position and size information of a rectangle in
 * two space.
 * The rectangle's position is usually signified by the coordinate of its
 * upper-left corner.
 */
type NppiRect C.NppiRect

/*
typedef struct
{
    int x;         //  x-coordinate of upper left corner (lowest memory address).
    int y;       // y-coordinate of upper left corner (lowest memory address).
    int width;    // Rectangle width.
    int height;   // Rectangle height.
} NppiRect;
*/

//NppiAxis enums NpiiAxis
type NppiAxis C.NppiAxis

const (
	NppHorizontalAxis = NppiAxis(C.NPP_HORIZONTAL_AXIS)
	NppVerticalAxis   = NppiAxis(C.NPP_VERTICAL_AXIS)
	NppBothAxis       = NppiAxis(C.NPP_BOTH_AXIS)
)

type NppCmpOp C.NppCmpOp

const (
	NppCmpLess      = NppCmpOp(C.NPP_CMP_LESS)
	NppCmpLessEq    = NppCmpOp(C.NPP_CMP_LESS_EQ)
	NppCmpEq        = NppCmpOp(C.NPP_CMP_EQ)
	NppCmpGreaterEq = NppCmpOp(C.NPP_CMP_GREATER_EQ)
	NppCmpGreater   = NppCmpOp(C.NPP_CMP_GREATER)
)

/**
NppRoundMode go wrapper for roundimg modes description from original header
 * Rounding Modes
 *
 * The enumerated rounding modes are used by a large number of NPP primitives
 * to allow the user to specify the method by which fractional values are converted
 * to integer values. Also see \ref rounding_modes.
 *
 * For NPP release 5.5 new names for the three rounding modes are introduced that are
 * based on the naming conventions for rounding modes set forth in the IEEE-754
 * floating-point standard. Developers are encouraged to use the new, longer names
 * to be future proof as the legacy names will be deprecated in subsequent NPP releases.
 *
*/
type NppRoundMode C.NppRoundMode

const (
	/* NppRndNear -From Original Header
	 * Round according to financial rule.
	 * All fractional numbers are rounded to their nearest integer. The ambiguous
	 * cases (i.e. \<integer\>.5) are rounded away from zero.
	 * E.g.
	 * - roundFinancial(0.4)  = 0
	 * - roundFinancial(0.5)  = 1
	 * - roundFinancial(-1.5) = -2
	 */
	NppRndNear             = NppRoundMode(C.NPP_RND_NEAR)
	NppRoundNearTiesToEven = NppRoundMode(C.NPP_ROUND_NEAREST_TIES_TO_EVEN) //equals NppRndNear
	/* NppRndFinancial - From Original Header
	 * Round towards zero (truncation).
	 * All fractional numbers of the form \<integer\>.\<decimals\> are truncated to
	 * \<integer\>.
	 * - roundZero(1.5) = 1
	 * - roundZero(1.9) = 1
	 * - roundZero(-2.5) = -2
	 */

	NppRndFinancial                 = NppRoundMode(C.NPP_RND_FINANCIAL)
	NppRoundNearestTiesAwayFromZero = NppRoundMode(C.NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO) //equals NppRndFinancial

	NppRndZero         = NppRoundMode(C.NPP_RND_ZERO)
	NppRoundTowardZero = NppRoundMode(C.NPP_ROUND_TOWARD_ZERO) //equals NppRndZero
	/*
	 * Other rounding modes supported by IEEE-754 (2008) floating-point standard:
	 *
	 * - NPP_ROUND_TOWARD_INFINITY // ceiling
	 * - NPP_ROUND_TOWARD_NEGATIVE_INFINITY // floor
	 *
	 */
)

type NppiBorderType C.NppiBorderType

const (
	NppBorderUndefined = NppiBorderType(C.NPP_BORDER_UNDEFINED)
	NppBorderNone      = NppiBorderType(C.NPP_BORDER_NONE)
	NppBorderConstant  = NppiBorderType(C.NPP_BORDER_CONSTANT)
	NppBorderReplicate = NppiBorderType(C.NPP_BORDER_REPLICATE)
	NppBorderWrap      = NppiBorderType(C.NPP_BORDER_WRAP)
	NppBorderMirror    = NppiBorderType(C.NPP_BORDER_MIRROR)
)

type NppHintAlgorithm C.NppHintAlgorithm

const (
	NppAlgoHintNone     = NppHintAlgorithm(C.NPP_ALG_HINT_NONE)
	NppAlgoHintFast     = NppHintAlgorithm(C.NPP_ALG_HINT_FAST)
	NppAlgoHintAccurate = NppHintAlgorithm(C.NPP_ALG_HINT_ACCURATE)
)

/*
 * Alpha composition controls.
 */
type NppiAlphaOp C.NppiAlphaOp

const (
	NppiOpAlphaOver       = NppiAlphaOp(C.NPPI_OP_ALPHA_OVER)
	NppiOpAlphaIn         = NppiAlphaOp(C.NPPI_OP_ALPHA_IN)
	NppiOpAlphaOut        = NppiAlphaOp(C.NPPI_OP_ALPHA_OUT)
	NppiOpAlphaAtop       = NppiAlphaOp(C.NPPI_OP_ALPHA_ATOP)
	NppiOpAlphaXOR        = NppiAlphaOp(C.NPPI_OP_ALPHA_XOR)
	NppiOpAlphaPlus       = NppiAlphaOp(C.NPPI_OP_ALPHA_PLUS)
	NppiOpAlphaOverPremul = NppiAlphaOp(C.NPPI_OP_ALPHA_OVER_PREMUL)
	NppiOpAlphaInPremul   = NppiAlphaOp(C.NPPI_OP_ALPHA_IN_PREMUL)
	NppiOpAlphaOutPremul  = NppiAlphaOp(C.NPPI_OP_ALPHA_OUT_PREMUL)
	NppiOpAlphaAtopPremul = NppiAlphaOp(C.NPPI_OP_ALPHA_ATOP_PREMUL)
	NppiOpAlphaXORPremul  = NppiAlphaOp(C.NPPI_OP_ALPHA_XOR_PREMUL)
	NppiOpAlphaPlusPremul = NppiAlphaOp(C.NPPI_OP_ALPHA_PLUS_PREMUL)
	NppiOpAlphaPremul     = NppiAlphaOp(C.NPPI_OP_ALPHA_PREMUL)
)

/**
 * The NppiHOGConfig structure defines the configuration parameters for the HOG descriptor:
 */
type NppiHOGConfig C.NppiHOGConfig

/*
typedef struct
{
    int      cellSize;              //square cell size (pixels).
    int      histogramBlockSize;    //square histogram block size (pixels).
    int      nHistogramBins;        //required number of histogram bins.
    NppiSize detectionWindowSize;   //detection window size (pixels).
} NppiHOGConfig;
*/

//#define NPP_HOG_MAX_CELL_SIZE                          (16) /**< max horizontal/vertical pixel size of cell.   */
//#define NPP_HOG_MAX_BLOCK_SIZE                         (64) /**< max horizontal/vertical pixel size of block.  */
//#define NPP_HOG_MAX_BINS_PER_CELL                      (16) /**< max number of histogram bins. */
//#define NPP_HOG_MAX_CELLS_PER_DESCRIPTOR              (256) /**< max number of cells in a descriptor window.   */
//#define NPP_HOG_MAX_OVERLAPPING_BLOCKS_PER_DESCRIPTOR (256) /**< max number of overlapping blocks in a descriptor window.   */
//#define NPP_HOG_MAX_DESCRIPTOR_LOCATIONS_PER_CALL     (128) /**< max number of descriptor window locations per function call.   */
type NppiHaarClassifier32f C.NppiHaarClassifier_32f

/*
typedef struct
{
    int      numClassifiers;     // number of classifiers
    Npp32s * classifiers;        // packed classifier data 40 bytes each
    size_t   classifierStep;
    NppiSize classifierSize;
    Npp32s * counterDevice;
} NppiHaarClassifier_32f;
*/
type NppiHaarBuffer C.NppiHaarBuffer

/*
typedef struct
{
    int      haarBufferSize;     //size of the buffer
    Npp32s * haarBuffer;        //buffer

} NppiHaarBuffer;
*/
type NppsZCType C.NppsZCType

const (
	NppZCR   = NppsZCType(C.nppZCR)   /**<  sign change */
	NppZCXor = NppsZCType(C.nppZCXor) /**<  sign change XOR */
	NppZCC   = NppsZCType(C.nppZCC)   /**<  sign change count_0 */
)

type NppiHuffmanTableType C.NppiHuffmanTableType

const (
	NppiDCTable = NppiHuffmanTableType(C.nppiDCTable) /**<  DC Table */
	NppiACTable = NppiHuffmanTableType(C.nppiACTable) /**<  AC Table */
)

type NppiNorm C.NppiNorm

const (
	NpiiNormInf = NppiNorm(C.nppiNormInf) /**<  maximum */
	NpiiNormL1  = NppiNorm(C.nppiNormL1)  /**<  sum */
	NpiiNormL2  = NppiNorm(C.nppiNormL2)  /**<  square root of sum of squares */
)
