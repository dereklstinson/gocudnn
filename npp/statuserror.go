package npp

//#include <nppdefs.h>
import "C"

/**
 * Error Status Codes
 *
 * Almost all NPP function return error-status information using
 * these return codes.
 * Negative return codes indicate errors, positive return codes indicate
 * warnings, a return code of 0 indicates success.
 */
type status C.NppStatus

func (n status) error() error {
	switch n {
	case status(C.NPP_NO_ERROR):
		return nil
	}
	return n
}

func (n status) ToError() error {
	return n.error()
}
func (n status) Error() string {
	switch n {
	case status(C.NPP_NOT_SUPPORTED_MODE_ERROR):
		return "NPP_NOT_SUPPORTED_MODE_ERROR"
	case status(C.NPP_INVALID_HOST_POINTER_ERROR):
		return "NPP_INVALID_HOST_POINTER_ERROR"
	case status(C.NPP_INVALID_DEVICE_POINTER_ERROR):
		return "NPP_INVALID_DEVICE_POINTER_ERROR"
	case status(C.NPP_LUT_PALETTE_BITSIZE_ERROR):
		return "NPP_LUT_PALETTE_BITSIZE_ERROR"
	case status(C.NPP_ZC_MODE_NOT_SUPPORTED_ERROR):
		return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR" /**<  ZeroCrossing mode not supported  */
	case status(C.NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY):
		return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY"
	case status(C.NPP_TEXTURE_BIND_ERROR):
		return "NPP_TEXTURE_BIND_ERROR"
	case status(C.NPP_WRONG_INTERSECTION_ROI_ERROR):
		return "NPP_WRONG_INTERSECTION_ROI_ERROR"
	case status(C.NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR):
		return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR"
	case status(C.NPP_MEMFREE_ERROR):
		return "NPP_MEMFREE_ERROR"
	case status(C.NPP_MEMSET_ERROR):
		return "NPP_MEMSET_ERROR"
	case status(C.NPP_MEMCPY_ERROR):
		return "NPP_MEMCPY_ERROR"
	case status(C.NPP_ALIGNMENT_ERROR):
		return "NPP_ALIGNMENT_ERROR"
	case status(C.NPP_CUDA_KERNEL_EXECUTION_ERROR):
		return "NPP_CUDA_KERNEL_EXECUTION_ERROR"
	case status(C.NPP_ROUND_MODE_NOT_SUPPORTED_ERROR):
		return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR" /**< Unsupported round mode*/
	case status(C.NPP_QUALITY_INDEX_ERROR):
		return "NPP_QUALITY_INDEX_ERROR" /**< Image pixels are constant for quality index */
	case status(C.NPP_RESIZE_NO_OPERATION_ERROR):
		return "NPP_RESIZE_NO_OPERATION_ERROR" /**< One of the output image dimensions is less than 1 pixel */
	case status(C.NPP_OVERFLOW_ERROR):
		return "NPP_OVERFLOW_ERROR" /**< Number overflows the upper or lower limit of the data type */
	case status(C.NPP_NOT_EVEN_STEP_ERROR):
		return "NPP_NOT_EVEN_STEP_ERROR" /**< Step value is not pixel multiple */
	case status(C.NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR):
		return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR" /**< Number of levels for histogram is less than 2 */
	case status(C.NPP_LUT_NUMBER_OF_LEVELS_ERROR):
		return "NPP_LUT_NUMBER_OF_LEVELS_ERROR" /**< Number of levels for LUT is less than 2 */
	case status(C.NPP_CORRUPTED_DATA_ERROR):
		return "NPP_CORRUPTED_DATA_ERROR" /**< Processed data is corrupted */
	case status(C.NPP_CHANNEL_ORDER_ERROR):
		return "NPP_CHANNEL_ORDER_ERROR" /**< Wrong order of the destination channels */
	case status(C.NPP_ZERO_MASK_VALUE_ERROR):
		return "NPP_ZERO_MASK_VALUE_ERROR" /**< All values of the mask are zero */
	case status(C.NPP_QUADRANGLE_ERROR):
		return "NPP_QUADRANGLE_ERROR" /**< The quadrangle is nonconvex or degenerates into triangle, line or point */
	case status(C.NPP_RECTANGLE_ERROR):
		return "NPP_RECTANGLE_ERROR" /**< Size of the rectangle region is less than or equal to 1 */
	case status(C.NPP_COEFFICIENT_ERROR):
		return "NPP_COEFFICIENT_ERROR" /**< Unallowable values of the transformation coefficients   */
	case status(C.NPP_NUMBER_OF_CHANNELS_ERROR):
		return "NPP_NUMBER_OF_CHANNELS_ERROR" /**< Bad or unsupported number of channels */
	case status(C.NPP_COI_ERROR):
		return "NPP_COI_ERROR" /**< Channel of interest is not 1, 2, or 3 */
	case status(C.NPP_DIVISOR_ERROR):
		return "NPP_DIVISOR_ERROR" /**< Divisor is equal to zero */
	case status(C.NPP_CHANNEL_ERROR):
		return "NPP_CHANNEL_ERROR" /**< Illegal channel index */
	case status(C.NPP_STRIDE_ERROR):
		return "NPP_STRIDE_ERROR" /**< Stride is less than the row length */
	case status(C.NPP_ANCHOR_ERROR):
		return "NPP_ANCHOR_ERROR" /**< Anchor point is outside mask */
	case status(C.NPP_MASK_SIZE_ERROR):
		return "NPP_MASK_SIZE_ERROR" /**< Lower bound is larger than upper bound */
	case status(C.NPP_RESIZE_FACTOR_ERROR):
		return "NPP_RESIZE_FACTOR_ERROR"
	case status(C.NPP_INTERPOLATION_ERROR):
		return "NPP_INTERPOLATION_ERROR"
	case status(C.NPP_MIRROR_FLIP_ERROR):
		return "NPP_MIRROR_FLIP_ERROR"
	case status(C.NPP_MOMENT_00_ZERO_ERROR):
		return "NPP_MOMENT_00_ZERO_ERROR"
	case status(C.NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR):
		return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR"
	case status(C.NPP_THRESHOLD_ERROR):
		return "NPP_THRESHOLD_ERROR"
	case status(C.NPP_CONTEXT_MATCH_ERROR):
		return "NPP_CONTEXT_MATCH_ERROR"
	case status(C.NPP_FFT_FLAG_ERROR):
		return "NPP_FFT_FLAG_ERROR"
	case status(C.NPP_FFT_ORDER_ERROR):
		return "NPP_FFT_ORDER_ERROR"
	case status(C.NPP_STEP_ERROR):
		return "NPP_STEP_ERROR" /**<  Step is less or equal zero */
	case status(C.NPP_SCALE_RANGE_ERROR):
		return "NPP_SCALE_RANGE_ERROR"
	case status(C.NPP_DATA_TYPE_ERROR):
		return "NPP_DATA_TYPE_ERROR"
	case status(C.NPP_OUT_OFF_RANGE_ERROR):
		return "NPP_OUT_OFF_RANGE_ERROR"
	case status(C.NPP_DIVIDE_BY_ZERO_ERROR):
		return "NPP_DIVIDE_BY_ZERO_ERROR"
	case status(C.NPP_MEMORY_ALLOCATION_ERR):
		return "NPP_MEMORY_ALLOCATION_ERR"
	case status(C.NPP_NULL_POINTER_ERROR):
		return "NPP_NULL_POINTER_ERROR"
	case status(C.NPP_RANGE_ERROR):
		return "NPP_RANGE_ERROR"
	case status(C.NPP_SIZE_ERROR):
		return "NPP_SIZE_ERROR"
	case status(C.NPP_BAD_ARGUMENT_ERROR):
		return "NPP_BAD_ARGUMENT_ERROR"
	case status(C.NPP_NO_MEMORY_ERROR):
		return "NPP_NO_MEMORY_ERROR"
	case status(C.NPP_NOT_IMPLEMENTED_ERROR):
		return "NPP_NOT_IMPLEMENTED_ERROR"
	case status(C.NPP_ERROR):
		return "NPP_ERROR"
	case status(C.NPP_ERROR_RESERVED):
		return "NPP_ERROR_RESERVED"
	case status(C.NPP_NO_ERROR):
		return "NPP_NO_ERROR" /**<  Error free operation */
	case status(C.NPP_NO_OPERATION_WARNING):
		return "NPP_NO_OPERATION_WARNING" /**<  Indicates that no operation was performed */
	case status(C.NPP_DIVIDE_BY_ZERO_WARNING):
		return "NPP_DIVIDE_BY_ZERO_WARNING" /**<  Divisor is zero however does not terminate the execution */
	case status(C.NPP_AFFINE_QUAD_INCORRECT_WARNING):
		return "NPP_AFFINE_QUAD_INCORRECT_WARNING" /**<  Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded. */
	case status(C.NPP_WRONG_INTERSECTION_ROI_WARNING):
		return "NPP_WRONG_INTERSECTION_ROI_WARNING" /**<  The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed. */
	case status(C.NPP_WRONG_INTERSECTION_QUAD_WARNING):
		return "NPP_WRONG_INTERSECTION_QUAD_WARNING" /**<  The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed. */
	case status(C.NPP_DOUBLE_SIZE_WARNING):
		return "NPP_DOUBLE_SIZE_WARNING" /**<  Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing. */
	case status(C.NPP_MISALIGNED_DST_ROI_WARNING):
		return "NPP_MISALIGNED_DST_ROI_WARNING" /**<  Speed reduction due to uncoalesced memory accesses warning. */
	}
	return "UNSUPPORTED STATUS FLAG ON GO SIDE OF BINDING"
}
/*
//GpuComputeCapability is the cuda compute capability
type GpuComputeCapability C.NppGpuComputeCapability

func (g GpuComputeCapability) String() string {
	switch g {
	case GpuComputeCapability(C.NPP_CUDA_UNKNOWN_VERSION):
		return "NPP_CUDA_UNKNOWN_VERSION"
	case GpuComputeCapability(C.NPP_CUDA_NOT_CAPABLE):
		return "NPP_CUDA_NOT_CAPABLE"
	case GpuComputeCapability(C.NPP_CUDA_1_0):
		return "NPP_CUDA_1_0"
	case GpuComputeCapability(C.NPP_CUDA_1_1):
		return "NPP_CUDA_1_1"
	case GpuComputeCapability(C.NPP_CUDA_1_2):
		return "NPP_CUDA_1_2"
	case GpuComputeCapability(C.NPP_CUDA_1_3):
		return "NPP_CUDA_1_3"
	case GpuComputeCapability(C.NPP_CUDA_2_0):
		return "NPP_CUDA_2_0"
	case GpuComputeCapability(C.NPP_CUDA_2_1):
		return "NPP_CUDA_2_1"
	case GpuComputeCapability(C.NPP_CUDA_3_0):
		return "NPP_CUDA_3_0"
	case GpuComputeCapability(C.NPP_CUDA_3_2):
		return "NPP_CUDA_3_2"
	case GpuComputeCapability(C.NPP_CUDA_3_5):
		return "NPP_CUDA_3_5"
	case GpuComputeCapability(C.NPP_CUDA_3_7):
		return "NPP_CUDA_3_7"
	case GpuComputeCapability(C.NPP_CUDA_5_0):
		return "NPP_CUDA_5_0"
	case GpuComputeCapability(C.NPP_CUDA_5_2):
		return "NPP_CUDA_5_2"
	case GpuComputeCapability(C.NPP_CUDA_5_3):
		return "NPP_CUDA_5_3"
	case GpuComputeCapability(C.NPP_CUDA_6_0):
		return "NPP_CUDA_6_0"
	case GpuComputeCapability(C.NPP_CUDA_6_1):
		return "NPP_CUDA_6_1"
	case GpuComputeCapability(C.NPP_CUDA_6_2):
		return "NPP_CUDA_6_2"
	case GpuComputeCapability(C.NPP_CUDA_6_3):
		return "NPP_CUDA_6_3"
	case GpuComputeCapability(C.NPP_CUDA_7_0):
		return "NPP_CUDA_7_0"
	case GpuComputeCapability(C.NPP_CUDA_7_2):
		return "NPP_CUDA_7_2"
		//	case GpuComputeCapability(C.NPP_CUDA_7_3):
		//		return "NPP_CUDA_7_3"
	case GpuComputeCapability(C.NPP_CUDA_7_5):
		return "NPP_CUDA_7_5"
	}
	return "SUPER COMPUTE CAPABILITY OR NONE IDK"
}
*/
//LibraryVersion is the version of the npp library
type LibraryVersion C.NppLibraryVersion
