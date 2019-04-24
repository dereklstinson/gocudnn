package nvjpeg

/*
#include <nvjpeg.h>
#include <cuda_runtime_api.h>
*/
import "C"
import "fmt"

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
		return fmt.Errorf("Unsupported Error- %d", n)
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
		return fmt.Sprintf("Unsupported Error, %d", n)

	}

}
