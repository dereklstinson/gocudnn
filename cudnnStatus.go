package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"strings"
)

//Status is the status of the cuda dnn
type Status C.cudnnStatus_t

//StatusSuccess is the zero error of Status.  None of the other flags are visable for now,
// of the Status.error() method
const StatusSuccess Status = 0

//GetErrorString is the function that makes a human readable message
func (status Status) GetErrorString() string {
	response := C.cudnnGetErrorString(C.cudnnStatus_t(status))
	return C.GoString(response)
}

//Error will return the error string if there was an error. If not it will return nil
func (status Status) error(comment string) error {
	if C.cudnnStatus_t(status) == C.CUDNN_STATUS_SUCCESS {
		return nil
	}
	x := comment + ":"
	return errors.New(x + "cudnn:" + status.GetErrorString())
}

func (status Status) c() C.cudnnStatus_t {
	return C.cudnnStatus_t(status)
}
func (status Status) Error(comment string) error {
	return status.error("::Exported Error Function:: " + comment)
}

//WrapErrorWithStatus  if the error string contains a cudnnStatus_t string then it will return the Status and nil,
// if it doens't the Status will be the flag for   CUDNN_STATUS_RUNTIME_FP_OVERFLOW but the error will not return a nil
func WrapErrorWithStatus(e error) (Status, error) {
	if e == nil {
		return Status(C.CUDNN_STATUS_SUCCESS), nil

	}
	x := e.Error()
	switch {
	case strings.Contains(x, "CUDNN_STATUS_NOT_INITIALIZED"):
		return Status(C.CUDNN_STATUS_NOT_INITIALIZED), nil
	case strings.Contains(x, "CUDNN_STATUS_ALLOC_FAILED"):
		return Status(C.CUDNN_STATUS_ALLOC_FAILED), nil
	case strings.Contains(x, "CUDNN_STATUS_BAD_PARAM"):
		return Status(C.CUDNN_STATUS_BAD_PARAM), nil
	case strings.Contains(x, "CUDNN_STATUS_ARCH_MISMATCH"):
		return Status(C.CUDNN_STATUS_ARCH_MISMATCH), nil
	case strings.Contains(x, "CUDNN_STATUS_MAPPING_ERROR"):
		return Status(C.CUDNN_STATUS_MAPPING_ERROR), nil
	case strings.Contains(x, "CUDNN_STATUS_EXECUTION_FAILED"):
		return Status(C.CUDNN_STATUS_EXECUTION_FAILED), nil
	case strings.Contains(x, "CUDNN_STATUS_INTERNAL_ERROR"):
		return Status(C.CUDNN_STATUS_INTERNAL_ERROR), nil
	case strings.Contains(x, "CUDNN_STATUS_NOT_SUPPORTED"):
		return Status(C.CUDNN_STATUS_NOT_SUPPORTED), nil
	case strings.Contains(x, "CUDNN_STATUS_LICENSE_ERROR"):
		return Status(C.CUDNN_STATUS_LICENSE_ERROR), nil
	case strings.Contains(x, "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING"):
		return Status(C.CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING), nil
	case strings.Contains(x, "CUDNN_STATUS_RUNTIME_IN_PROGRESS"):
		return Status(C.CUDNN_STATUS_RUNTIME_IN_PROGRESS), nil
	case strings.Contains(x, "CUDNN_STATUS_RUNTIME_FP_OVERFLOW"):
		return Status(C.CUDNN_STATUS_RUNTIME_FP_OVERFLOW), nil
	default:
		return Status(C.CUDNN_STATUS_RUNTIME_FP_OVERFLOW), errors.New("Unsupported error")
	}

}
