package cudnn

/*
#include <cudnn.h>
*/
import "C"
import "errors"

//Status is the status of the cuda dnn
type Status C.cudnnStatus_t

//enumeration returns for cudnn
const (
	StatusSuccess Status = iota
	StatusNotInitialized
	StatusAllocFailed
	StatusBadParam
	StatusInternalError
	StatusInvalidValue
	StatusArchMismatch
	StatusMappingError
	StatusExecutionFailed
	StatusNotSupported
	StatusLicenseError
	StatusRuntimePrerequisiteMissing
	StatusRuntimeInProgress
	StatusRuntimeFpOverflow
)

//GetErrorString is the function that makes a human readable message
func (status Status) GetErrorString() string {
	response := C.cudnnGetErrorString(C.cudnnStatus_t(status))
	return C.GoString(response)
}

//Error will return the error string if there was an error. If not it will return nil
func (status Status) error(comment string) error {
	if status == StatusSuccess {
		return nil
	}
	x := comment + ":"
	return errors.New(x + "cudnn:" + status.GetErrorString())
}
