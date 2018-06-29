package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import "errors"

//Status is the status of the cuda dnn
type Status C.cudnnStatus_t

//StatusSuccess is the zero error of Status.  None of the other flags are visable for now,
// of the Status.error() method
const StatusSuccess Status = 0

//GetErrorString is the function that makes a human readable message
func (status Status) geterrorstring() string {
	response := C.cudnnGetErrorString(C.cudnnStatus_t(status))
	return C.GoString(response)
}

//Error will return the error string if there was an error. If not it will return nil
func (status Status) error(comment string) error {
	if C.cudnnStatus_t(status) == C.CUDNN_STATUS_SUCCESS {
		return nil
	}
	x := comment + ":"
	return errors.New(x + "cudnn:" + status.geterrorstring())
}
