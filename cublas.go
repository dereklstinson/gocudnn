package gocudnn

/*
#include <cublas_v2.h>
*/
import "C"
import "errors"

//For Now cublas isn't going to be doing anything.  I will add it later

type cublasStatus C.cublasStatus_t

func (s cublasStatus) error() error {
	switch s {
	case cublasStatus(C.CUBLAS_STATUS_SUCCESS):
		return nil
	case cublasStatus(C.CUBLAS_STATUS_NOT_INITIALIZED):
		return errors.New("CUBLAS_STATUS_NOT_INITIALIZED")
	case cublasStatus(C.CUBLAS_STATUS_ALLOC_FAILED):
		return errors.New("CUBLAS_STATUS_ALLOC_FAILED")
	case cublasStatus(C.CUBLAS_STATUS_INVALID_VALUE):
		return errors.New("CUBLAS_STATUS_INVALID_VALUE")
	case cublasStatus(C.CUBLAS_STATUS_ARCH_MISMATCH):
		return errors.New("CUBLAS_STATUS_ARCH_MISMATCH")
	case cublasStatus(C.CUBLAS_STATUS_MAPPING_ERROR):
		return errors.New("CUBLAS_STATUS_MAPPING_ERROR")
	case cublasStatus(C.CUBLAS_STATUS_EXECUTION_FAILED):
		return errors.New("CUBLAS_STATUS_EXECUTION_FAILED")
	case cublasStatus(C.CUBLAS_STATUS_INTERNAL_ERROR):
		return errors.New("CUBLAS_STATUS_INTERNAL_ERROR")
	case cublasStatus(C.CUBLAS_STATUS_NOT_SUPPORTED):
		return errors.New("CUBLAS_STATUS_NOT_SUPPORTED")
	case cublasStatus(C.CUBLAS_STATUS_LICENSE_ERROR):
		return errors.New("CUBLAS_STATUS_LICENSE_ERROR")
	default:
		return errors.New("FromGoWrapper-Unrecognized Error for Cublas")
	}
}

//CublasContext is the context use for cublas
type CublasContext struct {
	x C.cublasHandle_t
}

//CreateCublasContext creates a CublasContext and returns a pointer
func CreateCublasContext() (*CublasContext, error) {
	var con C.cublasHandle_t
	err := C.cublasCreate(&con)
	return &CublasContext{
		x: con,
	}, cublasStatus(err).error()

}
