package cuda

/*
#include <cuda.h>


// Needed to check for NULL from Cgo.
const char * nullMessage = NULL;

const char * go_cuda_cu_err(CUresult res) {
	switch (res) {
	case CUDA_SUCCESS:
		return NULL;
	case CUDA_ERROR_INVALID_VALUE:
		return "CUDA_ERROR_INVALID_VALUE";
	case CUDA_ERROR_OUT_OF_MEMORY:
		return "CUDA_ERROR_OUT_OF_MEMORY";
	case CUDA_ERROR_NOT_INITIALIZED:
		return "CUDA_ERROR_NOT_INITIALIZED";
	case CUDA_ERROR_DEINITIALIZED:
		return "CUDA_ERROR_DEINITIALIZED";
	case CUDA_ERROR_NO_DEVICE:
		return "CUDA_ERROR_NO_DEVICE";
	case CUDA_ERROR_INVALID_DEVICE:
		return "CUDA_ERROR_INVALID_DEVICE";
	case CUDA_ERROR_INVALID_IMAGE:
		return "CUDA_ERROR_INVALID_IMAGE";
	case CUDA_ERROR_INVALID_CONTEXT:
		return "CUDA_ERROR_INVALID_CONTEXT";
	case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
		return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
	case CUDA_ERROR_MAP_FAILED:
		return "CUDA_ERROR_MAP_FAILED";
	case CUDA_ERROR_UNMAP_FAILED:
		return "CUDA_ERROR_UNMAP_FAILED";
	case CUDA_ERROR_ARRAY_IS_MAPPED:
		return "CUDA_ERROR_ARRAY_IS_MAPPED";
	case CUDA_ERROR_ALREADY_MAPPED:
		return "CUDA_ERROR_ALREADY_MAPPED";
	case CUDA_ERROR_NO_BINARY_FOR_GPU:
		return "CUDA_ERROR_NO_BINARY_FOR_GPU";
	case CUDA_ERROR_ALREADY_ACQUIRED:
		return "CUDA_ERROR_ALREADY_ACQUIRED";
	case CUDA_ERROR_NOT_MAPPED:
		return "CUDA_ERROR_NOT_MAPPED";
	case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
		return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
	case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
		return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
	case CUDA_ERROR_ECC_UNCORRECTABLE:
		return "CUDA_ERROR_ECC_UNCORRECTABLE";
	case CUDA_ERROR_UNSUPPORTED_LIMIT:
		return "CUDA_ERROR_UNSUPPORTED_LIMIT";
	case CUDA_ERROR_INVALID_SOURCE:
		return "CUDA_ERROR_INVALID_SOURCE";
	case CUDA_ERROR_FILE_NOT_FOUND:
		return "CUDA_ERROR_FILE_NOT_FOUND";
	case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
		return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
	case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
		return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
	case CUDA_ERROR_OPERATING_SYSTEM:
		return "CUDA_ERROR_OPERATING_SYSTEM";
	case CUDA_ERROR_INVALID_HANDLE:
		return "CUDA_ERROR_INVALID_HANDLE";
	case CUDA_ERROR_NOT_FOUND:
		return "CUDA_ERROR_NOT_FOUND";
	case CUDA_ERROR_NOT_READY:
		return "CUDA_ERROR_NOT_READY";
	case CUDA_ERROR_LAUNCH_FAILED:
		return "CUDA_ERROR_LAUNCH_FAILED";
	case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
		return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
	case CUDA_ERROR_LAUNCH_TIMEOUT:
		return "CUDA_ERROR_LAUNCH_TIMEOUT";
	case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
		return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
	default:
		return "CUDA_ERROR_UNKNOWN";
	}
}
*/
import "C"

// Error is a CUDA-related error.
type Error struct {
	// Context is typically a C function name.
	Context string

	// Name is the C constant name for the error,
	// such as "CURAND_STATUS_INTERNAL_ERROR".
	Name string

	// Message is the main error message.
	//
	// This may be human-readable, although it may often be
	// the same as Name.
	Message string
}

// newErrorDriver creates an Error from the result of a
// CUDA driver API call.
//
// If e is CUDA_SUCCESS, nil is returned.
func newErrorDriver(context string, e C.CUresult) error {
	return newErrorCStr(context, C.go_cuda_cu_err(e))
}

// newErrorRuntime creates an Error from the result of a
// CUDA runtime API call.
//
// If e is cudaSuccess, nil is returned.

func newErrorCStr(context string, cstr *C.char) error {
	if cstr == C.nullMessage {
		return nil
	}
	name := C.GoString(cstr)
	return &Error{
		Context: context,
		Name:    name,
		Message: name,
	}
}

// Error generates a message "context: message".
func (e *Error) Error() string {
	return e.Context + ": " + e.Message
}

/*
Code from github.com/unixpickle/cuda
*/
