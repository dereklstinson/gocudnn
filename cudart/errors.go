package cudart

/*
#include <cuda_runtime_api.h>
// Needed to check for NULL from Cgo.
const char * nullMessagex = NULL;
*/
import "C"

type status C.cudaError_t //eventually moving error handling to this

func (s status) error(message string) error {
	if s == status(C.cudaSuccess) {
		return nil
	}
	return newErrorCStr(message, C.cudaGetErrorString((C.cudaError_t)(s)))
}

func newErrorRuntime(context string, e C.cudaError_t) error {
	if e == C.cudaSuccess {
		return nil
	}
	return newErrorCStr(context, C.cudaGetErrorString(e))
}
func newErrorCStr(context string, cstr *C.char) error {
	if cstr == C.nullMessagex {
		return nil
	}
	name := C.GoString(cstr)
	return &Error{
		Context: context,
		Name:    name,
		Message: name,
	}
}

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

// Error generates a message "context: message".
func (e *Error) Error() string {
	return e.Context + ": " + e.Message
}
