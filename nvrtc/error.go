package nvrtc

/*
#include <nvrtc.h>
*/
import "C"
import (
	"errors"
	"fmt"
)

type status C.nvrtcResult

func (s status) Error() string {
	switch s {
	case status(C.NVRTC_SUCCESS):
		return "NVRTC_SUCCESS"
	case status(C.NVRTC_ERROR_OUT_OF_MEMORY):
		return "NVRTC_ERROR_OUT_OF_MEMORY"
	case status(C.NVRTC_ERROR_PROGRAM_CREATION_FAILURE):
		return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE"
	case status(C.NVRTC_ERROR_INVALID_INPUT):
		return "NVRTC_ERROR_INVALID_INPUT"
	case status(C.NVRTC_ERROR_INVALID_PROGRAM):
		return "NVRTC_ERROR_INVALID_PROGRAM"
	case status(C.NVRTC_ERROR_INVALID_OPTION):
		return "NVRTC_ERROR_INVALID_OPTION"
	case status(C.NVRTC_ERROR_COMPILATION):
		return "NVRTC_ERROR_COMPILATION"
	case status(C.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE):
		return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE"
	case status(C.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION):
		return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION"
	case status(C.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION):
		return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"
	case status(C.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID):
		return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID"
	case status(C.NVRTC_ERROR_INTERNAL_ERROR):
		return "NVRTC_ERROR_INTERNAL_ERROR"
	default:
		return fmt.Sprintf("Unsupported Error, %d", s)
	}
}
func (s status) error(commment string) error {
	switch s {
	case (status)(C.NVRTC_SUCCESS):
		return nil
	case status(C.NVRTC_ERROR_OUT_OF_MEMORY):
		return errors.New(commment + " : " + s.Error())
	case status(C.NVRTC_ERROR_PROGRAM_CREATION_FAILURE):
		return errors.New(commment + " : " + s.Error())
	case status(C.NVRTC_ERROR_INVALID_INPUT):
		return errors.New(commment + " : " + s.Error())
	case status(C.NVRTC_ERROR_INVALID_PROGRAM):
		return errors.New(commment + " : " + s.Error())
	case status(C.NVRTC_ERROR_INVALID_OPTION):
		return errors.New(commment + " : " + s.Error())
	case status(C.NVRTC_ERROR_COMPILATION):
		return errors.New(commment + " : " + s.Error())
	case status(C.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE):
		return errors.New(commment + " : " + s.Error())
	case status(C.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION):
		return errors.New(commment + " : " + s.Error())
	case status(C.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION):
		return errors.New(commment + " : " + s.Error())
	case status(C.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID):
		return errors.New(commment + " : " + s.Error())
	case status(C.NVRTC_ERROR_INTERNAL_ERROR):
		return errors.New(commment + " : " + s.Error())
	default:
		return errors.New(commment + " : " + s.Error())

	}
}
