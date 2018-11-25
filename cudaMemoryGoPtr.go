package gocudnn

/*
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <driver_types.h>

typedef struct cudaPointerAttributes cudaPointerAttributes;
typedef enum cudaMemoryType cudaMemoryType;
*/
import "C"

import (
	"fmt"
	"strconv"
	"unsafe"

	"github.com/pkg/errors"
)

//GoPointer holds a pointer to a slice
type GoPointer struct {
	ptr       unsafe.Pointer
	slice     interface{}
	size      SizeT
	typevalue string
	array     bool
}

func (mem *GoPointer) keepsalive() {

	//runtime.KeepAlive(mem)
	//we don't have to do anything because it is all on the go side
}

//FillSlice will fill the slice, but if the GoPointer has been coppied to device memory. Then the memory will not be up to date.
func (mem *GoPointer) FillSlice(input interface{}) error {
	bsizein, err := FindSizeT(input)
	if err != nil {
		return err
	}
	if bsizein != mem.ByteSize() {
		return errors.New("FillSlice: Sizes Don't Match " + strconv.Itoa(int(bsizein)) + " and " + strconv.Itoa(int(mem.ByteSize())))
	}
	inputtype, err := checkinterface(input)
	if err != nil {
		return err
	}
	memtype, err := checkinterface(mem.slice)
	if err != nil {
		return err
	}
	if inputtype != memtype {
		return errors.New("Fill SLice: Types Don't Match")

	}

	switch x := input.(type) {
	case []float32:
		y := tofloat32array(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []int32:
		y := toint32array(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []int:
		y := tointarray(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []float64:
		y := tofloat64array(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []uint32:
		y := touint32array(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []uint:
		y := touintarray(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	case []byte:
		y := tobytearray(mem.slice)
		for i := 0; i < len(x); i++ {
			x[i] = y[i]
		}
	}

	return nil
}

//ByteSize returns the size of the memory chunk
func (mem *GoPointer) ByteSize() SizeT {
	if mem.ptr == nil {
		return SizeT(0)
	}
	if mem == nil {
		return SizeT(0)
	}

	return mem.size
}

//Ptr returns an unsafe.Pointer
func (mem *GoPointer) Ptr() unsafe.Pointer {
	if mem == nil {
		return nil
	}
	return mem.ptr
}

//Stored returns an Location which can be used to by other programs
func (mem *GoPointer) Stored() Location {
	if mem.ptr == nil {
		return 0
	}
	return 1
}

//Free unassignes the pointers and does the garbage collection
func (mem *GoPointer) Free() error {
	mem.size = 0
	mem.ptr = nil
	mem.typevalue = ""
	return nil
}
func freegopointer(mem *GoPointer) error {
	return mem.Free()
}

//MakeGoPointer takes a slice and gives a GoPointer for that slice.  I wouldn't use that slice anylonger
func MakeGoPointer(input interface{}) (*GoPointer, error) {
	//fname:="MakeGoPointer"
	var ptr GoPointer
	ptr.slice = input
	var err error
	switch val := input.(type) {
	case []int:

		ptr.ptr = unsafe.Pointer(&val[0])

		ptr.typevalue = "int"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}

		return &ptr, nil
	case []int8:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "int8"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}

		return &ptr, nil

	case []byte:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "byte"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}

		return &ptr, nil
	case []float64:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "float64"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}

		return &ptr, nil
	case []float32:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "float32"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}

		return &ptr, nil
	case []int32:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "int32"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}

		return &ptr, nil
	case []uint32:

		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.typevalue = "uint32"
		ptr.size, err = FindSizeT(val)
		ptr.array = true
		if err != nil {
			return nil, err
		}

		return &ptr, nil

	case int:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "int"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case int8:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "int8"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil

	case byte:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "byte"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case float64:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "float64"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case float32:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "float32"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case int32:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "int32"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case uint32:

		ptr.ptr = unsafe.Pointer(&val)
		ptr.typevalue = "uint32"
		ptr.size, err = FindSizeT(val)
		if err != nil {
			return nil, err
		}
		return &ptr, nil
	case CInt:

		ptr.ptr = val.CPtr()
		ptr.typevalue = "CInt"
		ptr.size = SizeT(val.Bytes())
		return &ptr, nil
	case CDouble:

		ptr.ptr = val.CPtr()
		ptr.typevalue = "CDouble"
		ptr.size = SizeT(val.Bytes())
		return &ptr, nil
	case CFloat:

		ptr.ptr = val.CPtr()
		ptr.typevalue = "CFloat"
		ptr.size = SizeT(val.Bytes())
		return &ptr, nil
	case CUInt:

		ptr.ptr = val.CPtr()
		ptr.typevalue = "CUInt"
		ptr.size = SizeT(val.Bytes())
		return &ptr, nil
	default:
		thetype := fmt.Errorf("Type %T", val)
		return nil, errors.New("MakeGoPointer: Unsupported Type -- Type: " + thetype.Error())
	}
}
