package gocu

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/dereklstinson/half"
)

//Wrapper allows go memory to interact with cuda using the Mem interface
type Wrapper struct {
	ptr       unsafe.Pointer
	unitlen   uint
	unitbytes uint
	typeflag  int
}

//Ptr is an unsafe.Pointer of some cuda memory
func (g *Wrapper) Ptr() unsafe.Pointer {
	return g.ptr
}

//DPtr is a double pointer of the unsafe.Pointer
func (g *Wrapper) DPtr() *unsafe.Pointer {
	return &g.ptr
}

//OffSet returns a new Wrapper
func (g *Wrapper) OffSet(byunits uint) *Wrapper {

	offset := unsafe.Pointer(uintptr(g.ptr) + uintptr(byunits*g.unitbytes))

	return &Wrapper{
		ptr:       offset,
		unitlen:   g.unitlen - byunits,
		unitbytes: g.unitbytes,
	}
}

//TotalBytes returns the total bytes this has
func (g *Wrapper) TotalBytes() uint {
	return g.unitlen * g.unitbytes

}

//MakeGoMem returns a GoMem considering the input type.
//Will only support slices and pointers to go types
func MakeGoMem(input interface{}) (*Wrapper, error) {
	//fname:="MakeGoPointer"
	ptr := new(Wrapper)
	switch val := input.(type) {
	case []int:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.unitlen = (uint)(len(val))
		ptr.unitbytes = (uint)(unsafe.Sizeof(val[0]))
		ptr.typeflag = 1
		return ptr, nil
	case []int8:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.unitlen = (uint)(len(val))
		ptr.unitbytes = (uint)(unsafe.Sizeof(val[0]))
		ptr.typeflag = 2
		return ptr, nil
	case []byte:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.unitlen = (uint)(len(val))
		ptr.unitbytes = (uint)(unsafe.Sizeof(val[0]))

		return ptr, nil
	case []float64:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.unitlen = (uint)(len(val))
		ptr.unitbytes = (uint)(unsafe.Sizeof(val[0]))
		return ptr, nil
	case []uint32:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.unitlen = (uint)(len(val))
		ptr.unitbytes = (uint)(unsafe.Sizeof(val[0]))
		return ptr, nil
	case []float32:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.unitlen = (uint)(len(val))
		ptr.unitbytes = (uint)(unsafe.Sizeof(val[0]))
		return ptr, nil
	case []int32:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.unitlen = (uint)(len(val))
		ptr.unitbytes = (uint)(unsafe.Sizeof(val[0]))
		return ptr, nil
	case []half.Float16:
		ptr.ptr = unsafe.Pointer(&val[0])
		ptr.unitlen = (uint)(len(val))
		ptr.unitbytes = (uint)(unsafe.Sizeof(val[0]))
		return ptr, nil
	case *int:
		ptr.ptr = unsafe.Pointer(val)
		ptr.unitlen = 1
		ptr.unitbytes = (uint)(unsafe.Sizeof(val))
		return ptr, nil
	case *int8:
		ptr.ptr = unsafe.Pointer(val)
		ptr.unitlen = 1
		ptr.unitbytes = (uint)(unsafe.Sizeof(val))
		return ptr, nil
	case *byte:
		ptr.ptr = unsafe.Pointer(val)
		ptr.unitlen = 1
		ptr.unitbytes = (uint)(unsafe.Sizeof(val))
		return ptr, nil
	case *float64:
		ptr.ptr = unsafe.Pointer(val)
		ptr.unitlen = 1
		ptr.unitbytes = (uint)(unsafe.Sizeof(val))
		return ptr, nil
	case *float32:
		ptr.ptr = unsafe.Pointer(val)
		ptr.unitlen = 1
		ptr.unitbytes = (uint)(unsafe.Sizeof(val))
		return ptr, nil
	case *half.Float16:
		ptr.ptr = unsafe.Pointer(val)
		ptr.unitlen = 1
		ptr.unitbytes = (uint)(unsafe.Sizeof(val))
		return ptr, nil
	case *int32:
		ptr.ptr = unsafe.Pointer(val)
		ptr.unitlen = 1
		ptr.unitbytes = (uint)(unsafe.Sizeof(val))
		return ptr, nil
	case *uint32:
		ptr.ptr = unsafe.Pointer(val)
		ptr.unitlen = 1
		ptr.unitbytes = (uint)(unsafe.Sizeof(val))
		return ptr, nil
	default:
		thetype := fmt.Errorf("Type %T", val)
		return nil, errors.New("MakeGoPointer: Unsupported Type -- Type: " + thetype.Error())
	}
}

/*
case *CInt:
	ptr.ptr = unsafe.Pointer(val)
	ptr.unitlen = 1
	ptr.unitbytes = (uint)(unsafe.Sizeof(val))
	return ptr, nil
case *CDouble:
	ptr.ptr = unsafe.Pointer(val)
	ptr.unitlen = 1
	ptr.unitbytes = (uint)(unsafe.Sizeof(val))
	return ptr, nil
case *CFloat:

	ptr.ptr = unsafe.Pointer(val)
	ptr.unitlen = 1
	ptr.unitbytes = (uint)(unsafe.Sizeof(val))
	return ptr, nil
case *CUInt:

	ptr.ptr = val.CPtr()
	ptr.typevalue = "CUInt"
	ptr.size = SizeT(val.Bytes())
	return &ptr, nil
case *CHalf:
	ptr.ptr = val.CPtr()
	ptr.typevalue = "CUInt"
	ptr.size = SizeT(val.Bytes())
	return &ptr, nil

*/
