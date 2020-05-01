package npp

/*
import (
	"errors"

	"github.com/dereklstinson/gocudnn/gocu"
)

type AllocatorNpps struct {
	mode nppsallocmode
}

type nppsallocmode int

func (a *AllocatorNpps) Malloc(size uint) (*gocu.CudaPtr, error) {
	sizet := (int32)(size)
	switch a.mode {

	case nppsUint8:
		x := Malloc8u(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsInt8:
		x := Malloc8s(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsUint16:
		x := Malloc16u(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsInt16:
		x := Malloc16s(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsInt16Complex:
		x := Malloc16sc(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsUint32:
		x := Malloc32u(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsInt32:
		x := Malloc32s(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsInt32Complex:
		x := Malloc32sc(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsFloat32Complex:
		x := Malloc32fc(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsInt64:
		x := Malloc64s(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsInt64Complex:
		x := Malloc64sc(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsFloat64:
		x := Malloc64f(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	case nppsFloat64Complex:
		x := Malloc64fc(sizet).Ptr()
		if x == nil {
			return nil, errors.New("Error in AllocatorNpps")
		}
		return gocu.WrapUnsafe(x), nil
	}
	return nil, nil
}

//Uint8 sets the allocator mode to Uint8
func (a *AllocatorNpps) Uint8() {
	a.mode = nppsUint8
}

//Int8 sets the allocator mode to Int8
func (a *AllocatorNpps) Int8() {
	a.mode = nppsInt8
}

//Uint16 sets the allocator mode to Uint16
func (a *AllocatorNpps) Uint16() {
	a.mode = nppsUint16
}

//Int16 sets the allocator mode to Int16
func (a *AllocatorNpps) Int16() {
	a.mode = nppsInt16
}

//Int16Complex sets the allocator mode to Int16Complex
func (a *AllocatorNpps) Int16Complex() {
	a.mode = nppsInt16Complex
}

//Uint32 sets the allocator mode to Uint32
func (a *AllocatorNpps) Uint32() {
	a.mode = nppsUint32
}

//Int32 sets the allocator mode to Int32
func (a *AllocatorNpps) Int32() {
	a.mode = nppsInt32
}

//Int32Complex sets the allocator mode to Int32Complex
func (a *AllocatorNpps) Int32Complex() {
	a.mode = nppsInt32Complex
}

//Float32Complex sets the allocator mode to Float32Complex
func (a *AllocatorNpps) Float32Complex() {
	a.mode = nppsFloat32Complex
}

//Int64 sets the allocator mode to Int64
func (a *AllocatorNpps) Int64() {
	a.mode = nppsInt64
}

//Int64Complex sets the allocator mode to Int64Complex
func (a *AllocatorNpps) Int64Complex() {
	a.mode = nppsInt64Complex
}

//Float64 sets the allocator mode to Float64
func (a *AllocatorNpps) Float64() {
	a.mode = nppsFloat64
}

//Float64Complex sets the allocator mode to Float64Complex
func (a *AllocatorNpps) Float64Complex() {
	a.mode = nppsFloat64Complex
}

const (
	nppsUint8          = nppsallocmode(1)
	nppsInt8           = nppsallocmode(2)
	nppsUint16         = nppsallocmode(3)
	nppsInt16          = nppsallocmode(4)
	nppsInt16Complex   = nppsallocmode(5)
	nppsUint32         = nppsallocmode(6)
	nppsInt32          = nppsallocmode(7)
	nppsInt32Complex   = nppsallocmode(8)
	nppsFloat32Complex = nppsallocmode(9)
	nppsInt64          = nppsallocmode(10)
	nppsInt64Complex   = nppsallocmode(11)
	nppsFloat64        = nppsallocmode(12)
	nppsFloat64Complex = nppsallocmode(13)
)
*/
