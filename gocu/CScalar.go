package gocu

import (
	"C"
	"unsafe"

	"github.com/dereklstinson/half"
)

//CScalar is used for scalar multiplications with cudnn.  They have to be Ctypes. It could have easily been called voider
type CScalar interface {
	CPtr() unsafe.Pointer
	Bytes() int
	SizeT() uint
}

//CScalartoFloat64 changes a CScalar to a float64 value so it could be read or debugging.
func CScalartoFloat64(x CScalar) float64 {
	switch y := x.(type) {
	case CDouble:
		return float64(y)
	case CFloat:
		return float64(y)
	case CInt:
		return float64(y)
	case CUInt:
		return float64(y)
	case CHalf:
		h := (half.Float16)(y)
		return float64(h)
	case CInt8:
		return float64(y)
	case CUInt8:
		return float64(y)

	}
	panic("Unsupported val for CScalartoFloat64")

}

//CScalarConversion takes a go type and converts it to a CScalar interface. golang type int and int32 will both be converted to a CInt type.
//If a go type is not supported then it will return a nil.
//Current support is float64,float32,int, int32, int8,uint32, uint, uint8 ( I think byte should work because when I put it in the switch with uint8 it says duplicate type).
func CScalarConversion(gotype interface{}) CScalar {
	switch x := gotype.(type) {
	case float64:
		return CDouble(x)
	case float32:
		return CFloat(x)
	case int:
		return CInt(x)
	case int32:
		return CInt(x)
	case int8:
		return CInt8(x)
	case uint8:
		return CUInt8(x)
	case uint32:
		return CUInt(x)
	case uint:
		return CUInt(x)
	case half.Float16:
		return CHalf(x)
	case bool:

		if x == true {
			return CInt(255)
		}
		return CInt(0)

	case CScalar:
		return x

	default:
		return nil
	}
}

//CHalf is a half precision
type CHalf C.ushort

func (f CHalf) c() C.ushort { return C.ushort(f) }

//CPtr returns an unsafe pointer of the half
func (f CHalf) CPtr() unsafe.Pointer { return unsafe.Pointer(&f) }

//Bytes returns the number of bytes the CScalar has as an int
func (f CHalf) Bytes() int { return 4 }

//SizeT returns the number of bytes the CScalar has as an sizeT
func (f CHalf) SizeT() uint { return (4) }

//CFloat is a float in C
type CFloat C.float

func (f CFloat) c() C.float { return C.float(f) }

//CPtr returns an unsafe pointer of the float
func (f CFloat) CPtr() unsafe.Pointer { return unsafe.Pointer(&f) }

//Bytes returns the number of bytes the CScalar has
func (f CFloat) Bytes() int { return 4 }

//SizeT returns the number of bytes the CScalar has
func (f CFloat) SizeT() uint { return (4) }

//CDouble is a double in C
type CDouble C.double

func (d CDouble) c() C.double { return C.double(d) }

//CPtr returns an unsafe pointer of the double
func (d CDouble) CPtr() unsafe.Pointer { return unsafe.Pointer(&d) }

//Bytes returns the number of bytes the CScalar has
func (d CDouble) Bytes() int { return 8 }

//SizeT returns the number of bytes the CScalar has
func (d CDouble) SizeT() uint { return (8) }

//CInt is a int in C
type CInt C.int

func (i CInt) c() C.int { return C.int(i) }

//CPtr returns an unsafe pointer of the int
func (i CInt) CPtr() unsafe.Pointer { return unsafe.Pointer(&i) }

//Bytes returns the number of bytes the CScalar has
func (i CInt) Bytes() int { return 4 }

//SizeT returns the number of bytes the CScalar has
func (i CInt) SizeT() uint { return (4) }

//CUInt is an unsigned int in C
type CUInt C.uint

//CPtr returns an unsafe pointer of the Unsigned Int
func (i CUInt) CPtr() unsafe.Pointer { return unsafe.Pointer(&i) }

//Bytes returns the number of bytes the CScalar has
func (i CUInt) Bytes() int { return 4 }
func (i CUInt) c() C.uint  { return C.uint(i) }

//SizeT returns the number of bytes the CScalar has
func (i CUInt) SizeT() uint { return (4) }

//CInt8 is a signed char
type CInt8 C.char

func (c CInt8) c() C.char { return C.char(c) }

//CPtr retunrs an unsafe pointer for CInt8
func (c CInt8) CPtr() unsafe.Pointer { return unsafe.Pointer(&c) }

//Bytes returns the number of bytes the CScalar has
func (c CInt8) Bytes() int { return 1 }

//SizeT returns the number of bytes the CScalar has
func (c CInt8) SizeT() uint { return (1) }

//CUInt8 is a C.uchar
type CUInt8 C.uchar

func (c CUInt8) c() C.uchar { return C.uchar(c) }

//Bytes returns the number of bytes the CScalar has
func (c CUInt8) Bytes() int { return 1 }

//CPtr retunrs an unsafe pointer for CUInt8
func (c CUInt8) CPtr() unsafe.Pointer { return unsafe.Pointer(&c) }

//SizeT returns the number of bytes the CScalar has
func (c CUInt8) SizeT() uint { return (1) }
