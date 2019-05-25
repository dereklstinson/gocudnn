package npp

/*
//#include <npps_conversion_functions.h>
#include <nppdefs.h>
#include <npps_initialization.h>
*/
import "C"
import (
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
)

/*
 *
 * Unsigned
 *
 */

//Uint8 holds an unsafe pointer to convert to Uint8
type Uint8 struct {
	p unsafe.Pointer
}

//MakeUint8FromUnsafe will make a *Uint8 from an unsafe.Pointer
func MakeUint8FromUnsafe(p unsafe.Pointer) *Uint8 {
	return &Uint8{
		p: p,
	}
}

//Set sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Uint8) Set(val uint8, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_8u_Ctx((C.Npp8u)(val), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_8u((C.Npp8u)(val), n.cptr(), (C.int)(length))).error()
}
func (n *Uint8) c() C.Npp8u {
	x := (*C.Npp8u)(n.p)
	return *x
}
func (n *Uint8) cptr() *C.Npp8u {
	return (*C.Npp8u)(n.p)

}

//DPtr returns the double pointer
func (n *Uint8) DPtr() *unsafe.Pointer {
	return &n.p
}

//Ptr returns an unsafe pointer
func (n *Uint8) Ptr() unsafe.Pointer {
	return (n.p)

}
func (n *Uint8) wrap(p *C.Npp8u) {
	n.p = unsafe.Pointer(p)
}

//Offset returns the offset pointer
func (n *Uint8) Offset(elements int32) *Uint8 {

	mem := gocu.Offset(n, (uint)(elements))
	return &Uint8{
		p: mem.Ptr(),
	}
}

//Uint16 holds an unsafe pointer to convert to Uint16
type Uint16 struct {
	p unsafe.Pointer
}

//MakeUint16FromUnsafe will make a *Uint16 from an unsafe.Pointer
func MakeUint16FromUnsafe(p unsafe.Pointer) *Uint16 {
	return &Uint16{
		p: p,
	}
}

//Set sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Uint16) Set(val uint16, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_16u_Ctx((C.Npp16u)(val), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_16u((C.Npp16u)(val), n.cptr(), (C.int)(length))).error()
}

func (n *Uint16) c() C.Npp16u {
	x := (*C.Npp16u)(n.p)
	return *x
}
func (n *Uint16) cptr() *C.Npp16u {
	return (*C.Npp16u)(n.p)

}

//DPtr returns the double pointer
func (n *Uint16) DPtr() *unsafe.Pointer {
	return &n.p
}

//Ptr returns a pointer
func (n *Uint16) Ptr() unsafe.Pointer {
	return (n.p)

}
func (n *Uint16) wrap(p *C.Npp16u) {
	n.p = unsafe.Pointer(p)
}

//Offset returns the offset pointer
func (n *Uint16) Offset(elements int32) *Uint16 {

	mem := gocu.Offset(n, (uint)(elements*2))
	return &Uint16{
		p: mem.Ptr(),
	}
}

//Uint32 holds an unsafe pointer to convert to Uint32
type Uint32 struct {
	p unsafe.Pointer
}

//MakeUint32FromUnsafe will make a *Uint32 from an unsafe.Pointer
func MakeUint32FromUnsafe(p unsafe.Pointer) *Uint32 {
	return &Uint32{
		p: p,
	}
}

//Set sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Uint32) Set(val uint32, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_32u_Ctx((C.Npp32u)(val), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_32u((C.Npp32u)(val), n.cptr(), (C.int)(length))).error()
}
func (n *Uint32) c() C.Npp32u {
	x := (*C.Npp32u)(n.p)
	return *x
}
func (n *Uint32) cptr() *C.Npp32u {
	return (*C.Npp32u)(n.p)

}

//DPtr returns the double pointer
func (n *Uint32) DPtr() *unsafe.Pointer {
	return &n.p
}

//Ptr returns a pointer
func (n *Uint32) Ptr() unsafe.Pointer {
	return (n.p)

}
func (n *Uint32) wrap(p *C.Npp32u) {
	n.p = unsafe.Pointer(p)
}

//Offset returns the offset pointer
func (n *Uint32) Offset(elements int32) *Uint32 {

	mem := gocu.Offset(n, (uint)(elements*4))
	return &Uint32{
		p: mem.Ptr(),
	}
}

//Uint64 holds an unsafe pointer to convert to Uint64
type Uint64 struct {
	p unsafe.Pointer
}

//MakeUint64FromUnsafe will make a *Uint64 from an unsafe.Pointer
func MakeUint64FromUnsafe(p unsafe.Pointer) *Uint64 {
	return &Uint64{
		p: p,
	}
}
func (n *Uint64) c() C.Npp64u {
	x := (*C.Npp64u)(n.p)
	return *x
}
func (n *Uint64) cptr() *C.Npp64u {
	return (*C.Npp64u)(n.p)

}

//DPtr returns the double pointer
func (n *Uint64) DPtr() *unsafe.Pointer {
	return &n.p
}

//Ptr returns an unsafe pointer
func (n *Uint64) Ptr() unsafe.Pointer {
	return (n.p)

}
func (n *Uint64) wrap(p *C.Npp64u) {
	n.p = unsafe.Pointer(p)
}

//Offset returns the offset pointer
func (n *Uint64) Offset(elements int32) *Uint64 {

	mem := gocu.Offset(n, (uint)(elements*8))
	return &Uint64{
		p: mem.Ptr(),
	}
}

/*
*
*
*		Signed
*
*
 */

//Int8 holds an unsafe pointer to convert to Int8
type Int8 struct {
	p unsafe.Pointer
}

//MakeInt8FromUnsafe will make a *Int8 from an unsafe.Pointer
func MakeInt8FromUnsafe(p unsafe.Pointer) *Int8 {
	return &Int8{
		p: p,
	}
}

//Set sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Int8) Set(val int8, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_8s_Ctx((C.Npp8s)(val), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_8s((C.Npp8s)(val), n.cptr(), (C.int)(length))).error()
}
func (n *Int8) cptr() *C.Npp8s {
	return (*C.Npp8s)(n.p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Int8) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an double pointer used for allocating memory on device
func (n *Int8) DPtr() *unsafe.Pointer {
	return &n.p
}

func (n *Int8) c() C.Npp8s {
	x := (*C.Npp8s)(n.p)
	return *x
}
func (n *Int8) wrap(p *C.Npp8s) {
	n.p = unsafe.Pointer(p)
}

//Offset returns the offset pointer
func (n *Int8) Offset(elements int32) *Int8 {

	mem := gocu.Offset(n, (uint)(elements))
	return &Int8{
		p: mem.Ptr(),
	}
}

//Int16 wraps an unsafe pointer for an Int16
type Int16 struct {
	p unsafe.Pointer
}

//MakeInt16FromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeInt16FromUnsafe(p unsafe.Pointer) *Int16 {
	return &Int16{
		p: p,
	}
}

//Set sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Int16) Set(val int16, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_16s_Ctx((C.Npp16s)(val), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_16s((C.Npp16s)(val), n.cptr(), (C.int)(length))).error()
}
func (n *Int16) cptr() *C.Npp16s {
	return (*C.Npp16s)(n.p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Int16) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an double pointer used for allocating memory on device
func (n *Int16) DPtr() *unsafe.Pointer {
	return &n.p
}

func (n *Int16) c() C.Npp16s {
	x := (*C.Npp16s)(n.p)
	return *x
}
func (n *Int16) wrap(p *C.Npp16s) {
	n.p = unsafe.Pointer(p)
}

//Offset returns the offset pointer of Int16 Pointer
func (n *Int16) Offset(elements int32) *Int16 {

	mem := gocu.Offset(n, (uint)(elements*2))
	return &Int16{
		p: mem.Ptr(),
	}
}

//Int32 wraps an unsafe pointer for an Int32
type Int32 struct {
	p unsafe.Pointer
}

//MakeInt32FromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeInt32FromUnsafe(p unsafe.Pointer) *Int32 {
	return &Int32{
		p: p,
	}
}

//Set sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Int32) Set(val int32, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_32s_Ctx((C.Npp32s)(val), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_32s((C.Npp32s)(val), n.cptr(), (C.int)(length))).error()
}
func (n *Int32) cptr() *C.Npp32s {
	return (*C.Npp32s)(n.p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Int32) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an double pointer used for allocating memory on device
func (n *Int32) DPtr() *unsafe.Pointer {
	return &n.p
}

func (n *Int32) c() C.Npp32s {
	x := (*C.Npp32s)(n.p)
	return *x
}
func (n *Int32) wrap(p *C.Npp32s) {
	n.p = unsafe.Pointer(p)
}

//Offset returns the offset pointer
func (n *Int32) Offset(elements int32) *Int32 {

	mem := gocu.Offset(n, (uint)(elements*4))
	return &Int32{
		p: mem.Ptr(),
	}
}

//Int64 wraps an unsafe pointer for an Int64
type Int64 struct {
	p unsafe.Pointer
}

//MakeInt64FromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeInt64FromUnsafe(p unsafe.Pointer) *Int64 {
	return &Int64{
		p: p,
	}
}

//Set sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Int64) Set(val int64, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_64s_Ctx((C.Npp64s)(val), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_64s((C.Npp64s)(val), n.cptr(), (C.int)(length))).error()
}
func (n *Int64) cptr() *C.Npp64s {
	return (*C.Npp64s)(n.p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Int64) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an double pointer used for allocating memory on device
func (n *Int64) DPtr() *unsafe.Pointer {
	return &n.p
}

func (n *Int64) c() C.Npp64s {
	x := (*C.Npp64s)(n.p)
	return *x
}
func (n *Int64) wrap(p *C.Npp64s) {
	n.p = unsafe.Pointer(p)
}

//Offset returns the offset pointer
func (n *Int64) Offset(elements int32) *Int64 {

	mem := gocu.Offset(n, (uint)(elements*8))
	return &Int64{
		p: mem.Ptr(),
	}
}

/*
*
* Floating Point
*
 */

//Float32 wraps an unsafe pointer for an Float32
type Float32 struct {
	p unsafe.Pointer
}

//MakeFloat32FromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeFloat32FromUnsafe(p unsafe.Pointer) *Float32 {
	return &Float32{
		p: p,
	}
}

//Set sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Float32) Set(val float32, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_32f_Ctx((C.Npp32f)(val), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_32f((C.Npp32f)(val), n.cptr(), (C.int)(length))).error()
}
func (n *Float32) c() C.Npp32f {
	x := (*C.Npp32f)(n.p)
	return *x
}
func (n *Float32) cptr() *C.Npp32f {
	return (*C.Npp32f)(n.p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Float32) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an double pointer used for allocating memory on device
func (n *Float32) DPtr() *unsafe.Pointer {
	return &n.p
}

func (n *Float32) wrap(p *C.Npp32f) {
	n.p = unsafe.Pointer(p)
}

//Offset returns the offset pointer
func (n *Float32) Offset(elements int32) *Float32 {

	mem := gocu.Offset(n, (uint)(elements*4))
	return &Float32{
		p: mem.Ptr(),
	}
}

//Float64 wraps an unsafe pointer for an Float32
type Float64 struct {
	p unsafe.Pointer
}

//MakeFloat64FromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeFloat64FromUnsafe(p unsafe.Pointer) *Float64 {
	return &Float64{
		p: p,
	}
}

//Set sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Float64) Set(val float64, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_64f_Ctx((C.Npp64f)(val), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_64f((C.Npp64f)(val), n.cptr(), (C.int)(length))).error()
}
func (n *Float64) c() C.Npp64f {
	x := (*C.Npp64f)(n.p)
	return *x
}
func (n *Float64) cptr() *C.Npp64f {
	return (*C.Npp64f)(n.p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Float64) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an double pointer used for allocating memory on device
func (n *Float64) DPtr() *unsafe.Pointer {
	return &n.p
}

func (n *Float64) wrap(p *C.Npp64f) {
	n.p = unsafe.Pointer(p)
}

//Offset returns the offset pointer
func (n *Float64) Offset(elements int32) *Float64 {

	mem := gocu.Offset(n, (uint)(elements*8))
	return &Float64{
		p: mem.Ptr(),
	}
}

/*
*
* Complex Unsigned
*
 */

//Uint8Complex is a complex uint8
type Uint8Complex struct {
	p unsafe.Pointer
}

//MakeUint8ComplexFromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeUint8ComplexFromUnsafe(p unsafe.Pointer) *Uint8Complex {
	return &Uint8Complex{
		p: p,
	}
}
func (n *Uint8Complex) c() C.Npp8uc {
	x := (*C.Npp8uc)(n.p)
	return *x
}
func (n *Uint8Complex) cptr() *C.Npp8uc {
	return (*C.Npp8uc)(n.p)
}

func (n *Uint8Complex) wrap(p *C.Npp8uc) {
	n.p = unsafe.Pointer(p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Uint8Complex) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an *unsafe pointer to this variable location.
func (n *Uint8Complex) DPtr() *unsafe.Pointer {
	return &n.p
}

//Set sets the real and imaginary vals
func (n *Uint8Complex) Set(real, imaginary uint8) {
	x := (*C.Npp8uc)(n.p)
	x.re = (C.Npp8u)(real)
	x.im = (C.Npp8u)(imaginary)

}

//Get gets the real and imaginary vals
func (n *Uint8Complex) Get() (real, imaginary uint8) {
	x := (*C.Npp8uc)(n.p)
	real = (uint8)(x.re)
	imaginary = (uint8)(x.im)
	return real, imaginary
}

//Offset returns the offset pointer
func (n *Uint8Complex) Offset(elements int32) *Uint8Complex {

	mem := gocu.Offset(n, (uint)(elements*2))
	return &Uint8Complex{
		p: mem.Ptr(),
	}
}

/*Uint16Complex - See below
 * Complex Number
 * This struct represents an unsigned short complex number.
 */
type Uint16Complex struct {
	p unsafe.Pointer
}

//MakeUint16ComplexFromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeUint16ComplexFromUnsafe(p unsafe.Pointer) *Uint16Complex {
	return &Uint16Complex{
		p: p,
	}
}
func (n *Uint16Complex) c() C.Npp16uc {
	x := (*C.Npp16uc)(n.p)
	return *x
}
func (n *Uint16Complex) cptr() *C.Npp16uc {
	return (*C.Npp16uc)(n.p)
}
func (n *Uint16Complex) wrap(p *C.Npp16uc) {
	n.p = unsafe.Pointer(p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Uint16Complex) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an *unsafe pointer to this variable location.
func (n *Uint16Complex) DPtr() *unsafe.Pointer {
	return &n.p
}

//Set sets the real and imaginary vals
func (n *Uint16Complex) Set(real, imaginary uint16) {
	x := (*C.Npp16uc)(n.p)
	x.re = (C.Npp16u)(real)
	x.im = (C.Npp16u)(imaginary)

}

//Get gets the real and imaginary vals
func (n *Uint16Complex) Get() (real, imaginary uint16) {
	x := (*C.Npp16uc)(n.p)
	real = (uint16)(x.re)
	imaginary = (uint16)(x.im)
	return real, imaginary
}

//Offset returns the offset pointer
func (n *Uint16Complex) Offset(elements int32) *Uint16Complex {

	mem := gocu.Offset(n, (uint)(elements*4))
	return &Uint16Complex{
		p: mem.Ptr(),
	}
}

/*Uint32Complex - See below
 * Complex Number
 * This struct represents an unsigned short complex number.
 */
type Uint32Complex struct {
	p unsafe.Pointer
}

//MakeUint32ComplexFromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeUint32ComplexFromUnsafe(p unsafe.Pointer) *Uint32Complex {
	return &Uint32Complex{
		p: p,
	}
}
func (n *Uint32Complex) c() C.Npp32uc {
	x := (*C.Npp32uc)(n.p)
	return *x
}
func (n *Uint32Complex) cptr() *C.Npp32uc {
	return (*C.Npp32uc)(n.p)
}
func (n *Uint32Complex) wrap(p *C.Npp32uc) {
	n.p = unsafe.Pointer(p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Uint32Complex) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an *unsafe pointer to this variable location.
func (n *Uint32Complex) DPtr() *unsafe.Pointer {
	return &n.p
}

//Set sets the real and imaginary vals
func (n *Uint32Complex) Set(real, imaginary uint32) {
	x := (*C.Npp32uc)(n.p)
	x.re = (C.Npp32u)(real)
	x.im = (C.Npp32u)(imaginary)

}

//Get gets the real and imaginary vals
func (n *Uint32Complex) Get() (real, imaginary uint32) {
	x := (*C.Npp32uc)(n.p)
	real = (uint32)(x.re)
	imaginary = (uint32)(x.im)
	return real, imaginary
}

//Offset returns the offset pointer
func (n *Uint32Complex) Offset(elements int32) *Uint32Complex {

	mem := gocu.Offset(n, (uint)(elements*8))
	return &Uint32Complex{
		p: mem.Ptr(),
	}
}

/*
*
*Complex Signed
*
 */

/*Int16Complex - See below
 * * Complex Number
 * This struct represents a short complex number.
 */
type Int16Complex struct {
	p unsafe.Pointer
}

//MakeInt16ComplexFromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeInt16ComplexFromUnsafe(p unsafe.Pointer) *Int16Complex {
	return &Int16Complex{
		p: p,
	}
}

//SetGPU sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Int16Complex) SetGPU(val Int16Complex, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_16sc_Ctx(val.c(), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_16sc(val.c(), n.cptr(), (C.int)(length))).error()
}

func (n *Int16Complex) c() C.Npp16sc {
	x := (*C.Npp16sc)(n.p)
	return *x
}
func (n *Int16Complex) cptr() *C.Npp16sc {
	return (*C.Npp16sc)(n.p)
}
func (n *Int16Complex) wrap(p *C.Npp16sc) {
	n.p = unsafe.Pointer(p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Int16Complex) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an *unsafe pointer to this variable location.
func (n *Int16Complex) DPtr() *unsafe.Pointer {
	return &n.p
}

//Set sets the real and imaginary vals
func (n *Int16Complex) Set(real, imaginary int16) {
	x := (*C.Npp16sc)(n.p)
	x.re = (C.Npp16s)(real)
	x.im = (C.Npp16s)(imaginary)

}

//Get gets the real and imaginary vals
func (n *Int16Complex) Get() (real, imaginary int16) {
	x := (*C.Npp16sc)(n.p)
	real = (int16)(x.re)
	imaginary = (int16)(x.im)
	return real, imaginary
}

//Offset returns the offset pointer
func (n *Int16Complex) Offset(elements int32) *Int16Complex {

	mem := gocu.Offset(n, (uint)(elements*4))
	return &Int16Complex{
		p: mem.Ptr(),
	}
}

/*Int32Complex - See below
 * * Complex Number
 * This struct represents a short complex number.
 */
type Int32Complex struct {
	p unsafe.Pointer
}

//MakeInt32ComplexFromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeInt32ComplexFromUnsafe(p unsafe.Pointer) *Int32Complex {
	return &Int32Complex{
		p: p,
	}
}

//SetGPU sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Int32Complex) SetGPU(val Int32Complex, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_32sc_Ctx(val.c(), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_32sc(val.c(), n.cptr(), (C.int)(length))).error()
}

func (n *Int32Complex) c() C.Npp32sc {
	x := (*C.Npp32sc)(n.p)
	return *x
}
func (n *Int32Complex) cptr() *C.Npp32sc {
	return (*C.Npp32sc)(n.p)
}
func (n *Int32Complex) wrap(p *C.Npp32sc) {
	n.p = unsafe.Pointer(p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Int32Complex) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an *unsafe pointer to this variable location.
func (n *Int32Complex) DPtr() *unsafe.Pointer {
	return &n.p
}

//Set sets the real and imaginary vals
func (n *Int32Complex) Set(real, imaginary int32) {
	x := (*C.Npp32sc)(n.p)
	x.re = (C.Npp32s)(real)
	x.im = (C.Npp32s)(imaginary)

}

//Get gets the real and imaginary vals
func (n *Int32Complex) Get() (real, imaginary int32) {
	x := (*C.Npp32sc)(n.p)
	real = (int32)(x.re)
	imaginary = (int32)(x.im)
	return real, imaginary
}

//Offset returns the offset pointer
func (n *Int32Complex) Offset(elements int32) *Int32Complex {

	mem := gocu.Offset(n, (uint)(elements*8))
	return &Int32Complex{
		p: mem.Ptr(),
	}
}

/*Int64Complex - See below
 * * Complex Number
 * This struct represents a short complex number.
 */
type Int64Complex struct {
	p unsafe.Pointer
}

//MakeInt64ComplexFromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeInt64ComplexFromUnsafe(p unsafe.Pointer) *Int64Complex {
	return &Int64Complex{
		p: p,
	}
}

//SetGPU sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Int64Complex) SetGPU(val Int64Complex, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_64sc_Ctx(val.c(), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_64sc(val.c(), n.cptr(), (C.int)(length))).error()
}
func (n *Int64Complex) c() C.Npp64sc {
	x := (*C.Npp64sc)(n.p)
	return *x
}
func (n *Int64Complex) cptr() *C.Npp64sc {
	return (*C.Npp64sc)(n.p)
}
func (n *Int64Complex) wrap(p *C.Npp64sc) {
	n.p = unsafe.Pointer(p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Int64Complex) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an *unsafe pointer to this variable location.
func (n *Int64Complex) DPtr() *unsafe.Pointer {
	return &n.p
}

//Set sets the real and imaginary vals
func (n *Int64Complex) Set(real, imaginary int64) {
	x := (*C.Npp64sc)(n.p)
	x.re = (C.Npp64s)(real)
	x.im = (C.Npp64s)(imaginary)

}

//Get gets the real and imaginary vals
func (n *Int64Complex) Get() (real, imaginary int64) {
	x := (*C.Npp64sc)(n.p)
	real = (int64)(x.re)
	imaginary = (int64)(x.im)
	return real, imaginary
}

//Offset returns the offset pointer
func (n *Int64Complex) Offset(elements int32) *Int64Complex {

	mem := gocu.Offset(n, (uint)(elements*16))
	return &Int64Complex{
		p: mem.Ptr(),
	}
}

/*
*
* Floating Point Complex
*
 */

/*Float32Complex - See below
 * * Complex Number
 * This struct represents a short complex number.
 */
type Float32Complex struct {
	p unsafe.Pointer
}

//MakeFloat32ComplexFromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeFloat32ComplexFromUnsafe(p unsafe.Pointer) *Float32Complex {
	return &Float32Complex{
		p: p,
	}
}

//SetGPU sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Float32Complex) SetGPU(val Float32Complex, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_32fc_Ctx(val.c(), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_32fc(val.c(), n.cptr(), (C.int)(length))).error()
}

func (n *Float32Complex) c() C.Npp32fc {
	x := (*C.Npp32fc)(n.p)
	return *x
}
func (n *Float32Complex) cptr() *C.Npp32fc {
	return (*C.Npp32fc)(n.p)
}
func (n *Float32Complex) wrap(p *C.Npp32fc) {
	n.p = unsafe.Pointer(p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Float32Complex) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an *unsafe pointer to this variable location.
func (n *Float32Complex) DPtr() *unsafe.Pointer {
	return &n.p
}

//Set sets the real and imaginary vals
func (n *Float32Complex) Set(real, imaginary float32) {
	x := (*C.Npp32fc)(n.p)
	x.re = (C.Npp32f)(real)
	x.im = (C.Npp32f)(imaginary)

}

//Get gets the real and imaginary vals
func (n *Float32Complex) Get() (real, imaginary float32) {
	x := (*C.Npp32fc)(n.p)
	real = (float32)(x.re)
	imaginary = (float32)(x.im)
	return real, imaginary
}

//Offset returns the offset pointer
func (n *Float32Complex) Offset(elements int32) *Float32Complex {

	mem := gocu.Offset(n, (uint)(elements*8))
	return &Float32Complex{
		p: mem.Ptr(),
	}
}

/*Float64Complex - See below
 * * Complex Number
 * This struct represents a short complex number.
 */
type Float64Complex struct {
	p unsafe.Pointer
}

//MakeFloat64ComplexFromUnsafe will make a *Int16 from an unsafe.Pointer
func MakeFloat64ComplexFromUnsafe(p unsafe.Pointer) *Float64Complex {
	return &Float64Complex{
		p: p,
	}
}

//SetGPU sets the value passed to the number of elements defined in length. n needs to be pre allocated already.
func (n *Float64Complex) SetGPU(val Float64Complex, length int32, ctx *StreamContext) (err error) {
	if ctx != nil {
		return status(C.nppsSet_64fc_Ctx(val.c(), n.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsSet_64fc(val.c(), n.cptr(), (C.int)(length))).error()
}
func (n *Float64Complex) c() C.Npp64fc {
	x := (*C.Npp64fc)(n.p)
	return *x
}
func (n *Float64Complex) cptr() *C.Npp64fc {
	return (*C.Npp64fc)(n.p)
}
func (n *Float64Complex) wrap(p *C.Npp64fc) {
	n.p = unsafe.Pointer(p)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Float64Complex) Ptr() unsafe.Pointer {
	return n.p
}

//DPtr returns an *unsafe pointer to this variable location.
func (n *Float64Complex) DPtr() *unsafe.Pointer {
	return &n.p
}

//Set sets the real and imaginary vals
func (n *Float64Complex) Set(real, imaginary float32) {
	x := (*C.Npp64fc)(n.p)
	x.re = (C.Npp64f)(real)
	x.im = (C.Npp64f)(imaginary)

}

//Get gets the real and imaginary vals
func (n *Float64Complex) Get() (real, imaginary float32) {
	x := (*C.Npp64fc)(n.p)
	real = (float32)(x.re)
	imaginary = (float32)(x.im)
	return real, imaginary
}

//Offset returns the offset pointer
func (n *Float64Complex) Offset(elements int32) *Float64Complex {

	mem := gocu.Offset(n, (uint)(elements*16))
	return &Float64Complex{
		p: mem.Ptr(),
	}

}

//CopyUint8 copies src to dst with the length passed
func CopyUint8(src, dst *Uint8, length int32, ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppsCopy_8u_Ctx(src.cptr(), dst.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsCopy_8u(src.cptr(), dst.cptr(), (C.int)(length))).error()
}

//CopyInt16 copies src to dst with the length passed
func CopyInt16(src, dst *Int16, length int32, ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppsCopy_16s_Ctx(src.cptr(), dst.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsCopy_16s(src.cptr(), dst.cptr(), (C.int)(length))).error()
}

//CopyInt32 copies src to dst with the length passed
func CopyInt32(src, dst *Int32, length int32, ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppsCopy_32s_Ctx(src.cptr(), dst.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsCopy_32s(src.cptr(), dst.cptr(), (C.int)(length))).error()
}

//CopyFloat32 copies src to dst with the length passed
func CopyFloat32(src, dst *Float32, length int32, ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppsCopy_32f_Ctx(src.cptr(), dst.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsCopy_32f(src.cptr(), dst.cptr(), (C.int)(length))).error()
}

//CopyInt64 copies src to dst with the length passed
func CopyInt64(src, dst *Int64, length int32, ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppsCopy_64s_Ctx(src.cptr(), dst.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsCopy_64s(src.cptr(), dst.cptr(), (C.int)(length))).error()
}

//CopyInt16Complex copies src to dst with the length passed
func CopyInt16Complex(src, dst *Int16Complex, length int32, ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppsCopy_16sc_Ctx(src.cptr(), dst.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsCopy_16sc(src.cptr(), dst.cptr(), (C.int)(length))).error()
}

//CopyInt32Complex copies src to dst with the length passed
func CopyInt32Complex(src, dst *Int32Complex, length int32, ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppsCopy_32sc_Ctx(src.cptr(), dst.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsCopy_32sc(src.cptr(), dst.cptr(), (C.int)(length))).error()
}

//CopyInt64Complex copies src to dst with the length passed
func CopyInt64Complex(src, dst *Int64Complex, length int32, ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppsCopy_64sc_Ctx(src.cptr(), dst.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsCopy_64sc(src.cptr(), dst.cptr(), (C.int)(length))).error()
}

//CopyFloat32Complex copies src to dst with the length passed
func CopyFloat32Complex(src, dst *Float32Complex, length int32, ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppsCopy_32fc_Ctx(src.cptr(), dst.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsCopy_32fc(src.cptr(), dst.cptr(), (C.int)(length))).error()
}

//CopyFloat64Complex copies src to dst with the length passed
func CopyFloat64Complex(src, dst *Float64Complex, length int32, ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppsCopy_64fc_Ctx(src.cptr(), dst.cptr(), (C.int)(length), ctx.c())).error()
	}
	return status(C.nppsCopy_64fc(src.cptr(), dst.cptr(), (C.int)(length))).error()
}
