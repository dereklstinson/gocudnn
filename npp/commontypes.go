package npp

/*
#include <npps_conversion_functions.h>
#include <nppdefs.h>
*/
import "C"
import (
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
)

/*
 *
 * Uint8
 *
 */

//Uint8 holds an unsafe pointer to convert to Uint8
type Uint8 struct {
	p unsafe.Pointer
}

func (n *Uint8) c() C.Npp8u {
	x := (*C.Npp8u)(n.p)
	return *x
}
func (n *Uint8) cptr() *C.Npp8u {
	return (*C.Npp8u)(n.p)

}
func (n *Uint8) DPtr() *unsafe.Pointer {
	return &n.p
}
func (n *Uint8) Ptr() unsafe.Pointer {
	return (n.p)

}
func (n *Uint8) wrap(p *C.Npp8u) {
	n.p = unsafe.Pointer(p)
}
func (n *Uint8) Offset(elements int32) *Uint8 {

	mem := gocu.Offset(n, (uint)(elements))
	return &Uint8{
		p: mem.Ptr(),
	}
}

//Uint16 is a uint16.  A pointer of this type could be in cuda memory.
type Uint16 C.Npp16u /**<  16-bit unsigned integers */

func (n *Uint16) cptr() *C.Npp16u {
	return (*C.Npp16u)(n)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Uint16) Ptr() unsafe.Pointer {
	return unsafe.Pointer(n)
}

/*
//DPtr returns an double pointer used for allocating memory on device
func (n *Uint16) DPtr() *unsafe.Pointer {
	x := unsafe.Pointer(n)
	return (*unsafe.Pointer)(&x)
}
*/
func (n Uint16) c() C.Npp16u {
	return C.Npp16u(n)
}

//Int8 holds an unsafe pointer to convert to Int8
type Int8 struct {
	p unsafe.Pointer //C.Npp8s
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

func (n *Int8) Offset(elements int32) *Int8 {

	mem := gocu.Offset(n, (uint)(elements))
	return &Int8{
		p: mem.Ptr(),
	}
}

type Int16 struct {
	p unsafe.Pointer //C.Npp8s
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

func (n *Int16) Offset(elements int32) *Int16 {

	mem := gocu.Offset(n, (uint)(elements))
	return &Int16{
		p: mem.Ptr(),
	}
}
