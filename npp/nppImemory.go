package npp

/*
#include <nppi_support_functions.h>
#include <nppdefs.h>
*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

/*

Uint8


*/

/*Malloc8uC1 - 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc8uC1(nWidthPixels int32, nHeightPixels int32) (ptr *Uint8, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = new(Uint8)
	ptr.wrap(C.nppiMalloc_8u_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

/*Malloc8uC2 - 2 channel 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc8uC2(nWidthPixels int32, nHeightPixels int32) (ptr *Uint8, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = new(Uint8)
	ptr.wrap(C.nppiMalloc_8u_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

/*Malloc8uC3 - 3 channel 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc8uC3(nWidthPixels int32, nHeightPixels int32) (ptr *Uint8, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = new(Uint8)
	ptr.wrap(C.nppiMalloc_8u_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

/*Malloc8uC4 - 4 channel 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc8uC4(nWidthPixels int32, nHeightPixels int32) (ptr *Uint8, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = new(Uint8)
	ptr.wrap(C.nppiMalloc_8u_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)

	return ptr, int32(pStepBytes)
}

/*


Uint16


*/

/*Malloc16uC1 - 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16uC1(nWidthPixels int32, nHeightPixels int32) (ptr *Uint16, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Uint16)(C.nppiMalloc_16u_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16uC2 - 2 channel 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16uC2(nWidthPixels int32, nHeightPixels int32) (ptr *Uint16, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Uint16)(C.nppiMalloc_16u_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16uC3 - 3 channel 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16uC3(nWidthPixels int32, nHeightPixels int32) (ptr *Uint16, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Uint16)(C.nppiMalloc_16u_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16uC4 - 4 channel 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16uC4(nWidthPixels int32, nHeightPixels int32) (ptr *Uint16, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Uint16)(C.nppiMalloc_16u_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*


Npp16s


*/

/*Malloc16sC1 - 1 channel 16-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16sC1(nWidthPixels int32, nHeightPixels int32) (ptr *Int16, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int16)(C.nppiMalloc_16s_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16sC2 - 2 channel 16-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16sC2(nWidthPixels int32, nHeightPixels int32) (ptr *Int16, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int16)(C.nppiMalloc_16s_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16sC4 - 4 channel 16-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16sC4(nWidthPixels int32, nHeightPixels int32) (ptr *Int16, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int16)(C.nppiMalloc_16s_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*


Int16c


*/

/*Malloc16scC1 - 1 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16scC1(nWidthPixels int32, nHeightPixels int32) (ptr *Int16Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int16Complex)(C.nppiMalloc_16sc_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))

	//y := MakeNpp16scFromUnsafe(unsafe.Pointer(x), false)
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16scC2 - 2 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16scC2(nWidthPixels int32, nHeightPixels int32) (ptr *Int16Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int16Complex)(C.nppiMalloc_16sc_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16scC3 - 3 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16scC3(nWidthPixels int32, nHeightPixels int32) (ptr *Int16Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int16Complex)(C.nppiMalloc_16sc_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16scC4 - 4 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16scC4(nWidthPixels int32, nHeightPixels int32) (ptr *Int16Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int16Complex)(C.nppiMalloc_16sc_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*


Int32


*/

/*Malloc32sC1 - 1 channel 32-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32sC1(nWidthPixels int32, nHeightPixels int32) (ptr *Int32, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int32)(C.nppiMalloc_32s_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32sC3 - 3 channel 32-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32sC3(nWidthPixels int32, nHeightPixels int32) (ptr *Int32, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int32)(C.nppiMalloc_32s_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32sC4 - 4 channel 32-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32sC4(nWidthPixels int32, nHeightPixels int32) (ptr *Int32, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int32)(C.nppiMalloc_32s_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*


Int32Complex


*/

/*Malloc32scC1 - 1 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32scC1(nWidthPixels int32, nHeightPixels int32) (ptr *Int32Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int32Complex)(C.nppiMalloc_32sc_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32scC2 - 2 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32scC2(nWidthPixels int32, nHeightPixels int32) (ptr *Int32Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int32Complex)(C.nppiMalloc_32sc_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32scC3 - 3 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32scC3(nWidthPixels int32, nHeightPixels int32) (ptr *Int32Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int32Complex)(C.nppiMalloc_32sc_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32scC4 - 4 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32scC4(nWidthPixels int32, nHeightPixels int32) (ptr *Int32Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Int32Complex)(C.nppiMalloc_32sc_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*

Float32

*/

/*Malloc32fC1 - 1 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fC1(nWidthPixels int32, nHeightPixels int32) (ptr *Float32, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = (*Float32)(C.nppiMalloc_32f_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

/*Malloc32fC2 - 2 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fC2(nWidthPixels int32, nHeightPixels int32) (ptr *Float32, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = (*Float32)(C.nppiMalloc_32f_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

/*Malloc32fC3 - 3 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fC3(nWidthPixels int32, nHeightPixels int32) (ptr *Float32, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = (*Float32)(C.nppiMalloc_32f_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

/*Malloc32fC4 - 4 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fC4(nWidthPixels int32, nHeightPixels int32) (ptr *Float32, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = (*Float32)(C.nppiMalloc_32f_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

/*

Float32Complex

*/

/*Malloc32fcC1 - 1 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fcC1(nWidthPixels int32, nHeightPixels int32) (ptr *Float32Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = (*Float32Complex)(C.nppiMalloc_32fc_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

/*Malloc32fcC2 - 2 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fcC2(nWidthPixels int32, nHeightPixels int32) (ptr *Float32Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = (*Float32Complex)(C.nppiMalloc_32fc_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

/*Malloc32fcC3 - 3 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fcC3(nWidthPixels int32, nHeightPixels int32) (ptr *Float32Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = (*Float32Complex)(C.nppiMalloc_32fc_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

/*Malloc32fcC4 - 4 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fcC4(nWidthPixels int32, nHeightPixels int32) (ptr *Float32Complex, PaddingStepBytes int32) {
	var pStepBytes C.int
	ptr = (*Float32Complex)(C.nppiMalloc_32fc_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(ptr, nppiallocfree)
	return ptr, int32(pStepBytes)
}

func nppiallocfree(x interface{}) error {
	switch y := x.(type) {
	case *Uint8:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Int16:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Int32:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Float32:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Float32Complex:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Int16Complex:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Int32Complex:
		C.nppiFree(unsafe.Pointer(y))
		return nil

	}
	return errors.New("Unsupported type")

}
