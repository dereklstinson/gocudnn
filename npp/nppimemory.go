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

Npp8u


*/

/*Malloc8uC1 - 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc8uC1(nWidthPixels int32, nHeightPixels int32) (ptr *Npp8u, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp8u)(C.nppiMalloc_8u_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc8uC2 - 2 channel 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc8uC2(nWidthPixels int32, nHeightPixels int32) (ptr *Npp8u, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp8u)(C.nppiMalloc_8u_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc8uC3 - 3 channel 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc8uC3(nWidthPixels int32, nHeightPixels int32) (ptr *Npp8u, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp8u)(C.nppiMalloc_8u_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc8uC4 - 4 channel 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc8uC4(nWidthPixels int32, nHeightPixels int32) (ptr *Npp8u, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp8u)(C.nppiMalloc_8u_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*


Npp16u


*/

/*Malloc16uC1 - 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16uC1(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16u, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16u)(C.nppiMalloc_16u_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16uC2 - 2 channel 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16uC2(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16u, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16u)(C.nppiMalloc_16u_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16uC3 - 3 channel 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16uC3(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16u, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16u)(C.nppiMalloc_16u_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16uC4 - 4 channel 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16uC4(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16u, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16u)(C.nppiMalloc_16u_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
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
func Malloc16sC1(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16s, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16s)(C.nppiMalloc_16s_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16sC2 - 2 channel 16-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16sC2(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16s, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16s)(C.nppiMalloc_16s_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16sC4 - 4 channel 16-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16sC4(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16s, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16s)(C.nppiMalloc_16s_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*


Npp16sc


*/

/*Malloc16scC1 - 1 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16scC1(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16sc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16sc)(C.nppiMalloc_16sc_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))

	//y := MakeNpp16scFromUnsafe(unsafe.Pointer(x), false)
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16scC2 - 2 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16scC2(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16sc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16sc)(C.nppiMalloc_16sc_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16scC3 - 3 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16scC3(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16sc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16sc)(C.nppiMalloc_16sc_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc16scC4 - 4 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc16scC4(nWidthPixels int32, nHeightPixels int32) (ptr *Npp16sc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp16sc)(C.nppiMalloc_16sc_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*


Npp32s


*/

/*Malloc32sC1 - 1 channel 32-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32sC1(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32s, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32s)(C.nppiMalloc_32s_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32sC3 - 3 channel 32-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32sC3(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32s, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32s)(C.nppiMalloc_32s_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32sC4 - 4 channel 32-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32sC4(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32s, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32s)(C.nppiMalloc_32s_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*


Npp32sc


*/

/*Malloc32scC1 - 1 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32scC1(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32sc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32sc)(C.nppiMalloc_32sc_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32scC2 - 2 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32scC2(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32sc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32sc)(C.nppiMalloc_32sc_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32scC3 - 3 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32scC3(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32sc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32sc)(C.nppiMalloc_32sc_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32scC4 - 4 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32scC4(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32sc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32sc)(C.nppiMalloc_32sc_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*

Npp32f

*/

/*Malloc32fC1 - 1 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fC1(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32f, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32f)(C.nppiMalloc_32f_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32fC2 - 2 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fC2(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32f, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32f)(C.nppiMalloc_32f_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32fC3 - 3 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fC3(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32f, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32f)(C.nppiMalloc_32f_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32fC4 - 4 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fC4(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32f, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32f)(C.nppiMalloc_32f_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*

Npp32fc

*/

/*Malloc32fcC1 - 1 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fcC1(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32fc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32fc)(C.nppiMalloc_32fc_C1((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32fcC2 - 2 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fcC2(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32fc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32fc)(C.nppiMalloc_32fc_C2((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32fcC3 - 3 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fcC3(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32fc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32fc)(C.nppiMalloc_32fc_C3((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

/*Malloc32fcC4 - 4 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * returns ptr to newly allocated memory and PaddingStepBytes for line step byte padding
 */
func Malloc32fcC4(nWidthPixels int32, nHeightPixels int32) (ptr *Npp32fc, PaddingStepBytes int32) {
	var pStepBytes C.int
	x := (*Npp32fc)(C.nppiMalloc_32fc_C4((C.int)(nWidthPixels), (C.int)(nHeightPixels), &pStepBytes))
	runtime.SetFinalizer(x, nppiallocfree)
	return x, int32(pStepBytes)
}

func nppiallocfree(x interface{}) error {
	switch y := x.(type) {
	case *Npp8u:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Npp16u:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Npp16s:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Npp32s:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Npp32f:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Npp32fc:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Npp16sc:
		C.nppiFree(unsafe.Pointer(y))
		return nil
	case *Npp32sc:
		C.nppiFree(unsafe.Pointer(y))
		return nil

	}
	return errors.New("Unsupported type")

}
