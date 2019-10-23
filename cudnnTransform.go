package gocudnn

/*

#include <cudnn.h>
#include <stdint.h>
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//TransformD holds the transform tensor descriptor
type TransformD struct {
	descriptor C.cudnnTensorTransformDescriptor_t
	nbdims     C.uint32_t
	rtgc       bool
}

//FoldingDirection is used as a flag for TransformDescriptor which are revealed through
//type's methods.
type FoldingDirection C.cudnnFoldingDirection_t

func (f FoldingDirection) c() C.cudnnFoldingDirection_t {
	return C.cudnnFoldingDirection_t(f)
}
func (f *FoldingDirection) cptr() *C.cudnnFoldingDirection_t {
	return (*C.cudnnFoldingDirection_t)(f)
}

//Fold sets variable to Fold and returns Fold value
func (f *FoldingDirection) Fold() FoldingDirection {
	*f = FoldingDirection(C.CUDNN_TRANSFORM_FOLD)
	return *f
}

//UnFold sets variable to UnFold and returns UnFold value
func (f *FoldingDirection) UnFold() FoldingDirection {
	*f = FoldingDirection(C.CUDNN_TRANSFORM_UNFOLD)
	return *f
}

//CreateTransformDescriptor creates a transform descriptor
//
//Needs to be Set with Set method.
func CreateTransformDescriptor() (*TransformD, error) {
	t := new(TransformD)
	err := Status(C.cudnnCreateTensorTransformDescriptor(&t.descriptor)).error("cudnnCreateTensorTransformDescriptor")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		runtime.SetFinalizer(t, cudnnDestroyTensorTransformDescriptor)
	}
	return t, nil
}

//Set sets the TransformD
//
//padBefore,padAfter,FoldA can be nil if not using any one of those
//Custom flags for gocudnn added custom flags for TensorFormat will cause an error
func (t *TransformD) Set(nbDims uint32, destFormat TensorFormat, padBefore, padAfter []int32, foldA []uint32, direction FoldingDirection) error {
	var pafter, pbefore *C.int32_t
	var fold *C.uint32_t
	t.nbdims = (C.uint32_t)(nbDims)
	if padBefore != nil {
		pbefore = (*C.int32_t)(&padBefore[0])
	}
	if padAfter != nil {
		pafter = (*C.int32_t)(&padAfter[0])
	}
	if foldA != nil {
		fold = (*C.uint32_t)(&foldA[0])
	}

	return Status(C.cudnnSetTensorTransformDescriptor(t.descriptor, t.nbdims, destFormat.c(), pbefore, pafter, fold, direction.c())).error("cudnnSetTensorTransformDescriptor")

}

//Get gets the values of the transform descriptor
func (t *TransformD) Get() (destFormat TensorFormat, padBefore, padAfter []int32, foldA []uint32, direction FoldingDirection, err error) {
	var pafter, pbefore *C.int32_t
	var fold *C.uint32_t
	padBefore = make([]int32, t.nbdims)
	padAfter = make([]int32, t.nbdims)
	foldA = make([]uint32, t.nbdims)
	pbefore = (*C.int32_t)(&padBefore[0])
	pafter = (*C.int32_t)(&padAfter[0])
	fold = (*C.uint32_t)(&foldA[0])
	err = Status(C.cudnnGetTensorTransformDescriptor(t.descriptor, t.nbdims, destFormat.cptr(), pbefore, pafter, fold, direction.cptr())).error("cudnnGetTensorTransformDescriptor")
	return

}

//TransformFilter performs transform on filter
func (t *TransformD) TransformFilter(h *Handle, alpha float64, srcD *FilterD, src cutil.Mem, beta float64, destD *FilterD, dest cutil.Mem) error {
	sdtype, _, _, err := srcD.Get()
	if err != nil {
		return err
	}
	ddtye, _, _, err := destD.Get()
	if err != nil {
		return err
	}
	a := cscalarbydatatype(sdtype, alpha)
	b := cscalarbydatatype(ddtye, beta)
	return Status(C.cudnnTransformFilter(h.x, t.descriptor, a.CPtr(), srcD.descriptor, src.Ptr(), b.CPtr(), destD.descriptor, dest.Ptr())).error("TransformFilter")
}

//InitDest This function initializes and returns a destination tensor descriptor destDesc for tensor transform operations.
//The initialization is done with the desired parameters described in the transform descriptor TensorD.
//Note: The returned tensor descriptor will be packed.
func (t *TransformD) InitDest(src *TensorD) (dest *TensorD, destsib uint, err error) {
	dest = new(TensorD)
	var csib C.size_t
	err = Status(C.cudnnInitTransformDest(t.descriptor, src.descriptor, dest.descriptor, &csib)).error("TransformTensorEx")
	if err != nil {
		return nil, 0, err
	}
	_, _, _, _, err = dest.Get()
	if err != nil {
		return nil, 0, err
	}
	dest.frmt.Strided()
	destsib = (uint)(csib)
	return dest, destsib, nil
}

//TransformTensor transforms a tensor according to how TransformD was set
func (t *TransformD) TransformTensor(h *Handle, alpha float64, srcD *TensorD, src cutil.Mem, beta float64, destD *TensorD, dest cutil.Mem) error {

	a := cscalarbydatatype(srcD.dtype, alpha)
	b := cscalarbydatatype(destD.dtype, beta)
	return Status(C.cudnnTransformTensorEx(h.x, t.descriptor, a.CPtr(), srcD.descriptor, src.Ptr(), b.CPtr(), destD.descriptor, dest.Ptr())).error("TransformTensorEx")
}

//TransformTensorUS is like TransformTensor but uses unsafe.Pointer instead of cutil.Mem
func (t *TransformD) TransformTensorUS(h *Handle, alpha float64, srcD *TensorD, src unsafe.Pointer, beta float64, destD *TensorD, dest unsafe.Pointer) error {
	a := cscalarbydatatype(srcD.dtype, alpha)
	b := cscalarbydatatype(destD.dtype, beta)
	return Status(C.cudnnTransformTensorEx(h.x, t.descriptor, a.CPtr(), srcD.descriptor, src, b.CPtr(), destD.descriptor, dest)).error("TransformTensorEx")
}

//Destroy will destroy tensor if not using GC, but if GC is used then it will do nothing
func (t *TransformD) Destroy() error {
	if setfinalizer || t.rtgc {
		return nil
	}
	return cudnnDestroyTensorTransformDescriptor(t)
}

//GetFoldedConvBackwardDataDescriptors - Hidden Helper function to calculate folding descriptors  for dgrad
func GetFoldedConvBackwardDataDescriptors(h *Handle,
	filter *FilterD,
	diff *TensorD,
	conv *ConvolutionD,
	grad *TensorD,
	transform TensorFormat) (
	foldedfilter *FilterD,
	paddeddiff *TensorD,
	foldedConv *ConvolutionD,
	foldedgrad *TensorD,
	filterfold *TransformD,
	diffpad *TransformD,
	gradfold *TransformD,
	gradunfold *TransformD,
	err error) {
	foldedfilter = new(FilterD)
	paddeddiff = new(TensorD)
	foldedConv = new(ConvolutionD)
	foldedgrad = new(TensorD)
	filterfold = new(TransformD)
	diffpad = new(TransformD)
	gradfold = new(TransformD)
	gradunfold = new(TransformD)
	err = Status(C.cudnnGetFoldedConvBackwardDataDescriptors(h.x,
		filter.descriptor,
		diff.descriptor,
		conv.descriptor,
		grad.descriptor,
		transform.c(),
		foldedfilter.descriptor,
		paddeddiff.descriptor,
		foldedConv.descriptor,
		foldedgrad.descriptor,
		filterfold.descriptor,
		diffpad.descriptor,
		gradfold.descriptor,
		gradunfold.descriptor,
	)).error("GetFoldedConvBackwardDataDescriptors")
	runtime.SetFinalizer(foldedfilter, destroyfilterdescriptor)
	runtime.SetFinalizer(paddeddiff, destroytensordescriptor)
	runtime.SetFinalizer(foldedConv, destroyconvolutiondescriptor)
	runtime.SetFinalizer(foldedgrad, destroytensordescriptor)
	runtime.SetFinalizer(filterfold, cudnnDestroyTensorTransformDescriptor)
	runtime.SetFinalizer(diffpad, cudnnDestroyTensorTransformDescriptor)
	runtime.SetFinalizer(gradfold, cudnnDestroyTensorTransformDescriptor)
	runtime.SetFinalizer(gradunfold, cudnnDestroyTensorTransformDescriptor)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	_, _, _, err = foldedfilter.Get()
	if err != nil {
		return nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	_, _, _, _, err = paddeddiff.Get()
	if err != nil {
		return nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	_, _, _, _, err = foldedgrad.Get()
	if err != nil {
		return nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	filterfold.nbdims = C.uint32_t(foldedgrad.dims)
	diffpad.nbdims = C.uint32_t(foldedgrad.dims)
	gradfold.nbdims = C.uint32_t(foldedgrad.dims)
	gradunfold.nbdims = C.uint32_t(foldedgrad.dims)
	return foldedfilter, paddeddiff, foldedConv, foldedgrad, filterfold, diffpad, gradfold, gradunfold, nil
}
func cudnnDestroyTensorTransformDescriptor(t *TransformD) error {
	err := Status(C.cudnnDestroyTensorTransformDescriptor(t.descriptor)).error("cudnnDestroyTensorTransformDescriptor")
	if err != nil {
		return err
	}
	t = nil
	return nil
}
