package gocudnn

/*

#include <cudnn.h>
#include <stdint.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
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

//TransformTensor transforms a tensor according to how TransformD was set
func (t *TransformD) TransformTensor(h *Handle, alpha float64, srcD *TensorD, src gocu.Mem, beta float64, destD *TensorD, dest gocu.Mem) error {

	a := cscalarbydatatype(srcD.DataType(), alpha)
	b := cscalarbydatatype(destD.DataType(), beta)
	return Status(C.cudnnTransformTensorEx(h.x, t.descriptor, a.CPtr(), srcD.descriptor, src.Ptr(), b.CPtr(), destD.descriptor, dest.Ptr())).error("TransformTensorEx")
}

//Destroy will destroy tensor if not using GC, but if GC is used then it will do nothing
func (t *TransformD) Destroy() error {
	if setfinalizer || t.rtgc {
		return nil
	}
	return cudnnDestroyTensorTransformDescriptor(t)
}

func cudnnDestroyTensorTransformDescriptor(t *TransformD) error {
	err := Status(C.cudnnDestroyTensorTransformDescriptor(t.descriptor)).error("cudnnDestroyTensorTransformDescriptor")
	if err != nil {
		return err
	}
	t = nil
	return nil
}
