package gocudnn

/*
#include <cudnn.h>
*/
import "C"

/*GetReductionIndicesSize Helper function to return the minimum size in bytes of the index space to be passed to the reduction given the input and output tensors */
func (handle *Handle) GetReductionIndicesSize(
	reducer *ReduceTensor,
	aDesc, cDesc *TensorD) (SizeT, error) {
	var sizeinbytes C.size_t
	x := C.cudnnGetReductionIndicesSize(handle.x, reducer.tensorDesc, aDesc.descriptor, cDesc.descriptor, &sizeinbytes)
	return SizeT(sizeinbytes), Status(x).error("GetReductionIndicesSize")

}

//GetReductionWorkspaceSize  Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors
func (handle *Handle) GetReductionWorkspaceSize(
	reducer *ReduceTensor,
	aDesc, cDesc *TensorD) (SizeT, error) {
	var sizeinbytes C.size_t
	x := C.cudnnGetReductionWorkspaceSize(handle.x, reducer.tensorDesc, aDesc.descriptor, cDesc.descriptor, &sizeinbytes)
	return SizeT(sizeinbytes), Status(x).error("GetReductionWorkspaceSize")

}

//ReduceTensorOp Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
func (handle *Handle) ReduceTensorOp(
	data DataType,
	reducer *ReduceTensor,
	indices,
	workspace Memer,
	alpha CScalar,
	aDesc *TensorD,
	A Memer,
	beta CScalar,
	cDesc *TensorD,
	Ce Memer) error {

	x := C.cudnnReduceTensor(handle.x, reducer.tensorDesc, indices.Ptr(),
		C.size_t(indices.ByteSize()), workspace.Ptr(), C.size_t(workspace.ByteSize()),
		alpha.CPtr(), aDesc.descriptor, A.Ptr(), beta.CPtr(), cDesc.descriptor, Ce.Ptr())
	return Status(x).error("ReduceTensor")
}

//SetTensor -  Set all values of a tensor to a given value : y[i] = value[0]
func (handle *Handle) SetTensor(data DataType, yDesc *TensorD, y Memer, v CScalar) error {

	x := C.cudnnSetTensor(handle.x, yDesc.descriptor, y.Ptr(), v.CPtr())
	return Status(x).error("SetTensor")
}

//ScaleTensor - Scale all values of a tensor by a given factor : y[i] = alpha * y[i]
func (handle *Handle) ScaleTensor(data DataType, yDesc *TensorD, y Memer, alpha CScalar) error {

	x := C.cudnnScaleTensor(handle.x, yDesc.descriptor, y.Ptr(), alpha.CPtr())
	return Status(x).error("ScaleTensor")
}
