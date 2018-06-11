package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//ReduceTensor is the struct that is used for reduce tensor ops
type ReduceTensor struct {
	tensorDesc        C.cudnnReduceTensorDescriptor_t
	tensorOp          C.cudnnReduceTensorOp_t
	tensorCompType    C.cudnnDataType_t
	tensorNanOpt      C.cudnnNanPropagation_t
	tensorIndices     C.cudnnReduceTensorIndices_t
	tensorIndicesType C.cudnnIndicesType_t
}

//TensorOP returns the tensorop value for the ReduceTensor
func (reduce *ReduceTensor) TensorOP() ReduceTensorOp { return ReduceTensorOp(reduce.tensorOp) }

//CompType returns the Datatype of the reducetensor
func (reduce *ReduceTensor) CompType() DataType { return DataType(reduce.tensorCompType) }

//NanOpt returns the Nan operation flag for the reduce tensor
func (reduce *ReduceTensor) NanOpt() PropagationNAN { return PropagationNAN(reduce.tensorNanOpt) }

//Indices returns the indicies for the Reudce tensor
func (reduce *ReduceTensor) Indices() ReduceTensorIndices {
	return ReduceTensorIndices(reduce.tensorIndices)
}

//IndicType returns the IndicieType flag
func (reduce *ReduceTensor) IndicType() IndiciesType { return IndiciesType(reduce.tensorIndicesType) }

//CreateReduceTensorDescriptor creates the tensor discritper struct
func CreateReduceTensorDescriptor(reduceop ReduceTensorOp, datatype DataType, nanprop PropagationNAN, reducetensorinds ReduceTensorIndices, indicietype IndiciesType) (ReduceTensor, error) {
	var reduce ReduceTensor
	x := Status(C.cudnnCreateReduceTensorDescriptor(&reduce.tensorDesc)).error("CreateReduceTensorDescriptor-create")
	if x != nil {
		return reduce, x
	}
	reduce.tensorOp = C.cudnnReduceTensorOp_t(reduceop)
	reduce.tensorCompType = C.cudnnDataType_t(datatype)
	reduce.tensorNanOpt = C.cudnnNanPropagation_t(nanprop)
	reduce.tensorIndices = C.cudnnReduceTensorIndices_t(reducetensorinds)
	reduce.tensorIndicesType = C.cudnnIndicesType_t(indicietype)
	x = reduce.setReduceTensorDescriptor()
	return reduce, x
}

//SetReduceTensorDescriptor Sets the reduce tensor Descriptor
func (reduce *ReduceTensor) setReduceTensorDescriptor() error {

	x := C.cudnnSetReduceTensorDescriptor(reduce.tensorDesc, reduce.tensorOp, reduce.tensorCompType, reduce.tensorNanOpt, reduce.tensorIndices, reduce.tensorIndicesType)
	return Status(x).error("SetReduceTensorDescriptor")
}

/*
//GetReduceTensorDescriptor Gets a copy of reduce tensor descriptor
func (reduce *ReduceTensor) GetReduceTensorDescriptor() (ReduceTensor, error) {
	var reducex ReduceTensor
	reducex.tensorDesc = reduce.tensorDesc
	x := C.cudnnGetReduceTensorDescriptor(reducex.tensorDesc, &reducex.tensorOp, &reducex.tensorCompType, &reducex.tensorNanOpt, &reducex.tensorIndices, &reducex.tensorIndicesType)
	return reducex, Status(x).error("GetReduceTensorDescriptor")
}
*/

//DestroyReduceTensorDescriptor destroys the reducetensordescriptor
func (reduce *ReduceTensor) DestroyReduceTensorDescriptor() error {
	x := C.cudnnDestroyReduceTensorDescriptor(reduce.tensorDesc)
	err := Status(x).error("DestroyTensorDescriptor")

	return err
}
