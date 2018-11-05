package kernels

//XtraKerns returns the kernel names through methods
type XtraKerns struct {
}

//MSELoss Mean Squared Error Loss
func (t XtraKerns) MSELoss() string {
	return "MSELoss"
}

//ShapeToBatch4DNCHW transfers HW to batch and vice versa for NCHW tensors
func (t XtraKerns) ShapeToBatch4DNCHW() string {
	return "ShapetoBatch4DNCHW"
}

//Transpose switches values around from one dimention to the other
func (t XtraKerns) Transpose() string {
	return "Transpose"
}

//ShapeToBatch4DNHWC transfer HW to batch and vice versa through windows for NHWC tensors
func (t XtraKerns) ShapeToBatch4DNHWC() string {
	return "ShapetoBatch4DNHWC"
}

//NearestNeighborNHWC Resize NearestNeightbor for NHWC tensors
func (t XtraKerns) NearestNeighborNHWC() string {
	return "nearestneighborNHWC"
}

//NearestNeighborNCHW Resize NearestNeightbor for NCHW tensors
func (t XtraKerns) NearestNeighborNCHW() string {
	return "nearestneighborNCHW"
}

//NearestNeighborNHWCBack Resize NearestNeightbor for NHWC tensors and accumulates gradients
func (t XtraKerns) NearestNeighborNHWCBack() string {
	return "nearestneighborNHWCBack"
}

//NearestNeighborNCHWBack Resize NearestNeightbor for NCHW tensors and accumulates gradients
func (t XtraKerns) NearestNeighborNCHWBack() string {
	return "nearestneighborNCHWBack"
}

//ForwardParametricfloat Not tested
func (t XtraKerns) ForwardParametricfloat() string {
	return "forwardParametricfloat"
}

//BackwardParametricfloat Not tested
func (t XtraKerns) BackwardParametricfloat() string {
	return "backwardParametricfloat"
}

//ForwardParamFloatChan Not tested
func (t XtraKerns) ForwardParamFloatChan() string {
	return "forwardParametricfloatchannel"
}

//BackwardParamFloatChan Not tested
func (t XtraKerns) BackwardParamFloatChan() string {
	return "backwardParametricfloatchannel"
}

//ForwardLeakyfloat activation function Relu but negatives get a reduced value
func (t XtraKerns) ForwardLeakyfloat() string {
	return "forwardleakyfloat"
}

//BackwardLeakyfloat activation function Relu but negatives get a reduced value
func (t XtraKerns) BackwardLeakyfloat() string {
	return "backwardleakyfloat"
}

//AdaDelta ..
func (t XtraKerns) AdaDelta() string {
	return "adadeltafloat"
}

//AdaGrad ..
func (t XtraKerns) AdaGrad() string {
	return "adagradfloat"
}

//Adam ..
func (t XtraKerns) Adam() string {
	return "adamfloat"
}

/*
//Batch Just reduces the values of the gradients by dividing the batch size
func (t XtraKerns) Batch() string {
	return "batchregfloat"
}
//L1 ..
func (t XtraKerns) L1() string {
	return "l1regularizationfloat"
}

//L2 ..
func (t XtraKerns) L2() string {
	return "l2regularizationfloat"
}
*/
//L1L2 ..
func (t XtraKerns) L1L2() string {
	return "l1l2regularizationfloat"
}

//AlpaBetaCheck checks the alpha beta and counts how many of them are equal
func (t *XtraKerns) AlpaBetaCheck() string {
	return "alphabetacheck"

}
