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

//SwapEveryOther allows the user to swap batches between to tensors
//Either the even or the odd tensors.
func (t XtraKerns) SwapEveryOther() string {
	return "SwapEveryOther"
}

//SwapUpperLower takes to tensors and swaps the upper or lower batches between the two tensors
func (t XtraKerns) SwapUpperLower() string {
	return "SwapUpperLower"
}

//SwapEveryOtherInt8 allows the user to swap batches between to tensors
//Either the even or the odd tensors.
func (t XtraKerns) SwapEveryOtherInt8() string {
	return "SwapEveryOther"
}

//SwapUpperLowerInt8 takes to tensors and swaps the upper or lower batches between the two tensors
func (t XtraKerns) SwapUpperLowerInt8() string {
	return "SwapUpperLower"
}

//Int8ToFloat32Normalize takes an array of bytes or int8 and converts it to a float and divides by 255
func (t XtraKerns) Int8ToFloat32Normalize() string {
	return "Int8ToFloat32Normalize"
}

//Int8ToFloat32 takes bytes or chars or int8 and converts each byte to a float
func (t XtraKerns) Int8ToFloat32() string {
	return "Int8ToFloat32"
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

//ThreshForward Not tested
func (t XtraKerns) ThreshForward() string {
	return "ThreshForward"
}

//ThreshBackward Not tested
func (t XtraKerns) ThreshBackward() string {
	return "ThreshBackward"
}

//PreluForward Not tested
func (t XtraKerns) PreluForward() string {
	return "PreluForward"
}

//PreluBackward Not tested
func (t XtraKerns) PreluBackward() string {
	return "PreluBackward"
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
