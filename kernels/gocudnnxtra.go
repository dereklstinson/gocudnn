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
//ShapetoBatch4DNCHWFP16 transfers HW to batch and vice versa for NCHW tensors
func (t XtraKerns) ShapetoBatch4DNCHWFP16() string {
	return "ShapetoBatch4DNCHWFP16"
}

//SwapEveryOther allows the user to swap batches between to tensors
//Either the even or the odd tensors.
func (t XtraKerns) SwapEveryOther() string {
	return "SwapEveryOther"
}
//SwapEveryOtherFP16 does SwapEveryOther in fp16
func (t XtraKerns)SwapEveryOtherFP16() string{
	return "SwapEveryOtherFP16"
}
//SwapUpperLower takes to tensors and swaps the upper or lower batches between the two tensors
func (t XtraKerns) SwapUpperLower() string {
	return "SwapUpperLower"
}
//SwapUpperLowerFP16 takes to tensors and swaps the upper or lower batches between the two tensors
func (t XtraKerns) SwapUpperLowerFP16() string {
	return "SwapUpperLowerFP16"
}

//Transpose switches values around from one dimention to the other
func (t XtraKerns) Transpose() string {
	return "Transpose"
}
//TransposeFP16 switches values around from one dimention to the other
func (t XtraKerns)TransposeFP16()string{
	return "TransposeFP16"
}
//ShapeToBatch4DNHWC transfer HW to batch and vice versa through windows for NHWC tensors
func (t XtraKerns) ShapeToBatch4DNHWC() string {
	return "ShapetoBatch4DNHWC"
}
//ShapetoBatch4DNHWCFP16 transfer HW to batch and vice versa through windows for NHWC tensors
func (t XtraKerns) ShapetoBatch4DNHWCFP16() string {
	return "ShapetoBatch4DNHWCFP16"
}

//NearestNeighborNHWC Resize NearestNeightbor for NHWC tensors
func (t XtraKerns) NearestNeighborNHWC() string {
	return "NearestNeighborNHWC"
}
//NearestNeighborNHWCFP16 Resize NearestNeightbor for NHWC tensors
func (t XtraKerns) NearestNeighborNHWCFP16() string {
	return "NearestNeighborNHWCFP16"
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

//ForwardLeakyfloatalpha activation function Relu but negatives get a reduced value result = alpha * activationfunc()
func (t XtraKerns) ForwardLeakyfloatalpha() string {
	return "forwardleakyfloatalpha"
}

//BackwardLeakyfloatalpha activation function Relu but negatives get a reduced value and function gets the ----- result = alpha * activationfunc()
func (t XtraKerns) BackwardLeakyfloatalpha() string {
	return "backwardleakyfloatalpha"
}

//ForwardLeakyfloatalphabeta activation function Relu but negatives get a reduced value and function gets the ----- result = alpha * currentresult + beta * previousresult
func (t XtraKerns) ForwardLeakyfloatalphabeta() string {
	return "forwardleakyfloatalphabeta"
}

//BackwardLeakyfloatalphabeta activation function Relu but negatives get a reduced value and function gets the ----- result = alpha * currentresult + beta * previousresult
func (t XtraKerns) BackwardLeakyfloatalphabeta() string {
	return "backwardleakyfloatalphabeta"
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
