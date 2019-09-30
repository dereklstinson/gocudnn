package kernels

//XtraKerns returns the kernel names through methods
type XtraKerns struct {
}

//MSELoss Mean Squared Error Loss
func (t XtraKerns) MSELoss() string {
	return "MSELoss"
}
//MSELossFP16 Mean Squared Error Loss
func (t XtraKerns) MSELossFP16() string {
	return "MSELossFP16"
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
	return "NearestNeighborNCHW"
}
//NearestNeighborNCHWFP16 Resize NearestNeightbor for NCHW tensors
func (t XtraKerns) NearestNeighborNCHWFP16() string {
	return "NearestNeighborNCHWFP16"
}
//NearestNeighborNHWCBack Resize NearestNeightbor for NHWC tensors and accumulates gradients
func (t XtraKerns) NearestNeighborNHWCBack() string {
	return "NearestNeighborNHWCBack"
}
//NearestNeighborNHWCBackFP16 Resize NearestNeightbor for NHWC tensors and accumulates gradients
func (t XtraKerns) NearestNeighborNHWCBackFP16() string {
	return "NearestNeighborNHWCBackFP16"
}

//NearestNeighborNCHWBack Resize NearestNeightbor for NCHW tensors and accumulates gradients
func (t XtraKerns) NearestNeighborNCHWBack() string {
	return "NearestNeighborNCHWBack"
}
//NearestNeighborNCHWBackFP16 Resize NearestNeightbor for NCHW tensors and accumulates gradients
func (t XtraKerns) NearestNeighborNCHWBackFP16() string {
	return "NearestNeighborNCHWBackFP16"
}
//ThreshForward Not tested
func (t XtraKerns) ThreshForward() string {
	return "ThreshForward"
}
//ThreshForwardFP16 Not tested
func (t XtraKerns) ThreshForwardFP16() string {
	return "ThreshForwardFP16"
}
//ThreshBackward Not tested
func (t XtraKerns) ThreshBackward() string {
	return "ThreshBackward"
}
//ThreshBackwardFP16 Not tested
func (t XtraKerns) ThreshBackwardFP16() string {
	return "ThreshBackwardFP16"
}
//PreluForward Not tested
func (t XtraKerns) PreluForward() string {
	return "PreluForward"
}
//PreluForwardFP16 Not tested
func (t XtraKerns) PreluForwardFP16() string {
	return "PreluForwardFP16"
}
//PreluBackward Not tested
func (t XtraKerns) PreluBackward() string {
	return "PreluBackward"
}
//PreluBackwardFP16 Not tested
func (t XtraKerns) PreluBackwardFP16() string {
	return "PreluBackwardFP16"
}
//LeakyForward activation function Relu but negatives get a reduced value
func (t XtraKerns) LeakyForward() string {
	return "LeakyForward"
}
//LeakyForwardFP16 activation function Relu but negatives get a reduced value
func (t XtraKerns) LeakyForwardFP16() string {
	return "LeakyForwardFP16"
}
//LeakyBackward activation function Relu but negatives get a reduced value
func (t XtraKerns) LeakyBackward() string {
	return "LeakyBackward"
}
//LeakyBackwardFP16 activation function Relu but negatives get a reduced value
func (t XtraKerns) LeakyBackwardFP16() string {
	return "LeakyBackwardFP16"
}


//LeakyForwardAlpha activation function Relu but negatives get a reduced value result = alpha * activationfunc()
func (t XtraKerns) LeakyForwardAlpha() string {
	return "LeakyForwardAlpha"
}
//LeakyForwardAlphaFP16 activation function Relu but negatives get a reduced value result = alpha * activationfunc()
func (t XtraKerns) LeakyForwardAlphaFP16() string {
	return "LeakyForwardAlphaFP16"
}

//LeakyBackwardAlpha activation function Relu but negatives get a reduced value and function gets the ----- result = alpha * activationfunc()
func (t XtraKerns) LeakyBackwardAlpha() string {
	return "LeakyBackwardAlpha"
}
//LeakyBackwardAlphaFP16 activation function Relu but negatives get a reduced value and function gets the ----- result = alpha * activationfunc()
func (t XtraKerns) LeakyBackwardAlphaFP16() string {
	return "LeakyBackwardAlphaFP16"
}

//LeakyForwardAlphaBeta activation function Relu but negatives get a reduced value and function gets the ----- result = alpha * currentresult + beta * previousresult
func (t XtraKerns) LeakyForwardAlphaBeta() string {
	return "LeakyForwardAlphaBeta"
}
//LeakyForwardAlphaBetaFP16 activation function Relu but negatives get a reduced value and function gets the ----- result = alpha * currentresult + beta * previousresult
func (t XtraKerns) LeakyForwardAlphaBetaFP16() string {
	return "LeakyForwardAlphaBetaFP16"
}

//LeakyBackwardAlphaBeta activation function Relu but negatives get a reduced value and function gets the ----- result = alpha * currentresult + beta * previousresult
func (t XtraKerns) LeakyBackwardAlphaBeta() string {
	return "LeakyBackwardAlphaBeta"
}
//LeakyBackwardAlphaBetaFP16 activation function Relu but negatives get a reduced value and function gets the ----- result = alpha * currentresult + beta * previousresult
func (t XtraKerns) LeakyBackwardAlphaBetaFP16() string {
	return "LeakyBackwardAlphaBetaFP16"
}
//AdaDelta ..
func (t XtraKerns) AdaDelta() string {
	return "AdaDelta"
}
//AdaDeltaFP16 ..
func (t XtraKerns) AdaDeltaFP16() string {
	return "AdaDeltaFP16"
}
//AdaGrad ..
func (t XtraKerns) AdaGrad() string {
	return "AdaGrad"
}
//AdaGradFP16 ..
func (t XtraKerns) AdaGradFP16() string {
	return "AdaGradFP16"
}
//Adam ..
func (t XtraKerns) Adam() string {
	return "Adam"
}
//AdamFP16 ..
func (t XtraKerns) AdamFP16() string {
	return "AdamFP16"
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
	return "L1L2"
}
//L1L2FP16 ..
func (t XtraKerns) L1L2FP16() string {
	return "L1L2FP16"
}