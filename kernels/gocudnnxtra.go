package kernels

//XtraKerns returns the kernel names through methods
type XtraKerns struct {
}

func (t XtraKerns) ForwardParametricfloat() string {
	return "forwardParametricfloat"
}
func (t XtraKerns) BackwardParametricfloat() string {
	return "backwardParametricfloat"
}
func (t XtraKerns) ForwardLeakyfloat() string {
	return "forwardleakyfloat"
}
func (t XtraKerns) BackwardLeakyfloat() string {
	return "backwardleakyfloat"
}
func (t XtraKerns) Batch() string {
	return "batchregfloat"
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

//L1 ..
func (t XtraKerns) L1() string {
	return "l1regularizationfloat"
}

//L2 ..
func (t XtraKerns) L2() string {
	return "l2regularizationfloat"
}

//L1L2 ..
func (t XtraKerns) L1L2() string {
	return "l1l2regularizationfloat"
}
