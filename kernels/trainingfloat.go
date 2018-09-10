package kernels

//TrainingFloat returns the kernel names through methods
type TrainingFloat struct {
}

//AdaDelta ..
func (t TrainingFloat) AdaDelta() string {
	return "adadeltafloat"
}

//AdaGrad ..
func (t TrainingFloat) AdaGrad() string {
	return "adagradfloat"
}

//Adam ..
func (t TrainingFloat) Adam() string {
	return "adamfloat"
}

//L1 ..
func (t TrainingFloat) L1() string {
	return "l1regularizationfloat"
}

//L2 ..
func (t TrainingFloat) L2() string {
	return "l2regularizationfloat"
}

//L1L2 ..
func (t TrainingFloat) L1L2() string {
	return "l1l2regularizationfloat"
}
