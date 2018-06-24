package tests

import (
	"testing"

	"github.com/dereklstinson/GoCudnn"
)

func TestActivationHandle(t *testing.T) {

	var NaN gocudnn.PropagationNANFlag
	var Amode gocudnn.ActivationModeFlag
	handle := gocudnn.NewHandle()
	coef := gocudnn.CDouble(10.0)
	var Activation gocudnn.Activation

	aD, err := Activation.NewActivationDescriptor(Amode.Relu(), NaN.NotPropagateNan(), coef)
	if err != nil {
		t.Error(err)
	}
	alpha := gocudnn.CFloat(1.0)
	beta := alpha
	shape := gocudnn.Shape
	xD, xmem, err := testTensorFloat4dNHWC(shape(32, 32, 32, 32))
	if err != nil {
		t.Error(err)
	}
	yD, ymem, err := testTensorFloat4dNHWC(shape(32, 32, 32, 32))
	if err != nil {
		t.Error(err)
	}
	dxD, dxmem, err := testTensorFloat4dNHWC(shape(32, 32, 32, 32))
	if err != nil {
		t.Error(err)
	}
	dyD, dymem, err := testTensorFloat4dNHWC(shape(32, 32, 32, 32))
	if err != nil {
		t.Error(err)
	}

	err = Activation.Funcs.ActivationForward(handle, aD, alpha, xD, xmem, beta, yD, ymem)
	if err != nil {
		t.Error(err)
	}

	err = Activation.Funcs.ActivationBackward(handle, aD, alpha, yD, ymem, dyD, dymem, xD, xmem, beta, dxD, dxmem)
	if err != nil {
		t.Error(err)
	}

	xmem.Free()
	ymem.Free()
	dxmem.Free()
	dymem.Free()
}

/*


//ActivationForward does the forward activation function yrtn is returned and changed.
func (handle *Handle) ActivationForward(
	aD *ActivationD,
	alpha CScalar,
	xD *TensorD,
	x Memer,
	beta CScalar,
	yD *TensorD,
	yrtn Memer) error {
	return Status(C.cudnnActivationForward(handle.x, aD.descriptor, alpha.CPtr(), xD.descriptor, x.Ptr(), beta.CPtr(), yD.descriptor, yrtn.Ptr())).error("ActivationForward")
}

//ActivationBackward does the activation backward method
func (handle *Handle) ActivationBackward(
	aD *ActivationD,
	alpha CScalar,
	yD *TensorD,
	y Memer,
	dyD *TensorD,
	dy Memer,
	xD *TensorD,
	x Memer,
	beta CScalar,
	dxD *TensorD,
	dx Memer) error {
	return Status(C.cudnnActivationBackward(handle.x, aD.descriptor, alpha.CPtr(), yD.descriptor, y.Ptr(), dyD.descriptor, dy.Ptr(), xD.descriptor, x.Ptr(), beta.CPtr(), dxD.descriptor, dx.Ptr())).error("ActivationBackward")
}


*/
