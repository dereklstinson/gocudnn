package gocudnn

type XActivationMode uint
type XActivationModeFlag struct {
}

func (x XActivationModeFlag) Leaky() XActivationMode {
	return XActivationMode(1)
}
func (x XActivationModeFlag) Parametricish() XActivationMode {
	return XActivationMode(2)
}

func (xtra Xtra) NewActivationDescriptor()
