package xtrakerns

func AllKerns() (all []Kernel) {

	all = append(all, ConcatBackwardNCHWFP16())
	all = append(all, ConcatForwardNCHWFP16())
	all = append(all, ConcatForwardNCHW())
	all = append(all, ConcatBackwardNCHW())
	return all
}
