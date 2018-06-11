package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//LRNmode is used for the flags in LRNmode
type LRNmode C.cudnnLRNMode_t

func (l LRNmode) c() C.cudnnLRNMode_t { return C.cudnnLRNMode_t(l) }

//LRNmodeFlag is used to pass LRNmode flags through methods
type LRNmodeFlag struct {
}

//rossChanelDim1 returns LRNmode( C.CUDNN_LRN_CROSS_CHANNEL_DIM1)
func (l LRNmodeFlag) rossChanelDim1() LRNmode {
	return LRNmode(C.CUDNN_LRN_CROSS_CHANNEL_DIM1)
}

//DivNormMode is usde for C.cudnnDivNormMode_t flags
type DivNormMode C.cudnnDivNormMode_t

//DivNormModeFlag is used to pass flags for DivNormMode through methods
type DivNormModeFlag struct {
}

//PrecomputedMeans return DivNormMode(C.CUDNN_DIVNORM_PRECOMPUTED_MEANS)
func (d DivNormModeFlag) PrecomputedMeans() DivNormMode {
	return DivNormMode(C.CUDNN_DIVNORM_PRECOMPUTED_MEANS)
}

func (d DivNormMode) c() C.cudnnDivNormMode_t { return C.cudnnDivNormMode_t(d) }
