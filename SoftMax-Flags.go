package gocudnn

/*

#include <cudnn.h>
*/
import "C"

//SoftMaxAlgorithm is used for flags
type SoftMaxAlgorithm C.cudnnSoftmaxAlgorithm_t

//SoftMaxAlgorithmFlag used to pass SoftMaxAlgorithm flags through methods
type SoftMaxAlgorithmFlag struct {
}

//Fast returns SoftMaxAlgorithm(C.CUDNN_SOFTMAX_FAST)
func (s SoftMaxAlgorithmFlag) Fast() SoftMaxAlgorithm { /* straightforward implementation */
	return SoftMaxAlgorithm(C.CUDNN_SOFTMAX_FAST)
}

//Accurate returns SoftMaxAlgorithm(C.CUDNN_SOFTMAX_ACCURATE)
func (s SoftMaxAlgorithmFlag) Accurate() SoftMaxAlgorithm { /* subtract max from every point to avoid overflow */
	return SoftMaxAlgorithm(C.CUDNN_SOFTMAX_ACCURATE)
}

//Log returns SoftMaxAlgorithm(C.CUDNN_SOFTMAX_LOG)
func (s SoftMaxAlgorithmFlag) Log() SoftMaxAlgorithm {
	return SoftMaxAlgorithm(C.CUDNN_SOFTMAX_LOG)
}

func (sm SoftMaxAlgorithm) c() C.cudnnSoftmaxAlgorithm_t { return C.cudnnSoftmaxAlgorithm_t(sm) }

//SoftMaxMode is used for softmaxmode flags
type SoftMaxMode C.cudnnSoftmaxMode_t

//SoftMaxModeFlag passes SoftMaxMode flags through methods
type SoftMaxModeFlag struct {
}

//Instance returns SoftMaxMode(C.CUDNN_SOFTMAX_MODE_INSTANCE)
func (s SoftMaxModeFlag) Instance() SoftMaxMode { /* subtract max from every point to avoid overflow */
	return SoftMaxMode(C.CUDNN_SOFTMAX_MODE_INSTANCE)
}

//Channel returns SoftMaxMode(C.CUDNN_SOFTMAX_MODE_CHANNEL)
func (s SoftMaxModeFlag) Channel() SoftMaxMode {
	return SoftMaxMode(C.CUDNN_SOFTMAX_MODE_CHANNEL)
}

func (sm SoftMaxMode) c() C.cudnnSoftmaxMode_t { return C.cudnnSoftmaxMode_t(sm) }
