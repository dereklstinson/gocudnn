package gocudnn

/*

#include <cudnn.h>
*/
import "C"
import "fmt"

//ConvolutionBackwardBias Function to compute the bias gradient for batch convolution db is returned
func (handle *Handle) ConvolutionBackwardBias(alpha CScaler, dyD TensorD, dy Memer, beta CScaler, dbD TensorD, db Memer) error {
	return Status(C.cudnnConvolutionBackwardBias(handle.x, alpha.CPtr(), dyD.descriptor, dy.Ptr(), beta.CPtr(), dbD.descriptor, db.Ptr())).error("ConvolutionBackwardBias")
}

//ConvBwdFilterPref are used for flags for the backwds filters
type ConvBwdFilterPref C.cudnnConvolutionBwdFilterPreference_t

//These are the backwards filter flags
const (
	ConvBwdFilterNoWorkspace           ConvBwdFilterPref = C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE
	ConvBwdFilterPrefFastest           ConvBwdFilterPref = C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST
	ConvBwdFilterSpecifyWorkspaceLimit ConvBwdFilterPref = C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
)

func (bw ConvBwdFilterPref) c() C.cudnnConvolutionBwdFilterPreference_t {
	return C.cudnnConvolutionBwdFilterPreference_t(bw)
}

//ConvBwdFiltAlgo Used for ConvBwdFiltAlgo flags
type ConvBwdFiltAlgo C.cudnnConvolutionBwdFilterAlgo_t

// ConvBwdFiltAlgo flags
const (
	ConvBwdFiltAlgo0         ConvBwdFiltAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 /* non-deterministic */
	ConvBwdFiltAlgo1         ConvBwdFiltAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
	ConvBwdFiltAlgoFFT       ConvBwdFiltAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT
	ConvBwdFiltAlgo3         ConvBwdFiltAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3        /* non-deterministic */
	ConvBwdFiltAlgoWinGrad   ConvBwdFiltAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD /* not implemented */
	ConvBwdFiltAlgoNonFused  ConvBwdFiltAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED
	ConvBwdFiltAlgoFFTTiling ConvBwdFiltAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING
	ConvBwdFiltAlgoCount     ConvBwdFiltAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT
)

func (cb ConvBwdFiltAlgo) c() C.cudnnConvolutionBwdFilterAlgo_t {
	return C.cudnnConvolutionBwdFilterAlgo_t(cb)
}
func (cb ConvBwdFiltAlgo) print() {
	switch cb {
	case ConvBwdFiltAlgo0:
		fmt.Println("ConvBwdFiltAlgo0")
	case ConvBwdFiltAlgo1:
		fmt.Println("ConvBwdFiltAlgo1")
	case ConvBwdFiltAlgoFFT:
		fmt.Println("ConvBwdFiltAlgoFFT")
	case ConvBwdFiltAlgo3:
		fmt.Println("ConvBwdFiltAlgo3")
	case ConvBwdFiltAlgoWinGrad:
		fmt.Println("ConvBwdFiltAlgoWinGrad")
	case ConvBwdFiltAlgoNonFused:
		fmt.Println("ConvBwdFiltAlgoNonFused")
	case ConvBwdFiltAlgoFFTTiling:
		fmt.Println("ConvBwdFiltAlgoFFTTiling")
	case ConvBwdFiltAlgoCount:
		fmt.Println("ConvBwdFiltAlgoCount")
	default:
		fmt.Println("Not supported")
	}
}

//ConvBwdDataAlgoPerf is the return struct in the finding algorithm funcs
type ConvBwdDataAlgoPerf C.cudnnConvolutionBwdDataAlgoPerf_t
