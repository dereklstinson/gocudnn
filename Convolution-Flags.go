package gocudnn

/*
#include <cudnn.h>
*/
import "C"

/*
*
*
*       ConvolutionModeFlag
*
*
 */

//ConvolutionModeFlag is used to pass Convolution Mode Flags in a
//semi-safe way for human users by using methods
type ConvolutionModeFlag struct {
}

//ConvolutionMode is the type to describe the convolution mode flags
type ConvolutionMode C.cudnnConvolutionMode_t

//Convolution returns  ConvolutionMode(C.CUDNN_CONVOLUTION)
func (c ConvolutionModeFlag) Convolution() ConvolutionMode {
	return ConvolutionMode(C.CUDNN_CONVOLUTION)
}

// CrossCorrelation returns ConvolutionMode(C.CUDNN_CROSS_CORRELATION)
func (c ConvolutionModeFlag) CrossCorrelation() ConvolutionMode {
	return ConvolutionMode(C.CUDNN_CROSS_CORRELATION)
}

func (c ConvolutionMode) c() C.cudnnConvolutionMode_t { return C.cudnnConvolutionMode_t(c) }

/*
*
*
*       ConvBwdDataPrefFlag
*
*
 */

//ConvBwdDataPrefFlag used to pass ConvBwdDataPref flags through methods
type ConvBwdDataPrefFlag struct {
}

//ConvBwdDataPref used for flags on bwddatapref
type ConvBwdDataPref C.cudnnConvolutionBwdDataPreference_t

//NoWorkSpace returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE)
func (c ConvBwdDataPrefFlag) NoWorkSpace() ConvBwdDataPref {
	return ConvBwdDataPref(C.CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE)
}

//PreferFastest returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
func (c ConvBwdDataPrefFlag) PreferFastest() ConvBwdDataPref {
	return ConvBwdDataPref(C.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST)
}

//SpecifyWorkSpaceLimit returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
func (c ConvBwdDataPrefFlag) SpecifyWorkSpaceLimit() ConvBwdDataPref {
	return ConvBwdDataPref(C.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT)
}

func (cbd ConvBwdDataPref) c() C.cudnnConvolutionBwdDataPreference_t {
	return C.cudnnConvolutionBwdDataPreference_t(cbd)
}

/*
*
*
*       ConvBwdDataAlgoFlag
*
*
 */

//ConvBwdDataAlgoFlag is used to pass ConvBwdDataAlgo Flags
type ConvBwdDataAlgoFlag struct {
}

//ConvBwdDataAlgo used for flags in the bacward data algorithms
type ConvBwdDataAlgo C.cudnnConvolutionBwdDataAlgo_t

//Algo0 return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0) /* non-deterministic */
func (c ConvBwdDataAlgoFlag) Algo0() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
}

//Algo1 return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)
func (c ConvBwdDataAlgoFlag) Algo1() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)
}

//FFT return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)
func (c ConvBwdDataAlgoFlag) FFT() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)
}

//FFTTiling return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)
func (c ConvBwdDataAlgoFlag) FFTTiling() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)
}

//Winograd 	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
func (c ConvBwdDataAlgoFlag) Winograd() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
}

//WinogradNonFused return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)
func (c ConvBwdDataAlgoFlag) WinogradNonFused() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)
}

//Count return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)
func (c ConvBwdDataAlgoFlag) Count() ConvBwdDataAlgo {
	return ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)
}

func (cbd ConvBwdDataAlgo) c() C.cudnnConvolutionBwdDataAlgo_t {
	return C.cudnnConvolutionBwdDataAlgo_t(cbd)
}

/*
*
*
*       ConvBwdFilterPrefFlag
*
*
 */

//ConvBwdFilterPref are used for flags for the backwds filters
type ConvBwdFilterPref C.cudnnConvolutionBwdFilterPreference_t

//ConvBwdFilterPrefFlag is used to pass ConvBwdFilterPref flags through methods
type ConvBwdFilterPrefFlag struct {
}

//NoWorkspace return ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
func (c ConvBwdFilterPrefFlag) NoWorkspace() ConvBwdFilterPref {
	return ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
}

//PrefFastest return ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)
func (c ConvBwdFilterPrefFlag) PrefFastest() ConvBwdFilterPref {
	return ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)
}

//SpecifyWorkspaceLimit return ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)
func (c ConvBwdFilterPrefFlag) SpecifyWorkspaceLimit() ConvBwdFilterPref {
	return ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)
}

func (bw ConvBwdFilterPref) c() C.cudnnConvolutionBwdFilterPreference_t {
	return C.cudnnConvolutionBwdFilterPreference_t(bw)
}

/*
*
*
*       ConvBwdFiltAlgoFlag
*
*
 */

//ConvBwdFiltAlgo Used for ConvBwdFiltAlgo flags
type ConvBwdFiltAlgo C.cudnnConvolutionBwdFilterAlgo_t

//ConvBwdFiltAlgoFlag is used to pass ConvBwdFiltAlgo Flags
type ConvBwdFiltAlgoFlag struct {
}

//Algo0 return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0) /* non-deterministic */
func (c ConvBwdFiltAlgoFlag) Algo0() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)
}

//Algo1 return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
func (c ConvBwdFiltAlgoFlag) Algo1() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
}

//FFT return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT)
func (c ConvBwdFiltAlgoFlag) FFT() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT)
}

//Algo3 return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)
func (c ConvBwdFiltAlgoFlag) Algo3() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)
}

//Winograd 	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD)
func (c ConvBwdFiltAlgoFlag) Winograd() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD)
}

//WinogradNonFused return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED)
func (c ConvBwdFiltAlgoFlag) WinogradNonFused() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED)
}

//FFTTiling return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING)
func (c ConvBwdFiltAlgoFlag) FFTTiling() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING)
}

//Count return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)
func (c ConvBwdFiltAlgoFlag) Count() ConvBwdFiltAlgo {
	return ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)
}
func (cb ConvBwdFiltAlgo) c() C.cudnnConvolutionBwdFilterAlgo_t {
	return C.cudnnConvolutionBwdFilterAlgo_t(cb)
}

/*
*
*
*       ConvolutionFwdPrefFlag
*
*
 */

// ConvolutionFwdPreference used for flags
type ConvolutionFwdPreference C.cudnnConvolutionFwdPreference_t

/* helper function to provide the convolution algo that fit best the requirement */
//these are flags for ConvolutionFwdPreference

//ConvolutionFwdPrefFlag transfer flags for ConvolutionFwdPreference through methods
type ConvolutionFwdPrefFlag struct {
}

//NoWorkSpace returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE)
func (c ConvolutionFwdPrefFlag) NoWorkSpace() ConvolutionFwdPreference {
	return ConvolutionFwdPreference(C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE)
}

//PreferFastest returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
func (c ConvolutionFwdPrefFlag) PreferFastest() ConvolutionFwdPreference {
	return ConvolutionFwdPreference(C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
}

//SpecifyWorkSpaceLimit returns ConvolutionFwdPreference( C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
func (c ConvolutionFwdPrefFlag) SpecifyWorkSpaceLimit() ConvolutionFwdPreference {
	return ConvolutionFwdPreference(C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
}

/*
*
*
*       ConvFwdAlgoFlag
*
*
 */

//ConvFwdAlgo flags for cudnnConvFwdAlgo_t
type ConvFwdAlgo C.cudnnConvolutionFwdAlgo_t

//ConvFwdAlgoFlag transfer flags for ConvFwdAlgo through methods
type ConvFwdAlgoFlag struct {
}

//ImplicitGemm returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
func (c ConvFwdAlgoFlag) ImplicitGemm() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
}

//ImplicitPrecompGemm returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
func (c ConvFwdAlgoFlag) ImplicitPrecompGemm() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
}

//Gemm returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
func (c ConvFwdAlgoFlag) Gemm() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
}

//Direct returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
func (c ConvFwdAlgoFlag) Direct() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
}

//FFT returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_FFT)
func (c ConvFwdAlgoFlag) FFT() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT)
}

//FFTTiling returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
func (c ConvFwdAlgoFlag) FFTTiling() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
}

//WinoGrad  returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
func (c ConvFwdAlgoFlag) WinoGrad() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
}

//WinoGradNonFused   returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
func (c ConvFwdAlgoFlag) WinoGradNonFused() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
}

//Count    returns ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
func (c ConvFwdAlgoFlag) Count() ConvFwdAlgo {
	return ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
}

func (a ConvFwdAlgo) c() C.cudnnConvolutionFwdAlgo_t {
	return C.cudnnConvolutionFwdAlgo_t(a)
}
