package gocudnn

/*

#include <cudnn.h>
*/
import "C"

/*
 *  pooling mode
 */

//PoolingModeFlag is used to pass PoolingMode flags for human users semi-safely using methods
type PoolingModeFlag struct {
}

//PoolingMode is used for flags in pooling
type PoolingMode C.cudnnPoolingMode_t

//Max returns PoolingMode(C.CUDNN_POOLING_MAX) flag
func (p PoolingModeFlag) Max() PoolingMode {
	return PoolingMode(C.CUDNN_POOLING_MAX)
}

//AverageCountIncludePadding returns PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) flag
func (p PoolingModeFlag) AverageCountIncludePadding() PoolingMode {
	return PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
}

//AverageCountExcludePadding returns PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) flag
func (p PoolingModeFlag) AverageCountExcludePadding() PoolingMode {
	return PoolingMode(C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
}

//MaxDeterministic returns PoolingMode(C.CUDNN_POOLING_MAX_DETERMINISTIC) flag
func (p PoolingModeFlag) MaxDeterministic() PoolingMode {
	return PoolingMode(C.CUDNN_POOLING_MAX_DETERMINISTIC)
}

func (p PoolingMode) c() C.cudnnPoolingMode_t { return C.cudnnPoolingMode_t(p) }
