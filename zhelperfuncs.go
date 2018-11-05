package gocudnn

/*
#include <cudnn.h>
*/
import "C"

func int32Tocint(x []int32) []C.int {
	y := make([]C.int, len(x))
	for i := 0; i < len(x); i++ {
		y[i] = C.int(x[i])
	}
	return y
}
func cintToint32(x []C.int) []int32 {
	y := make([]int32, len(x))
	for i := 0; i < len(x); i++ {
		y[i] = int32(x[i])
	}
	return y
}
func comparedims(dims ...[]int32) bool {
	totallength := len(dims)
	if totallength == 1 {
		return true
	}
	for i := 1; i < totallength; i++ {
		if len(dims[0]) != len(dims[i]) {
			return false
		}
		for j := 0; j < len(dims[0]); j++ {
			if dims[0][j] != dims[i][j] {
				return false
			}
		}
	}
	return true
}
func findvolume(dims []int32) int32 {
	mult := int32(1)
	for i := range dims {
		mult *= dims[i]
	}
	return mult
}

//FindLength returns the length of of the array considering the number of bytes and the Datatype
func FindLength(s SizeT, dtype DataType) uint32 {
	var dflg DataTypeFlag
	var size uint32
	switch dtype {
	case dflg.Float():
		size = uint32(s / SizeT(4))
	case dflg.Double():
		size = uint32(s / SizeT(8))
	case dflg.Int32():
		size = uint32(s / SizeT(4))
	case dflg.Int8():
		size = uint32(s / SizeT(1))
	case dflg.UInt8():
		size = uint32(s / SizeT(1))
	default:
		size = 0
	}

	return size
}

//FindSizeTfromVol takes a volume of dims and returns the size in bytes in SizeT
func FindSizeTfromVol(volume []int32, dtype DataType) SizeT {
	vol := int32(1)
	for i := int32(0); i < int32(len(volume)); i++ {
		vol *= volume[i]
	}
	switch dtype {
	case DataType(C.CUDNN_DATA_FLOAT):
		return SizeT(vol * int32(4))
	case DataType(C.CUDNN_DATA_DOUBLE):
		return SizeT(vol * int32(8))
	case DataType(C.CUDNN_DATA_INT8):
		return SizeT(vol)
	case DataType(C.CUDNN_DATA_INT32):
		return SizeT(vol * int32(4))
	default:
		return SizeT(0)
	}

}
