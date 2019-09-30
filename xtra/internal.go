package xtra

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
func stridecalc(dims []int32) []int32 {
	strides := make([]int32, len(dims))
	stride := int32(1)
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= dims[i]
	}
	return strides
}
