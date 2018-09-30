package gocudnn

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
