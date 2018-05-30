package cudnn

import "C"

func intTocint(x []int) []C.int {
	y := make([]C.int, len(x))
	for i := 0; i < len(x); i++ {
		y[i] = C.int(x[i])
	}
	return y
}
func cintToint(x []C.int) []int {
	y := make([]int, len(x))
	for i := 0; i < len(x); i++ {
		y[i] = int(x[i])
	}
	return y
}
