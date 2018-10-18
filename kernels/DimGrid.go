package kernels

//SimpleGridCalculator will output the size of the grid given the size of the block. size is the number of values. blocksizelllll should be multiples of 32!
func SimpleGridCalculator(blocksize uint32, size uint32) uint32 {
	return ((size - 1) / blocksize) + 1
}

//DivUp is same function from tensorflow
func DivUp(a, b int32) int32 {
	return (a + b - 1) / b
}

//DivUpUint32 does it in uint 32
func DivUpUint32(a, b uint32) uint32 {
	return (a + b - 1) / b
}
