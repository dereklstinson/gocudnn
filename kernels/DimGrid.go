package kernels

//SimpleGridCalculator will output the size of the grid given the size of the block. size is the number of values. blocksizelllll should be multiples of 32!
func SimpleGridCalculator(blocksize uint32, size uint32) uint32 {
	return ((size - 1) / blocksize) + 1
}
