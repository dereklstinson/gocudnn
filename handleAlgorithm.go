package gocudnn

/*

#include <cudnn.h>

*/
import "C"

//GetAlgorithmSpaceSize gets the size in bytes of the algorithm
func (handle *Handle) GetAlgorithmSpaceSize(algoD *AlgorithmD) (SizeT, error) {
	var sizet C.size_t
	err := Status(C.cudnnGetAlgorithmSpaceSize(handle.x, algoD.descriptor, &sizet)).error("GetAlgorithmSpaceSize")
	return SizeT(sizet), err
}

//SaveAlgorithm saves the algorithm to host
func (handle *Handle) SaveAlgorithm(algoD *AlgorithmD, algoSpace Memer) error {
	return Status(C.cudnnSaveAlgorithm(
		handle.x,
		algoD.descriptor,
		algoSpace.Ptr(),
		algoSpace.ByteSize().c(),
	)).error("SaveAlgorithm")
}

//RestoreAlgorithm from host
func (handle *Handle) RestoreAlgorithm(algoD *AlgorithmD, algoSpace Memer) error {
	return Status(C.cudnnRestoreAlgorithm(
		handle.x,
		algoSpace.Ptr(),
		algoSpace.ByteSize().c(),
		algoD.descriptor,
	)).error("RestoreAlgorithm")
}
