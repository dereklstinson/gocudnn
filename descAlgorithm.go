package gocudnn

/*
#include <cudnn.h>

*/
import "C"

//AlgorithmD holds the C.cudnnAlgorithmDescriptor_t
type AlgorithmD struct {
	descriptor C.cudnnAlgorithmDescriptor_t
}

//Algorithm is used to pass generic stuff
type Algorithm C.cudnnAlgorithm_t

func (a Algorithm) c() C.cudnnAlgorithm_t { return C.cudnnAlgorithm_t(a) }

//CreateAlgorithmDescriptor returns an *AlgorthmD, error
func CreateAlgorithmDescriptor() (*AlgorithmD, error) {

	var desc C.cudnnAlgorithmDescriptor_t
	err := Status(C.cudnnCreateAlgorithmDescriptor(&desc)).error("CreateAlgorithmDescriptor")
	return &AlgorithmD{
		descriptor: desc,
	}, err

}

func (a *AlgorithmD) cudnnSetAlgorithmDescriptor(algo Algorithm) error {

	err := Status(C.cudnnSetAlgorithmDescriptor(
		a.descriptor,
		algo.c(),
	)).error("CreateAlgorithmDescriptor")
	return err
}
