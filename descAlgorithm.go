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

//SetAlgorithmDescriptor sets the algorithm descriptor
func (a *AlgorithmD) SetAlgorithmDescriptor(algo Algorithm) error {

	err := Status(C.cudnnSetAlgorithmDescriptor(
		a.descriptor,
		algo.c(),
	)).error("SetAlgorithmDescriptor")
	return err
}

// GetAlgorithmDescriptor returns a Algorithm
func (a *AlgorithmD) GetAlgorithmDescriptor() (Algorithm, error) {
	var algo C.cudnnAlgorithm_t
	err := Status(C.cudnnGetAlgorithmDescriptor(
		a.descriptor,
		&algo,
	)).error("GetAlgorithmDescriptor")
	return Algorithm(algo), err
}

//CopyAlgorithmDescriptor returns a copy of AlgorithmD
func (a *AlgorithmD) CopyAlgorithmDescriptor() (*AlgorithmD, error) {
	var desc C.cudnnAlgorithmDescriptor_t
	err := Status(C.cudnnCopyAlgorithmDescriptor(
		a.descriptor,
		desc,
	)).error("CopyAlgorithmDescriptor")
	if err != nil {
		return nil, err
	}
	return &AlgorithmD{
		descriptor: desc,
	}, nil
}

//DestroyDescriptor destroys descriptor
func (a *AlgorithmD) DestroyDescriptor() error {
	return Status(C.cudnnDestroyAlgorithmDescriptor(a.descriptor)).error("DestroyDescriptor")
}

//CreateAlgorithmPerformance creates and returns an AlgorithmPerformance //This might have to return an array be an array
func CreateAlgorithmPerformance(numberToCreate int32) ([]AlgorithmPerformance, error) {
	//var algoperf C.cudnnAlgorithmPerformance_t
	algoperf := make([]C.cudnnAlgorithmPerformance_t, numberToCreate)

	err := Status(C.cudnnCreateAlgorithmPerformance(
		&algoperf[0],
		C.int(numberToCreate),
	)).error("CreateAlgorithmPerformance")
	return calgoperftogoarray(algoperf), err
}

//GetAlgorithmPerformance gets algorithm performance. it returns AlgorithmD, Status, float32(time), SizeT(memorysize in bytes)
//I didn't include the setalgorithmperformance func, but it might need to be made.
func (a *AlgorithmPerformance) GetAlgorithmPerformance() (AlgorithmD, Status, float32, SizeT, error) {
	var algoD AlgorithmD
	var status C.cudnnStatus_t
	var time C.float
	var mem C.size_t

	err := Status(C.cudnnGetAlgorithmPerformance(
		a.descriptor,
		&algoD.descriptor,
		&status,
		&time,
		&mem,
	)).error("GetAlgorithmPerformance")
	return algoD, Status(status), float32(time), SizeT(mem), err
}

//DestroyPerformance destroys the perfmance
func (a *AlgorithmPerformance) DestroyPerformance() error {
	return Status(C.cudnnDestroyAlgorithmPerformance(
		&a.descriptor,
		C.int(0),
	)).error("DestroyPerformance")
}

func calgoperftogoarray(input []C.cudnnAlgorithmPerformance_t) []AlgorithmPerformance {
	size := len(input)
	output := make([]AlgorithmPerformance, size)
	for i := 0; i < size; i++ {
		output[i].descriptor = (input[i])
		output[i].index = C.int(i)
	}
	return output
}
