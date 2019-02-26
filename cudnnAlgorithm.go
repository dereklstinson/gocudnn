package gocudnn

/*
#include <cudnn.h>

*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//Algorithm is an empty struct that is used to call algorithm type functions
type Algorithm struct {
	Funcs AlgoFuncs
}

//AlgorithmD holds the C.cudnnAlgorithmDescriptor_t
type AlgorithmD struct {
	descriptor C.cudnnAlgorithmDescriptor_t
}

func (a *AlgorithmD) keepsalive() {
	runtime.KeepAlive(a)
}

//Algos is used to pass generic stuff
type Algos C.cudnnAlgorithm_t

func (a Algos) c() C.cudnnAlgorithm_t { return C.cudnnAlgorithm_t(a) }

//NewAlgorithmDescriptor creates and sets an *AlgorthmD
func (a Algorithm) NewAlgorithmDescriptor(algo Algos) (descriptor *AlgorithmD, err error) {

	var desc C.cudnnAlgorithmDescriptor_t
	err = Status(C.cudnnCreateAlgorithmDescriptor(&desc)).error("CreateAlgorithmDescriptor")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetAlgorithmDescriptor(
		desc,
		algo.c(),
	)).error("SetAlgorithmDescriptor")
	descriptor = &AlgorithmD{
		descriptor: desc,
	}
	if setfinalizer {
		runtime.SetFinalizer(descriptor, destroyalgorithmdescriptor)
	}
	return descriptor, err

}

// GetAlgorithmDescriptor returns a Algorithm
func (a *AlgorithmD) GetAlgorithmDescriptor() (Algos, error) {
	var algo C.cudnnAlgorithm_t
	err := Status(C.cudnnGetAlgorithmDescriptor(
		a.descriptor,
		&algo,
	)).error("GetAlgorithmDescriptor")
	return Algos(algo), err
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
	return destroyalgorithmdescriptor(a)
}
func destroyalgorithmdescriptor(a *AlgorithmD) error {
	return Status(C.cudnnDestroyAlgorithmDescriptor(a.descriptor)).error("DestroyDescriptor")

}

//CreateAlgorithmPerformance creates and returns an AlgorithmPerformance //This might have to return an array be an array
func (a Algorithm) CreateAlgorithmPerformance(numberToCreate int32) ([]AlgorithmPerformance, error) {
	//var algoperf C.cudnnAlgorithmPerformance_t
	algoperf := make([]C.cudnnAlgorithmPerformance_t, numberToCreate)

	err := Status(C.cudnnCreateAlgorithmPerformance(
		&algoperf[0],
		C.int(numberToCreate),
	)).error("CreateAlgorithmPerformance")
	return calgoperftogoarray(algoperf), err
}

//SetAlgorithmPerformance sets the algo performance
func (a *AlgorithmPerformance) SetAlgorithmPerformance(aD *AlgorithmD, s Status, time float32, memory uint) error {
	if setkeepalive {
		keepsalivebuffer(a, aD)
	}
	return Status(C.cudnnSetAlgorithmPerformance(a.descriptor, aD.descriptor, s.c(), C.float(time), C.size_t(memory))).error("SetAlgorithmPerformance")
}

func (a *AlgorithmPerformance) keepsalive() {
	runtime.KeepAlive(a)
}

//GetAlgorithmPerformance gets algorithm performance. it returns AlgorithmD, Status, float32(time), SizeT(memorysize in bytes)
//I didn't include the setalgorithmperformance func, but it might need to be made.
func (a *AlgorithmPerformance) GetAlgorithmPerformance() (AlgorithmD, Status, float32, uint, error) {
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
	return algoD, Status(status), float32(time), uint(mem), err
}

//DestroyPerformance destroys the perfmance
func (a *AlgorithmPerformance) DestroyPerformance() error {
	return destroyalgorithmperformance(a)
}

func destroyalgorithmperformance(a *AlgorithmPerformance) error {
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

//AlgoFuncs is a nil struct used to calling algo functions
type AlgoFuncs struct {
}

//GetAlgorithmSpaceSize gets the size in bytes of the algorithm
func (a *AlgorithmD) GetAlgorithmSpaceSize(handle *Handle) (uint, error) {
	var sizet C.size_t
	err := Status(C.cudnnGetAlgorithmSpaceSize(handle.x, a.descriptor, &sizet)).error("GetAlgorithmSpaceSize")
	if setkeepalive {
		keepsalivebuffer(handle, a)
	}
	return uint(sizet), err
}

//SaveAlgorithm saves the algorithm to host
func (a *AlgorithmD) SaveAlgorithm(handle *Handle, algoSpace gocu.Mem, sizeinbytes uint) error {
	if setkeepalive {
		keepsalivebuffer(handle, a, algoSpace)
	}
	return Status(C.cudnnSaveAlgorithm(
		handle.x,
		a.descriptor,
		algoSpace.Ptr(),
		C.size_t(sizeinbytes),
	)).error("SaveAlgorithm")
}

//RestoreAlgorithm from host
func (a *AlgorithmD) RestoreAlgorithm(handle *Handle, algoSpace gocu.Mem, sizeinbytes uint) error {
	if setkeepalive {
		keepsalivebuffer(handle, a, algoSpace)
	}
	return Status(C.cudnnRestoreAlgorithm(
		handle.x,
		algoSpace.Ptr(),
		C.size_t(sizeinbytes),
		a.descriptor,
	)).error("RestoreAlgorithm")
}
