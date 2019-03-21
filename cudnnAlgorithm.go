package gocudnn

/*
#include <cudnn.h>

*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//AlgorithmD holds the C.cudnnAlgorithmDescriptor_t
type AlgorithmD struct {
	descriptor C.cudnnAlgorithmDescriptor_t
	gogc       bool
}

//Algorithm is used to pass generic stuff
type Algorithm C.cudnnAlgorithm_t

func (a Algorithm) c() C.cudnnAlgorithm_t      { return C.cudnnAlgorithm_t(a) }
func (a *Algorithm) cptr() *C.cudnnAlgorithm_t { return (*C.cudnnAlgorithm_t)(a) }

//CreateAlgorithmDescriptor creates an AlgorithmD that needs to be set
func CreateAlgorithmDescriptor() (*AlgorithmD, error) {

	x := new(AlgorithmD)
	x.gogc = setfinalizer
	err := Status(C.cudnnCreateAlgorithmDescriptor(&x.descriptor)).error("CreateAlgorithmDescriptor")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		runtime.SetFinalizer(x, destroyalgorithmdescriptor)
	}
	return x, nil

}
//Set sets the algorthm into the algorithmd
func (a *AlgorithmD) Set(algo Algorithm) error {
	return Status(C.cudnnSetAlgorithmDescriptor(
		a.descriptor,
		algo.c(),
	)).error("SetAlgorithmDescriptor")
}

// Get returns AlgrothmD values a Algorithm.
func (a *AlgorithmD) Get() (Algorithm, error) {
	var algo C.cudnnAlgorithm_t
	err := Status(C.cudnnGetAlgorithmDescriptor(
		a.descriptor,
		&algo,
	)).error("GetAlgorithmDescriptor")
	return Algorithm(algo), err
}

//Copy returns a copy of AlgorithmD
func (a *AlgorithmD) Copy() (*AlgorithmD, error) {
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

//Destroy destroys descriptor. Right now since gocudnn is on go's gc this won't do anything
func (a *AlgorithmD) Destroy() error {
	if a.gogc||setfinalizer {
		return nil
	}
	return destroyalgorithmdescriptor(a)
}
func destroyalgorithmdescriptor(a *AlgorithmD) error {
	err := Status(C.cudnnDestroyAlgorithmDescriptor(a.descriptor)).error("DestroyDescriptor")
	if err != nil {
		return err
	}
	a = nil
	return nil

}

//CreateAlgorithmPerformance creates and returns an AlgorithmPerformance //This might have to return an array be an array
func CreateAlgorithmPerformance(numberToCreate int32) ([]AlgorithmPerformance, error) {
	//var algoperf C.cudnnAlgorithmPerformance_t
	algoperf := make([]C.cudnnAlgorithmPerformance_t, numberToCreate)

	err := Status(C.cudnnCreateAlgorithmPerformance(
		&algoperf[0],
		C.int(numberToCreate),
	)).error("CreateAlgorithmPerformance")
	return calgoperftogoarray(algoperf, setfinalizer), err
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

//Destroy destroys the perfmance
func (a *AlgorithmPerformance) Destroy() error {
	if a.gogc || setfinalizer {
		return nil
	}
	return destroyalgorithmperformance(a)
}

func destroyalgorithmperformance(a *AlgorithmPerformance) error {
	return Status(C.cudnnDestroyAlgorithmPerformance(
		&a.descriptor,
		C.int(0),
	)).error("DestroyPerformance")
}

func calgoperftogoarray(input []C.cudnnAlgorithmPerformance_t, gogc bool) []AlgorithmPerformance {
	size := len(input)
	output := make([]AlgorithmPerformance, size)
	for i := 0; i < size; i++ {
		output[i].gogc = gogc
		output[i].descriptor = (input[i])
		output[i].index = C.int(i)
	}
	return output
}

//GetAlgorithmSpaceSize gets the size in bytes of the algorithm
func (a *AlgorithmD) GetAlgorithmSpaceSize(handle *Handle) (uint, error) {
	var sizet C.size_t
	err := Status(C.cudnnGetAlgorithmSpaceSize(handle.x, a.descriptor, &sizet)).error("GetAlgorithmSpaceSize")

	return uint(sizet), err
}

//SaveAlgorithm saves the algorithm to host
func (a *AlgorithmD) SaveAlgorithm(handle *Handle, algoSpace gocu.Mem, sizeinbytes uint) error {

	return Status(C.cudnnSaveAlgorithm(
		handle.x,
		a.descriptor,
		algoSpace.Ptr(),
		C.size_t(sizeinbytes),
	)).error("SaveAlgorithm")
}

//RestoreAlgorithm from host
func (a *AlgorithmD) RestoreAlgorithm(handle *Handle, algoSpace gocu.Mem, sizeinbytes uint) error {

	return Status(C.cudnnRestoreAlgorithm(
		handle.x,
		algoSpace.Ptr(),
		C.size_t(sizeinbytes),
		a.descriptor,
	)).error("RestoreAlgorithm")
}
