package tests

import (
	"testing"

	"github.com/dereklstinson/GoCudnn"
)

func TestAlgorithm(t *testing.T) {
	var Algo gocudnn.Algorithm

	AlgoD, err := Algo.CreateAlgorithmDescriptor()
	if err != nil {
		t.Error(err)
	}
	var algo gocudnn.Algos
	err = AlgoD.SetAlgorithmDescriptor(algo)
	if err != nil {
		t.Error(err)
	}
	_, err = AlgoD.GetAlgorithmDescriptor()
	if err != nil {
		t.Error(err)
	}

	algoperf, err := Algo.CreateAlgorithmPerformance(100)
	if err != nil {
		t.Error(err)
	}
	if len(algoperf) != 100 {
		t.Error("should be 100")
	}

	for i := 0; i < len(algoperf); i++ {

		_, _, _, _, err := algoperf[i].GetAlgorithmPerformance()
		if err != nil {
			t.Error(err)
		}
		//	fmt.Println(a, b, c, d, err)
	}

}

/*




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




*/
