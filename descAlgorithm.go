package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"C"
	"encoding/binary"
)
import (
	"errors"
)

//AlgorithmD holds the C.cudnnAlgorithmDescriptor_t
type AlgorithmD struct {
	descriptor C.cudnnAlgorithmDescriptor_t
}

func algorize(input interface{}) (*Algorithm, error) {
	var makebytes [4]byte
	holder := make([]byte, 4)
	switch x := input.(type) {
	case ConvolutionFwdAlgo:
		y := uint32(x)
		binary.LittleEndian.PutUint32(holder, y)
	default:
		return nil, errors.New("Not supported Type")
	}
	for i := 0; i < 4; i++ {
		makebytes[i] = holder[i]
	}
	return &Algorithm{
		algo: makebytes,
	}, nil
}

//Algorithm is used to pass generic stuff
type Algorithm C.cudnnAlgorithm_t

func (a Algorithm) c() C.cudnnAlgorithm_t { return C.cudnnAlgorithm_t(a) }

func CreateAlgorithmDescriptor() (*AlgorithmD, error) {

	var desc C.cudnnAlgorithmDescriptor_t
	err := Status(C.cudnnCreateAlgorithmDescriptor(&desc)).error("CreateAlgorithmDescriptor")
	return &AlgorithmD{
		descriptor: desc,
	}, err

}

func (a *AlgorithmD) cudnnSetAlgorithmDescriptor(algo interface{}) error {
	algorithm, err := algorize(algo)
	if err != nil {
		return err
	}
	err = Status(C.cudnnSetAlgorithmDescriptor(
		a.descriptor,
		C.cudnnAlgorithm_t(*algorithm),
	)).error("CreateAlgorithmDescriptor")
	return err
}
