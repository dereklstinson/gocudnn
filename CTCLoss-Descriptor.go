package gocudnn

/*
#include <cudnn.h>



void MakeAlgorithmforCTCL(cudnnAlgorithm_t *input,cudnnCTCLossAlgo_t Algo ){
	input->algo.CTCLossAlgo=Algo;
}
*/
import "C"

//Algo returns al algo
func (c CTCLossAlgo) Algo() Algorithm {
	var algo C.cudnnAlgorithm_t
	C.MakeAlgorithmforCTCL(&algo, c.c())
	return Algorithm(algo)
}

//CTCLossD holdes the C.cudnnCTCLossDescriptor_t
type CTCLossD struct {
	descriptor C.cudnnCTCLossDescriptor_t
}

//NewCTCLossDescriptor Creates and sets a CTCLossD if there is no error
func NewCTCLossDescriptor(data DataType) (*CTCLossD, error) {
	var desc C.cudnnCTCLossDescriptor_t
	err := Status(C.cudnnCreateCTCLossDescriptor(&desc)).error("CreateCTCLossDescriptor-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetCTCLossDescriptor(desc, data.c())).error("CreateCTCLossDescriptor-set")
	if err != nil {
		return nil, err
	}
	return &CTCLossD{
		descriptor: desc,
	}, nil
}

//GetDescriptor returns the datatype and error
func (c *CTCLossD) GetDescriptor() (DataType, error) {
	var data C.cudnnDataType_t
	err := Status(C.cudnnGetCTCLossDescriptor(c.descriptor, &data)).error("GetDescriptor")
	return DataType(data), err

}

//DestroyDescriptor destroys the descriptor inside CTCLossD
func (c *CTCLossD) DestroyDescriptor() error {
	return Status(C.cudnnDestroyCTCLossDescriptor(c.descriptor)).error("DestroyDescriptor")
}
