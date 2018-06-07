package gocudnn

/*
#include <cudnn.h>



void MakeAlgorithmforCTCL(cudnnAlgorithm_t *input,cudnnCTCLossAlgo_t Algo ){
	input->algo.CTCLossAlgo=Algo;
}
*/
import "C"

//CTCLossAlgo used to hold flags
type CTCLossAlgo C.cudnnCTCLossAlgo_t

//CTCLossAlgoFlag used to give a semi safe way of exporting CTCLossAlgo flags through methods
type CTCLossAlgoFlag struct {
}

//Deterministic returns CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_DETERMINISTIC)
func (c CTCLossAlgoFlag) Deterministic() CTCLossAlgo {
	return CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_DETERMINISTIC)
}

//NonDeterministic returns   CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC) Flag
func (c CTCLossAlgoFlag) NonDeterministic() CTCLossAlgo {
	return CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC)
}

func (c CTCLossAlgo) c() C.cudnnCTCLossAlgo_t {
	return C.cudnnCTCLossAlgo_t(c)
}

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
