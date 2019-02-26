package gocudnn

/*

#include <cudnn.h>

void MakeAlgorithmforCTCL(cudnnAlgorithm_t *input,cudnnCTCLossAlgo_t Algo ){
	input->algo.CTCLossAlgo=Algo;
}
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//CTCLoss is used to call CTC funcs and flags
type CTCLoss struct {
	Funcs CTCLossFuncs
	Flags CTCLossAlgoFlag
}

//Algo returns al algo
func (c CTCLossAlgo) Algo() Algos {
	var algo C.cudnnAlgorithm_t
	C.MakeAlgorithmforCTCL(&algo, c.c())
	return Algos(algo)
}

//CTCLossD holdes the C.cudnnCTCLossDescriptor_t
type CTCLossD struct {
	descriptor C.cudnnCTCLossDescriptor_t
}

func (c *CTCLossD) keepsalive() {
	runtime.KeepAlive(c)
}

//NewCTCLossDescriptor Creates and sets a CTCLossD if there is no error
func (ctc CTCLoss) NewCTCLossDescriptor(data DataType) (descriptor *CTCLossD, err error) {
	var desc C.cudnnCTCLossDescriptor_t
	err = Status(C.cudnnCreateCTCLossDescriptor(&desc)).error("CreateCTCLossDescriptor-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetCTCLossDescriptor(desc, data.c())).error("CreateCTCLossDescriptor-set")
	if err != nil {
		return nil, err
	}
	descriptor = &CTCLossD{
		descriptor: desc,
	}
	if setfinalizer {
		runtime.SetFinalizer(descriptor, destroyctclossdescriptor)
	}
	return descriptor, nil
}

//GetDescriptor returns the datatype and error
func (c *CTCLossD) GetDescriptor() (DataType, error) {
	var data C.cudnnDataType_t
	err := Status(C.cudnnGetCTCLossDescriptor(c.descriptor, &data)).error("GetDescriptor")
	if setkeepalive {
		c.keepsalive()
	}
	return DataType(data), err

}

//DestroyDescriptor destroys the descriptor inside CTCLossD
func (c *CTCLossD) DestroyDescriptor() error {
	return destroyctclossdescriptor(c)
}
func destroyctclossdescriptor(c *CTCLossD) error {
	return Status(C.cudnnDestroyCTCLossDescriptor(c.descriptor)).error("DestroyDescriptor")
}

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

//CTCLossFuncs is a empty struct used to call CTCLoss funcs
type CTCLossFuncs struct {
}

//Need to finish this

//CTCLoss calculates loss
func (c *CTCLossD) CTCLoss(
	handle *Handle,
	probsD *TensorD, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)  */
	probs gocu.Mem, /* probabilities after softmax, in GPU memory */
	labels []int32, /* labels, in CPU memory */
	labelLengths []int32, /* the length of each label, in CPU memory */
	inputLengths []int32, /* the lengths of timing steps in each batch, in CPU memory */
	costs gocu.Mem, //output /* the returned costs of CTC, in GPU memory */
	gradientsD *TensorD, /* Tensor descriptor for gradients, the dimensions are T,N,A */
	gradients gocu.Mem, //output  /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
	algo CTCLossAlgo, /* algorithm selected, supported now 0 and 1 */
	wspace gocu.Mem, /* pointer to the workspace, in GPU memory */
	wspacesize uint,
) error {
	toclabels := int32Tocint(labels)
	toclablen := int32Tocint(labelLengths)
	tocinlen := int32Tocint(inputLengths)
	if wspace == nil {
		return Status(C.cudnnCTCLoss(
			handle.x,
			probsD.descriptor,
			probs.Ptr(),
			&toclabels[0],
			&toclablen[0],
			&tocinlen[0],
			costs.Ptr(),
			gradientsD.descriptor,
			gradients.Ptr(),
			algo.c(),
			c.descriptor,
			nil,
			C.size_t(0),
		)).error("CTCLoss")
	}
	err := Status(C.cudnnCTCLoss(
		handle.x,
		probsD.descriptor,
		probs.Ptr(),
		&toclabels[0],
		&toclablen[0],
		&tocinlen[0],
		costs.Ptr(),
		gradientsD.descriptor,
		gradients.Ptr(),
		algo.c(),
		c.descriptor,
		wspace.Ptr(),
		C.size_t(wspacesize),
	)).error("CTCLoss")

	return err
}

//GetCTCLossWorkspaceSize calculates workspace size
func (c *CTCLossD) GetCTCLossWorkspaceSize(
	handle *Handle,
	probsD *TensorD, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)  */
	gradientsD *TensorD, /* Tensor descriptor for gradients, the dimensions are T,N,A */
	labels []int32, /* labels, in CPU memory */
	labelLengths []int32, /* the length of each label, in CPU memory */
	inputLengths []int32, /* the lengths of timing steps in each batch, in CPU memory */
	algo CTCLossAlgo, /* algorithm selected, supported now 0 and 1 */
) (uint, error) {
	toclabels := int32Tocint(labels)
	toclablen := int32Tocint(labelLengths)
	tocinlen := int32Tocint(inputLengths)
	var bsize C.size_t
	err := Status(C.cudnnGetCTCLossWorkspaceSize(
		handle.x,
		probsD.descriptor,
		gradientsD.descriptor,
		&toclabels[0],
		&toclablen[0],
		&tocinlen[0],
		algo.c(),
		c.descriptor,
		&bsize,
	)).error("GetCTCLossWorkspaceSize")
	if setkeepalive {
		keepsalivebuffer(handle, probsD, gradientsD, c)
	}
	return uint(bsize), err
}
