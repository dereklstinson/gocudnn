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

//Algo returns al algo
func (c CTCLossAlgo) Algo() Algorithm {
	var algo C.cudnnAlgorithm_t
	C.MakeAlgorithmforCTCL(&algo, c.c())
	return Algorithm(algo)
}

//CTCLossD holdes the C.cudnnCTCLossDescriptor_t
type CTCLossD struct {
	descriptor C.cudnnCTCLossDescriptor_t
	gogc       bool
}

//CreateCTCLossDescriptor creates
func CreateCTCLossDescriptor() (*CTCLossD, error) {
	x := new(CTCLossD)
	err := Status(C.cudnnCreateCTCLossDescriptor(&x.descriptor)).error("CreateCTCLossDescriptor-create")
	if err != nil {
		return nil, err
	}
	x.gogc = true
	if setfinalizer || x.gogc {
		runtime.SetFinalizer(x, cudnnDestroyCTCLossDescriptor)
	}
	return x, err
}

//Set sets the CTCLossD
func (c *CTCLossD) Set(data DataType) error {
	return Status(C.cudnnSetCTCLossDescriptor(c.descriptor, data.c())).error("CreateCTCLossDescriptor-set")
}

//Get returns the datatype and error
func (c *CTCLossD) Get() (DataType, error) {
	var data C.cudnnDataType_t
	err := Status(C.cudnnGetCTCLossDescriptor(c.descriptor, &data)).error("GetDescriptor")

	return DataType(data), err

}

//Destroy destroys the descriptor inside CTCLossD if go's gc is not in use.
//if gc is being used destroy will just return nil
func (c *CTCLossD) Destroy() error {
	if setfinalizer || c.gogc {
		return nil
	}
	return cudnnDestroyCTCLossDescriptor(c)
}
func cudnnDestroyCTCLossDescriptor(c *CTCLossD) error {
	return Status(C.cudnnDestroyCTCLossDescriptor(c.descriptor)).error("DestroyDescriptor")
}

//CTCLossAlgo used to hold flags
type CTCLossAlgo C.cudnnCTCLossAlgo_t

//Deterministic sets c to and returns CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_DETERMINISTIC)
func (c *CTCLossAlgo) Deterministic() CTCLossAlgo {
	*c = CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_DETERMINISTIC)
	return *c
}

//NonDeterministic sets c to and returns CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC) Flag
func (c *CTCLossAlgo) NonDeterministic() CTCLossAlgo {
	*c = CTCLossAlgo(C.CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC)
	return *c
}

func (c CTCLossAlgo) c() C.cudnnCTCLossAlgo_t {
	return C.cudnnCTCLossAlgo_t(c)
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
