package gocudnn

/*

#include <cudnn.h>

*/
import "C"

//Need to finish this

//CTCLoss calculates loss
func (handle *Handle) CTCLoss(
	probsD *TensorD, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)  */
	probs Memer, /* probabilities after softmax, in GPU memory */
	labels []int32, /* labels, in CPU memory */
	labelLengths []int32, /* the length of each label, in CPU memory */
	inputLengths []int32, /* the lengths of timing steps in each batch, in CPU memory */
	costs Memer, //output /* the returned costs of CTC, in GPU memory */
	gradientsD *TensorD, /* Tensor descriptor for gradients, the dimensions are T,N,A */
	gradients Memer, //output  /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
	algo CTCLossAlgo, /* algorithm selected, supported now 0 and 1 */
	ctclossD CTCLossD,
	wspace Memer, /* pointer to the workspace, in GPU memory */
) error {
	toclabels := int32Tocint(labels)
	toclablen := int32Tocint(labelLengths)
	tocinlen := int32Tocint(inputLengths)
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
		ctclossD.descriptor,
		wspace.Ptr(),
		wspace.ByteSize().c(),
	)).error("CTCLoss")
	return err
}

//GetCTCLossWorkspaceSize calculates workspace size
func (handle *Handle) GetCTCLossWorkspaceSize(
	probsD *TensorD, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)  */
	gradientsD *TensorD, /* Tensor descriptor for gradients, the dimensions are T,N,A */
	labels []int32, /* labels, in CPU memory */
	labelLengths []int32, /* the length of each label, in CPU memory */
	inputLengths []int32, /* the lengths of timing steps in each batch, in CPU memory */
	algo CTCLossAlgo, /* algorithm selected, supported now 0 and 1 */
	ctclossD CTCLossD,
) (SizeT, error) {
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
		ctclossD.descriptor,
		&bsize,
	)).error("GetCTCLossWorkspaceSize")
	return SizeT(bsize), err
}
