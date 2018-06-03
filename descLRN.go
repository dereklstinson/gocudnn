package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"strconv"
)

// LRND holds the LRN Descriptor
type LRND struct {
	descriptor C.cudnnLRNDescriptor_t
}

const (
	lrnminN    = uint32(1)
	lrnmaxN    = uint32(16)
	lrnminK    = float64(1e-5)
	lrnminBeta = float64(0.01)
)

//NewLRNDecriptor creates and sets and returns an LRN descriptor
func NewLRNDecriptor(
	lrnN uint32,
	lrnAlpha,
	lrnBeta,
	lrnK float64,
) (*LRND, error) {
	if lrnN < lrnminN || lrnN > lrnmaxN || lrnK < lrnminK || lrnBeta < 0.01 {
		min := strconv.Itoa(int(lrnminN))
		max := strconv.Itoa(int(lrnmaxN))
		return nil, errors.New("NewLRNDecriptor: lrnN <" + min + "|| lrnN>" + max + "or lrnminK<1e-5|| lrnBeta < 0.01")
	}
	var desc C.cudnnLRNDescriptor_t
	err := Status(C.cudnnCreateLRNDescriptor(&desc)).error("NewLRNDecriptor-create")
	if err != nil {
		return nil, err
	}
	err = Status(C.cudnnSetLRNDescriptor(
		desc,
		C.unsigned(lrnN),
		C.double(lrnAlpha),
		C.double(lrnBeta),
		C.double(lrnK),
	)).error("NewLRNDecriptor-set")
	if err != nil {
		return nil, err
	}
	return &LRND{descriptor: desc}, nil
}

//GetDescriptor returns the descriptor values
func (lrn *LRND) GetDescriptor() (uint32, float64, float64, float64, error) {
	var N C.unsigned
	var Al, Bet, K C.double

	err := Status(C.cudnnGetLRNDescriptor(
		lrn.descriptor,
		&N,
		&Al,
		&Bet,
		&K,
	)).error("GetDescriptor-LRN")
	return uint32(N), float64(Al), float64(Bet), float64(K), err
}

//DestroyDescriptor destroys the descriptor
func (lrn *LRND) DestroyDescriptor() error {
	return Status(C.cudnnDestroyLRNDescriptor(lrn.descriptor)).error("DestroyDescriptor")
}
