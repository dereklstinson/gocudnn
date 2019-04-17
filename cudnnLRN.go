package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"runtime"
	"strconv"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//LRN is a struct that is used in making lrn layers. It holds the Funcs, and Flags

// LRND holds the LRN Descriptor
type LRND struct {
	descriptor C.cudnnLRNDescriptor_t
	gogc       bool
}

const (
	lrnminN    = uint32(1)
	lrnmaxN    = uint32(16)
	lrnminK    = float64(1e-5)
	lrnminBeta = float64(0.01)
)

//MinN returns the constant lrminN
func (l LRND) MinN() uint32 {
	return lrnminN
}

//MaxN returns the constant lrnmaxN
func (l LRND) MaxN() uint32 {
	return lrnmaxN
}

//MinK returns lrnminK constant
func (l LRND) MinK() float64 {
	return lrnminK
}

//MinBeta returns lrnminBeta constant
func (l LRND) MinBeta() float64 {
	return lrnminBeta
}

//CreateLRNDescriptor creates an RND descriptor
func CreateLRNDescriptor() (*LRND, error) {
	x := new(LRND)
	err := Status(C.cudnnCreateLRNDescriptor(&x.descriptor)).error("NewLRNDecriptor-create")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		x.gogc = true
		runtime.SetFinalizer(x, destroylrndescriptor)
	}
	return x, nil
}

//Set sets the LRND
func (l *LRND) Set(lrnN uint32,
	lrnAlpha,
	lrnBeta,
	lrnK float64) error {
	if lrnN < lrnminN || lrnN > lrnmaxN || lrnK < lrnminK || lrnBeta < 0.01 {
		min := strconv.Itoa(int(lrnminN))
		max := strconv.Itoa(int(lrnmaxN))
		return errors.New("NewLRNDecriptor: lrnN <" + min + "|| lrnN>" + max + "or lrnminK<1e-5|| lrnBeta < 0.01")
	}
	return Status(C.cudnnSetLRNDescriptor(
		l.descriptor,
		C.unsigned(lrnN),
		C.double(lrnAlpha),
		C.double(lrnBeta),
		C.double(lrnK),
	)).error("NewLRNDecriptor-set")
}

//Get returns the descriptor values that were set with set
func (l *LRND) Get() (uint32, float64, float64, float64, error) {
	var N C.unsigned
	var Al, Bet, K C.double

	err := Status(C.cudnnGetLRNDescriptor(
		l.descriptor,
		&N,
		&Al,
		&Bet,
		&K,
	)).error("GetDescriptor-LRN")

	return uint32(N), float64(Al), float64(Bet), float64(K), err
}

//Destroy destroys the descriptor if not using gc it will just return nil if not on.
//Currently gc is always on
func (l *LRND) Destroy() error {
	if l.gogc || setfinalizer {
		return nil
	}
	return destroylrndescriptor(l)
}
func destroylrndescriptor(l *LRND) error {
	return Status(C.cudnnDestroyLRNDescriptor(l.descriptor)).error("DestroyDescriptor")
}

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

//LRNCrossChannelForward  LRN cross-channel forward computation. Double parameters cast to tensor data type
func (l *LRND) LRNCrossChannelForward(
	handle *Handle,
	mode LRNmode,
	alpha float64,
	xD *TensorD, x gocu.Mem,
	beta float64,
	yD *TensorD, y gocu.Mem,
) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnLRNCrossChannelForward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor, x.Ptr(),
		b.CPtr(),
		yD.descriptor, y.Ptr(),
	)).error("LRNCrossChannelForward")
}

//LRNCrossChannelForwardUS is like LRNCrossChannelForward but using unsafe.Pointer instead of gocu.Mem
func (l *LRND) LRNCrossChannelForwardUS(
	handle *Handle,
	mode LRNmode,
	alpha float64,
	xD *TensorD, x unsafe.Pointer,
	beta float64,
	yD *TensorD, y unsafe.Pointer,
) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnLRNCrossChannelForward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor, x,
		b.CPtr(),
		yD.descriptor, y,
	)).error("LRNCrossChannelForward")
}

//LRNCrossChannelBackward  LRN cross-channel backward computation. Double parameters cast to tensor data type
func (l *LRND) LRNCrossChannelBackward(
	handle *Handle,
	mode LRNmode,
	alpha float64,
	yD *TensorD, y gocu.Mem,
	dyD *TensorD, dy gocu.Mem,
	xD *TensorD, x gocu.Mem,
	beta float64,
	dxD *TensorD, dx gocu.Mem,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	return Status(C.cudnnLRNCrossChannelBackward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		yD.descriptor, y.Ptr(),
		dyD.descriptor, dy.Ptr(),
		xD.descriptor, x.Ptr(),
		b.CPtr(),
		dxD.descriptor, dx.Ptr(),
	)).error("LRNCrossChannelForward")
}

//LRNCrossChannelBackwardUS is like LRNCrossChannelBackward but using unsafe.Pointer instead of gocu.Mem
func (l *LRND) LRNCrossChannelBackwardUS(
	handle *Handle,
	mode LRNmode,
	alpha float64,
	yD *TensorD, y unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	xD *TensorD, x unsafe.Pointer,
	beta float64,
	dxD *TensorD, dx unsafe.Pointer,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	return Status(C.cudnnLRNCrossChannelBackward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		yD.descriptor, y,
		dyD.descriptor, dy,
		xD.descriptor, x,
		b.CPtr(),
		dxD.descriptor, dx,
	)).error("LRNCrossChannelForward")
}

//DivisiveNormalizationForward   LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y
func (l *LRND) DivisiveNormalizationForward(
	handle *Handle,
	mode DivNormMode,
	alpha float64,
	xD TensorD, x, means, temp, temp2 gocu.Mem,
	beta float64,
	yD TensorD, y gocu.Mem,
) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnDivisiveNormalizationForward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		means.Ptr(),
		temp.Ptr(),
		temp2.Ptr(),
		b.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("DivisiveNormalizationForward")
}

//DivisiveNormalizationForwardUS is like DivisiveNormalizationForward but using unsafe.Pointer instead of gocu.Mem
func (l *LRND) DivisiveNormalizationForwardUS(
	handle *Handle,
	mode DivNormMode,
	alpha float64,
	xD TensorD, x, means, temp, temp2 unsafe.Pointer,
	beta float64,
	yD TensorD, y unsafe.Pointer,
) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnDivisiveNormalizationForward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor, x, means, temp, temp2,
		b.CPtr(),
		yD.descriptor, y,
	)).error("DivisiveNormalizationForward")
}

//DivisiveNormalizationBackward  LRN cross-channel backward computation. Double parameters cast to tensor data type
func (l *LRND) DivisiveNormalizationBackward(
	handle *Handle,
	mode DivNormMode,
	alpha float64,
	xD *TensorD, x, means, dy, temp, temp2 gocu.Mem,
	beta float64,
	dXdMeansDesc *TensorD, dx, dMeans gocu.Mem,
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(xD.dtype, beta)
	return Status(C.cudnnDivisiveNormalizationBackward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor, x.Ptr(), means.Ptr(), dy.Ptr(), temp.Ptr(), temp2.Ptr(),
		b.CPtr(),
		dXdMeansDesc.descriptor, dx.Ptr(), dMeans.Ptr(),
	)).error("DivisiveNormalizationBackward")
}

//DivisiveNormalizationBackwardUS is like DivisiveNormalizationBackward but using unsafe.Pointer instead of gocu.Mem
func (l *LRND) DivisiveNormalizationBackwardUS(
	handle *Handle,
	mode DivNormMode,
	alpha float64,
	xD *TensorD, x, means, dy, temp, temp2 unsafe.Pointer,
	beta float64,
	dXdMeansDesc *TensorD, dx, dMeans unsafe.Pointer,
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(xD.dtype, beta)
	return Status(C.cudnnDivisiveNormalizationBackward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor, x, means, dy, temp, temp2,
		b.CPtr(),
		dXdMeansDesc.descriptor, dx, dMeans,
	)).error("DivisiveNormalizationBackward")
}

//LRNmode is used for the flags in LRNmode
type LRNmode C.cudnnLRNMode_t

func (l LRNmode) c() C.cudnnLRNMode_t { return C.cudnnLRNMode_t(l) }

//CrossChanelDim1 sets l to and returns LRNmode( C.CUDNN_LRN_CROSS_CHANNEL_DIM1)
func (l *LRNmode) CrossChanelDim1() LRNmode { *l = LRNmode(C.CUDNN_LRN_CROSS_CHANNEL_DIM1); return *l }

//DivNormMode is usde for C.cudnnDivNormMode_t flags
type DivNormMode C.cudnnDivNormMode_t

//PrecomputedMeans sets d to and returns DivNormMode(C.CUDNN_DIVNORM_PRECOMPUTED_MEANS)
func (d *DivNormMode) PrecomputedMeans() DivNormMode {
	*d = DivNormMode(C.CUDNN_DIVNORM_PRECOMPUTED_MEANS)
	return *d
}

func (d DivNormMode) c() C.cudnnDivNormMode_t { return C.cudnnDivNormMode_t(d) }
