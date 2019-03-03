package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"runtime"
	"strconv"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//LRN is a struct that is used in making lrn layers. It holds the Funcs, and Flags
type LRN struct {
	LRNFlgs LRNmodeFlag
	DivFlgs DivNormModeFlag
}

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

//NewLRNDecriptor creates and sets and returns an LRN descriptor
func (l LRN) NewLRNDecriptor(
	lrnN uint32,
	lrnAlpha,
	lrnBeta,
	lrnK float64,
) (descriptor *LRND, err error) {
	if lrnN < lrnminN || lrnN > lrnmaxN || lrnK < lrnminK || lrnBeta < 0.01 {
		min := strconv.Itoa(int(lrnminN))
		max := strconv.Itoa(int(lrnmaxN))
		return nil, errors.New("NewLRNDecriptor: lrnN <" + min + "|| lrnN>" + max + "or lrnminK<1e-5|| lrnBeta < 0.01")
	}
	var desc C.cudnnLRNDescriptor_t
	err = Status(C.cudnnCreateLRNDescriptor(&desc)).error("NewLRNDecriptor-create")
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
	descriptor = &LRND{descriptor: desc}
	if setfinalizer {
		runtime.SetFinalizer(descriptor, destroylrndescriptor)
	}
	return descriptor, nil
}
func (l *LRND) keepsalive() {
	runtime.KeepAlive(l)
}

//GetDescriptor returns the descriptor values
func (l *LRND) GetDescriptor() (uint32, float64, float64, float64, error) {
	var N C.unsigned
	var Al, Bet, K C.double

	err := Status(C.cudnnGetLRNDescriptor(
		l.descriptor,
		&N,
		&Al,
		&Bet,
		&K,
	)).error("GetDescriptor-LRN")
	if setkeepalive {
		l.keepsalive()
	}
	return uint32(N), float64(Al), float64(Bet), float64(K), err
}

//DestroyDescriptor destroys the descriptor
func (l *LRND) DestroyDescriptor() error {
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
	xD *TensorD,
	x gocu.Mem,
	beta float64,
	yD *TensorD,
	y gocu.Mem,
) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	return Status(C.cudnnLRNCrossChannelForward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		b.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("LRNCrossChannelForward")
}

//LRNCrossChannelBackward  LRN cross-channel backward computation. Double parameters cast to tensor data type
func (l *LRND) LRNCrossChannelBackward(
	handle *Handle,
	mode LRNmode,
	alpha float64,
	yD *TensorD,
	y gocu.Mem,
	dyD *TensorD,
	dy gocu.Mem,
	xD *TensorD,
	x gocu.Mem,
	beta float64,
	dxD *TensorD,
	dx gocu.Mem,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	return Status(C.cudnnLRNCrossChannelBackward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		yD.descriptor,
		y.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		xD.descriptor,
		x.Ptr(),
		b.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
	)).error("LRNCrossChannelForward")
}

//DivisiveNormalizationForward   LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y
func (l *LRND) DivisiveNormalizationForward(
	handle *Handle,
	mode DivNormMode,
	alpha float64,
	xD TensorD, /* same desc for means, temp, temp2 */
	x gocu.Mem,
	means gocu.Mem, /* if NULL, means are assumed to be zero */
	temp gocu.Mem,
	temp2 gocu.Mem,
	beta float64,
	yD TensorD,
	y gocu.Mem,
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

//DivisiveNormalizationBackward  LRN cross-channel backward computation. Double parameters cast to tensor data type
func (l *LRND) DivisiveNormalizationBackward(
	handle *Handle,
	mode DivNormMode,
	alpha float64,
	xD *TensorD, /* same desc for x, means, dy, temp, temp2 */
	x gocu.Mem,
	means gocu.Mem, /* if NULL, means are assumed to be zero */
	dy gocu.Mem,
	temp gocu.Mem,
	temp2 gocu.Mem,
	beta float64,
	dXdMeansDesc *TensorD, /* same desc for dx, dMeans */
	dx gocu.Mem, /* output x differential */
	dMeans gocu.Mem, /* output means differential, can be NULL */
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(xD.dtype, beta)
	return Status(C.cudnnDivisiveNormalizationBackward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		means.Ptr(),
		dy.Ptr(),
		temp.Ptr(),
		temp2.Ptr(),
		b.CPtr(),
		dXdMeansDesc.descriptor,
		dx.Ptr(),
		dMeans.Ptr(),
	)).error("DivisiveNormalizationBackward")
}

//LRNmode is used for the flags in LRNmode
type LRNmode C.cudnnLRNMode_t

func (l LRNmode) c() C.cudnnLRNMode_t { return C.cudnnLRNMode_t(l) }

//LRNmodeFlag is used to pass LRNmode flags through methods
type LRNmodeFlag struct {
}

//rossChanelDim1 returns LRNmode( C.CUDNN_LRN_CROSS_CHANNEL_DIM1)
func (l LRNmodeFlag) rossChanelDim1() LRNmode {
	return LRNmode(C.CUDNN_LRN_CROSS_CHANNEL_DIM1)
}

//DivNormMode is usde for C.cudnnDivNormMode_t flags
type DivNormMode C.cudnnDivNormMode_t

//DivNormModeFlag is used to pass flags for DivNormMode through methods
type DivNormModeFlag struct {
}

//PrecomputedMeans return DivNormMode(C.CUDNN_DIVNORM_PRECOMPUTED_MEANS)
func (d DivNormModeFlag) PrecomputedMeans() DivNormMode {
	return DivNormMode(C.CUDNN_DIVNORM_PRECOMPUTED_MEANS)
}

func (d DivNormMode) c() C.cudnnDivNormMode_t { return C.cudnnDivNormMode_t(d) }
