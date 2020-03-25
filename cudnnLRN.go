package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"unsafe"

	"github.com/dereklstinson/cutil"
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
	err := Status(C.cudnnCreateLRNDescriptor(&x.descriptor)).error("CreateLRNDescriptor()")
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
	)).error("(l *LRND) Set")
}

//Get returns the descriptor values that were set with set
func (l *LRND) Get() (lrnN uint32, lrnAlpha float64, lrnBeta float64, lrnK float64, err error) {
	var N C.unsigned
	var Al, Bet, K C.double

	err = Status(C.cudnnGetLRNDescriptor(
		l.descriptor,
		&N,
		&Al,
		&Bet,
		&K,
	)).error("(l *LRND) Get()")
	lrnN, lrnAlpha, lrnBeta, lrnK = uint32(N), float64(Al), float64(Bet), float64(K)
	return lrnN, lrnAlpha, lrnBeta, lrnK, err
}
func (l *LRND) String() string {
	lrnN, lrnAlpha, lrnBeta, lrnK, err := l.Get()
	if err != nil {
		return fmt.Sprintf("LRND{\nError\n}\n")
	}
	return fmt.Sprintf("LRND{\nN: %v,\nAlpha: %v,\nBeta: %v\n,K: %v\n}\n", lrnN, lrnAlpha, lrnBeta, lrnK)
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
	return Status(C.cudnnDestroyLRNDescriptor(l.descriptor)).error("destroylrndescriptor(l *LRND)")
}

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

//LRNCrossChannelForward  LRN cross-channel forward computation. Double parameters cast to tensor data type
func (l *LRND) LRNCrossChannelForward(
	handle *Handle,
	mode LRNmode,
	alpha float64,
	xD *TensorD, x cutil.Mem,
	beta float64,
	yD *TensorD, y cutil.Mem,
) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnLRNCrossChannelForward(
				handle.x,
				l.descriptor,
				mode.c(),
				a.CPtr(),
				xD.descriptor, x.Ptr(),
				b.CPtr(),
				yD.descriptor, y.Ptr(),
			)).error("(l *LRND) LRNCrossChannelForward")

		})
	}
	return Status(C.cudnnLRNCrossChannelForward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor, x.Ptr(),
		b.CPtr(),
		yD.descriptor, y.Ptr(),
	)).error("(l *LRND) LRNCrossChannelForward")
}

//LRNCrossChannelForwardUS is like LRNCrossChannelForward but using unsafe.Pointer instead of cutil.Mem
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
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnLRNCrossChannelForward(
				handle.x,
				l.descriptor,
				mode.c(),
				a.CPtr(),
				xD.descriptor, x,
				b.CPtr(),
				yD.descriptor, y,
			)).error("(l *LRND) LRNCrossChannelForwardUS")
		})
	}
	return Status(C.cudnnLRNCrossChannelForward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor, x,
		b.CPtr(),
		yD.descriptor, y,
	)).error("(l *LRND) LRNCrossChannelForwardUS")
}

//LRNCrossChannelBackward  LRN cross-channel backward computation. Double parameters cast to tensor data type
func (l *LRND) LRNCrossChannelBackward(
	handle *Handle,
	mode LRNmode,
	alpha float64,
	yD *TensorD, y cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	xD *TensorD, x cutil.Mem,
	beta float64,
	dxD *TensorD, dx cutil.Mem,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
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
			)).error(" (l *LRND) LRNCrossChannelBackward")
		})
	}
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
	)).error(" (l *LRND) LRNCrossChannelBackward")
}

//LRNCrossChannelBackwardUS is like LRNCrossChannelBackward but using unsafe.Pointer instead of cutil.Mem
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
	if handle.w != nil {
		return handle.w.Work(func() error {
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
			)).error("(l *LRND) LRNCrossChannelBackwardUS")
		})
	}
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
	)).error("(l *LRND) LRNCrossChannelBackwardUS")
}

//DivisiveNormalizationForward   LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y
func (l *LRND) DivisiveNormalizationForward(
	handle *Handle,
	mode DivNormMode,
	alpha float64,
	xD TensorD, x, means, temp, temp2 cutil.Mem,
	beta float64,
	yD TensorD, y cutil.Mem,
) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
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
			)).error("(l *LRND) DivisiveNormalizationForward")
		})
	}
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
	)).error("(l *LRND) DivisiveNormalizationForward")
}

//DivisiveNormalizationForwardUS is like DivisiveNormalizationForward but using unsafe.Pointer instead of cutil.Mem
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
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnDivisiveNormalizationForward(
				handle.x,
				l.descriptor,
				mode.c(),
				a.CPtr(),
				xD.descriptor, x, means, temp, temp2,
				b.CPtr(),
				yD.descriptor, y,
			)).error(" (l *LRND) DivisiveNormalizationForwardUS")
		})
	}
	return Status(C.cudnnDivisiveNormalizationForward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor, x, means, temp, temp2,
		b.CPtr(),
		yD.descriptor, y,
	)).error(" (l *LRND) DivisiveNormalizationForwardUS")
}

//DivisiveNormalizationBackward  LRN cross-channel backward computation. Double parameters cast to tensor data type
func (l *LRND) DivisiveNormalizationBackward(
	handle *Handle,
	mode DivNormMode,
	alpha float64,
	xD *TensorD, x, means, dy, temp, temp2 cutil.Mem,
	beta float64,
	dXdMeansDesc *TensorD, dx, dMeans cutil.Mem,
) error {
	a := cscalarbydatatype(xD.dtype, alpha)
	b := cscalarbydatatype(xD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnDivisiveNormalizationBackward(
				handle.x,
				l.descriptor,
				mode.c(),
				a.CPtr(),
				xD.descriptor, x.Ptr(), means.Ptr(), dy.Ptr(), temp.Ptr(), temp2.Ptr(),
				b.CPtr(),
				dXdMeansDesc.descriptor, dx.Ptr(), dMeans.Ptr(),
			)).error("(l *LRND) DivisiveNormalizationBackward")
		})
	}
	return Status(C.cudnnDivisiveNormalizationBackward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor, x.Ptr(), means.Ptr(), dy.Ptr(), temp.Ptr(), temp2.Ptr(),
		b.CPtr(),
		dXdMeansDesc.descriptor, dx.Ptr(), dMeans.Ptr(),
	)).error("(l *LRND) DivisiveNormalizationBackward")
}

//DivisiveNormalizationBackwardUS is like DivisiveNormalizationBackward but using unsafe.Pointer instead of cutil.Mem
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
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnDivisiveNormalizationBackward(
				handle.x,
				l.descriptor,
				mode.c(),
				a.CPtr(),
				xD.descriptor, x, means, dy, temp, temp2,
				b.CPtr(),
				dXdMeansDesc.descriptor, dx, dMeans,
			)).error("(l *LRND) DivisiveNormalizationBackwardUS")
		})
	}
	return Status(C.cudnnDivisiveNormalizationBackward(
		handle.x,
		l.descriptor,
		mode.c(),
		a.CPtr(),
		xD.descriptor, x, means, dy, temp, temp2,
		b.CPtr(),
		dXdMeansDesc.descriptor, dx, dMeans,
	)).error("(l *LRND) DivisiveNormalizationBackwardUS")
}

//LRNmode is used for the flags in LRNmode
type LRNmode C.cudnnLRNMode_t

func (l LRNmode) c() C.cudnnLRNMode_t { return C.cudnnLRNMode_t(l) }

//CrossChanelDim1 sets l to and returns LRNmode( C.CUDNN_LRN_CROSS_CHANNEL_DIM1)
func (l *LRNmode) CrossChanelDim1() LRNmode { *l = LRNmode(C.CUDNN_LRN_CROSS_CHANNEL_DIM1); return *l }
func (l LRNmode) String() string {
	flg := l
	var s string
	switch l {
	case flg.CrossChanelDim1():
		s = "CrossChanelDim1"
	}
	return "LRNmode: " + s
}

//DivNormMode is usde for C.cudnnDivNormMode_t flags
type DivNormMode C.cudnnDivNormMode_t

//PrecomputedMeans sets d to and returns DivNormMode(C.CUDNN_DIVNORM_PRECOMPUTED_MEANS)
func (d *DivNormMode) PrecomputedMeans() DivNormMode {
	*d = DivNormMode(C.CUDNN_DIVNORM_PRECOMPUTED_MEANS)
	return *d
}
func (d DivNormMode) String() string {
	flg := d
	var s string
	switch d {
	case flg.PrecomputedMeans():
		s = "PrecomputedMeans"
	}
	return "DivNormMode: " + s
}
func (d DivNormMode) c() C.cudnnDivNormMode_t { return C.cudnnDivNormMode_t(d) }
