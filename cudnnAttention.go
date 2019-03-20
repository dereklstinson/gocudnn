package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//AttnQueryMap type is a flag for multihead attention.  Flags are exposed through type methods.
type AttnQueryMap C.cudnnAttnQueryMap_t

//AllToOne - multiple Q-s when beam width > 1 map to a single (K,V) set.
//Method changes to AllToOne, and returns that value.
func (a *AttnQueryMap) AllToOne() AttnQueryMap {
	*a = AttnQueryMap(C.CUDNN_ATTN_QUERYMAP_ALL_TO_ONE)
	return *a
}

//OneToOne - multiple Q-s when beam width > 1 map to corresponding (K,V) sets.
//Method changes to OneToOne, and returns that value.
func (a *AttnQueryMap) OneToOne() AttnQueryMap {
	*a = AttnQueryMap(C.CUDNN_ATTN_QUERYMAP_ONE_TO_ONE)
	return *a
}
func (a AttnQueryMap) c() C.cudnnAttnQueryMap_t {
	return C.cudnnAttnQueryMap_t(a)
}

//AttentionD holds opaque values used for attention operations
type AttentionD struct {
	descriptor C.cudnnAttnDescriptor_t
}

//NewcAttentionD creates and sets a AttnDescriptor
func NewcAttentionD(
	qMap AttnQueryMap,
	nHead int32,
	smScaler float64,
	dtype DataType,
	computePrecision DataType,
	mtype MathType,
	attn *DropOutD,
	post *DropOutD,
	qSize, kSize, vSize int32,
	qProjSize, kProjSize, vProjSize, oProjSize int32,
	qoMaxSeqLen, kvMaxSeqLen int32,
	maxBatchSize, maxBeamSize int32,
) (*AttentionD, error) {
	d := new(AttentionD)
	err := Status(C.cudnnCreateAttnDescriptor(&d.descriptor)).error("NewAttnDescriptor-cudnnCreateAttnDescriptor")
	if err != nil {
		return nil, err
	}
	x := func(y int32) C.int { //I did this so I didn't have to constantly type (C.int)(value)
		return (C.int)(y)
	}
	err = Status(C.cudnnSetAttnDescriptor(d.descriptor,
		qMap.c(),
		x(nHead),
		(C.double)(smScaler),
		dtype.c(),
		computePrecision.c(),
		mtype.c(),
		attn.descriptor,
		post.descriptor,
		x(qSize), x(kSize), x(vSize),
		x(qProjSize), x(kProjSize), x(vProjSize), x(oProjSize),
		x(qoMaxSeqLen), x(kvMaxSeqLen), x(maxBatchSize), x(maxBeamSize),
	)).error("NewAttnDescriptor-cudnnSetAttnDescriptor")
	runtime.SetFinalizer(d, cudnnDestroyAttnDescriptor)
	if err != nil {
		return nil, err
	}
	return d, nil
}
func cudnnDestroyAttnDescriptor(d *AttentionD) error {
	err := Status(C.cudnnDestroyAttnDescriptor(d.descriptor)).error("cudnnDestroyAttnDescriptor")
	if err != nil {
		return err
	}
	d = nil
	return nil
}

//Get gets all the values for the AttentionD - There is a lot.
func (a *AttentionD) Get() (
	qMap AttnQueryMap,
	nHead int32,
	smScaler float64,
	dtype DataType,
	computePrecision DataType,
	mtype MathType,
	attn *DropOutD,
	post *DropOutD,
	qSize, kSize, vSize int32,
	qProjSize, kProjSize, vProjSize, oProjSize int32,
	qoMaxSeqLen, kvMaxSeqLen int32,
	maxBatchSize, maxBeamSize int32,
	err error) {
	x := func(y C.int) int32 {
		return (int32)(y)
	}
	var nh, qs, ks, vs, qps, kps, vps, ops, qom, kvm, mbas, mbes C.int
	var sms C.double
	var qm C.cudnnAttnQueryMap_t
	var dt, cp C.cudnnDataType_t
	var mt C.cudnnMathType_t
	attn = new(DropOutD)
	post = new(DropOutD)
	err = Status(C.cudnnGetAttnDescriptor(a.descriptor, &qm, &nh, &sms, &dt, &cp, &mt, &attn.descriptor, &post.descriptor, &qs, &ks, &vs, &qps, &kps, &vps, &ops, &qom, &kvm, &mbas, &mbes)).error("cudnnGetAttnDescriptor")
	if err != nil {
		return
	}
	qMap = AttnQueryMap(qm)
	nHead, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLen, kvMaxSeqLen, maxBatchSize, maxBeamSize = x(nh), x(qs), x(ks), x(vs), x(qps), x(kps), x(vps), x(ops), x(qom), x(kvm), x(mbas), x(mbes)
	smScaler = (float64)(sms)
	dtype, computePrecision = (DataType)(dt), (DataType)(cp)
	mtype = (MathType)(mt)
	return qMap, nHead, smScaler, dtype, computePrecision, mtype, attn, post, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLen, kvMaxSeqLen, maxBatchSize, maxBeamSize, nil
}

//GetMultiHeadBuffers returns the Size In Bytes (SIB) needed for allocation for operation.
func (a *AttentionD) GetMultiHeadBuffers(h *Handle) (weightbuffSIB, wspaceSIB, rspaceSIB uint, err error) {
	var weight, wspace, rspace C.size_t
	err = Status(C.cudnnGetMultiHeadAttnBuffers(h.x, a.descriptor, &weight, &wspace, &rspace)).error("cudnnGetMultiHeadAttnBuffers")
	if err != nil {
		return
	}
	weightbuffSIB, wspaceSIB, rspaceSIB = (uint)(weight), (uint)(wspace), (uint)(rspace)
	return weightbuffSIB, wspaceSIB, rspaceSIB, err
}

//MultiHeadAttnWeightKind is a flag for the kind of weights used flags are exposed through type's methods.
type MultiHeadAttnWeightKind C.cudnnMultiHeadAttnWeightKind_t

//Queries - sets value to MultiHeadAttnWeightKind(C.CUDNN_MH_ATTN_Q_WEIGHTS) and returns that value.
//From cudnn.h - input projection weights for 'queries'
func (m *MultiHeadAttnWeightKind) Queries() MultiHeadAttnWeightKind {
	*m = MultiHeadAttnWeightKind(C.CUDNN_MH_ATTN_Q_WEIGHTS)
	return *m
}
func (m MultiHeadAttnWeightKind) c() C.cudnnMultiHeadAttnWeightKind_t {
	return C.cudnnMultiHeadAttnWeightKind_t(m)
}

//Keys - sets value to MultiHeadAttnWeightKind(C.CUDNN_MH_ATTN_K_WEIGHTS) and returns that value.
//From cudnn.h -input projection weights for 'keys'
func (m *MultiHeadAttnWeightKind) Keys() MultiHeadAttnWeightKind {
	*m = MultiHeadAttnWeightKind(C.CUDNN_MH_ATTN_K_WEIGHTS)
	return *m
}

//Values - sets value to MultiHeadAttnWeightKind(C.CUDNN_MH_ATTN_V_WEIGHTS) and returns that value.
//From cudnn.h - input projection weights for 'values'
func (m *MultiHeadAttnWeightKind) Values() MultiHeadAttnWeightKind {
	*m = MultiHeadAttnWeightKind(C.CUDNN_MH_ATTN_V_WEIGHTS)
	return *m
}

//Output - sets value to MultiHeadAttnWeightKind(C.CUDNN_MH_ATTN_O_WEIGHTS) and returns that value.
//From cudnn.h -  output projection weights
func (m *MultiHeadAttnWeightKind) Output() MultiHeadAttnWeightKind {
	*m = MultiHeadAttnWeightKind(C.CUDNN_MH_ATTN_O_WEIGHTS)
	return *m
}

//GetMultiHeadAttnWeights returns a Descripter for w and its goco.Mem
func (a *AttentionD) GetMultiHeadAttnWeights(h *Handle, wkind MultiHeadAttnWeightKind, wbuffSIB uint, wbuff gocu.Mem) (wD *TensorD, w gocu.Mem, err error) {
	w = new(gocu.CudaPtr)
	wD, err = createtensordescriptor(true)
	if err != nil {
		return nil, nil, err
	}
	err = Status(C.cudnnGetMultiHeadAttnWeights(h.x, a.descriptor, wkind.c(), C.size_t(wbuffSIB), w.Ptr(), wD.descriptor, w.DPtr())).error("cudnnGetMultiHeadAttnWeights")
	if err != nil {
		return nil, nil, err
	}
	wD.frmt, wD.dtype, wD.dimsarray, wD.stride, err = wD.GetDescrptor()
	if err != nil {
		return
	}
	return wD, w, err
}

//Forward look at documentation.  Kind of more confusing than normal
//if currIdx <0  trainingmode, currIdx >=0 inference mode
func (a *AttentionD) Forward(
	h *Handle,
	currIdx int32, //if currIdx <0  trainingmode, currIdx >=0 inference mode
	loWinIdx []int32, // array of lower (inclusive) key and value time step windows
	hiWinIdx []int32, // array of upper (exclusive) key and value time step windows
	seqLengthArrayQRO []int32, // array of lengths for for queries,residuals,and out
	seqLengthArrayKV []int32, // array of lengths for keys and values
	qrDesc *SeqDataD, queries, residuals gocu.Mem,
	kDesc *SeqDataD, keys gocu.Mem,
	vDesc *SeqDataD, values gocu.Mem,
	oDesc *SeqDataD, out gocu.Mem,
	wbuffSIB uint, wbuff gocu.Mem,
	wspaceSIB uint, wspace gocu.Mem,
	rspaceSIB uint, rspace gocu.Mem) error {
	lo := int32Tocint(loWinIdx)
	hi := int32Tocint(hiWinIdx)
	QRO := int32Tocint(seqLengthArrayQRO)
	KV := int32Tocint(seqLengthArrayKV)
	return Status(C.cudnnMultiHeadAttnForward(h.x, a.descriptor, (C.int)(currIdx), &lo[0], &hi[0], &QRO[0], &KV[0],
		qrDesc.descriptor, queries.Ptr(), residuals.Ptr(), kDesc.descriptor, keys.Ptr(), vDesc.descriptor, values.Ptr(), oDesc.descriptor, out.Ptr(),
		(C.size_t)(wbuffSIB), wbuff.Ptr(), (C.size_t)(wspaceSIB), wspace.Ptr(), (C.size_t)(rspaceSIB), rspace.Ptr())).error("Forward")

}

//BackwardData does the backward propigation for data.
func (a *AttentionD) BackwardData(
	h *Handle,
	loWinIdx []int32, // array of lower (inclusive) key and value time step windows
	hiWinIdx []int32, // array of upper (exclusive) key and value time step windows
	seqLengthArrayDQDO []int32, //array of lengths for dqueries and dout
	seqLengthArrayDKDV []int32, //array of lengths for dkeys and dvalues
	doDesc *SeqDataD, dout gocu.Mem,
	dqDesc *SeqDataD, dqueries, queries gocu.Mem, //dqueries is output
	dkDesc *SeqDataD, dkeys, keys gocu.Mem, //dkeys is output
	dvDesc *SeqDataD, dvalues, values gocu.Mem, //dvalues is output
	wbuffSIB uint, wbuff gocu.Mem, wspaceSIB uint, wspace gocu.Mem, rspaceSIB uint, rspace gocu.Mem) error {
	lo := int32Tocint(loWinIdx)
	hi := int32Tocint(hiWinIdx)
	DQDO := int32Tocint(seqLengthArrayDQDO)
	DKDV := int32Tocint(seqLengthArrayDKDV)
	return Status(C.cudnnMultiHeadAttnBackwardData(h.x, a.descriptor, &lo[0], &hi[0], &DQDO[0], &DKDV[0],
		doDesc.descriptor, dout.Ptr(),
		dqDesc.descriptor, dqueries.Ptr(), queries.Ptr(),
		dkDesc.descriptor, dkeys.Ptr(), keys.Ptr(),
		dvDesc.descriptor, dvalues.Ptr(), values.Ptr(),
		(C.size_t)(wbuffSIB), wbuff.Ptr(),
		(C.size_t)(wspaceSIB), wspace.Ptr(),
		(C.size_t)(rspaceSIB), rspace.Ptr())).error("BackwardData")
}

//WgradMode is used for flags and can be changed through methods
type WgradMode C.cudnnWgradMode_t

func (w WgradMode) c() C.cudnnWgradMode_t {
	return (C.cudnnWgradMode_t)(w)
}

//Add sets w to Add and returns Add flag
func (w *WgradMode) Add() WgradMode {
	*w = WgradMode(C.CUDNN_WGRAD_MODE_ADD)
	return *w
}

//Set sets w to Set and returns Set flag
func (w *WgradMode) Set() WgradMode {
	*w = WgradMode(C.CUDNN_WGRAD_MODE_SET)
	return *w
}

//BackwardWeights does the backward propigation for weights.
func (a *AttentionD) BackwardWeights(
	h *Handle,
	wgmode WgradMode,
	qDesc *SeqDataD, queries gocu.Mem,
	kDesc *SeqDataD, keys gocu.Mem,
	vDesc *SeqDataD, values gocu.Mem,
	doDesc *SeqDataD, dout gocu.Mem,
	wbuffSIB uint, wbuff, dwbuff gocu.Mem,
	wspaceSIB uint, wspace gocu.Mem, rspaceSIB uint, rspace gocu.Mem) error {

	return Status(C.cudnnMultiHeadAttnBackwardWeights(h.x, a.descriptor, wgmode.c(),
		qDesc.descriptor, queries.Ptr(),
		kDesc.descriptor, keys.Ptr(),
		vDesc.descriptor, values.Ptr(),
		doDesc.descriptor, dout.Ptr(),
		(C.size_t)(wbuffSIB), wbuff.Ptr(), dwbuff.Ptr(),
		(C.size_t)(wspaceSIB), wspace.Ptr(),
		(C.size_t)(rspaceSIB), rspace.Ptr())).error("BackwardData")
}
