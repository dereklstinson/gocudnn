package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"
)

//SeqDataAxis is a flag type setting and returning SeqDataAxis flags through methods
//Caution: Methods will also change the value of variable that calls the method.
//		   If you need to make a case switch make another variable and call it flag and use that.
type SeqDataAxis C.cudnnSeqDataAxis_t

//Time index in time.
//Method sets type to Time and returns Time value.
func (s *SeqDataAxis) Time() SeqDataAxis {
	*s = SeqDataAxis(C.CUDNN_SEQDATA_TIME_DIM)
	return *s
}

//Batch -index in batch
//Method sets type to Batch and returns Batch value
func (s *SeqDataAxis) Batch() SeqDataAxis { *s = SeqDataAxis(C.CUDNN_SEQDATA_BATCH_DIM); return *s }

//Beam -index in beam
//Method sets type to Beam and returns Beam value
func (s *SeqDataAxis) Beam() SeqDataAxis { *s = SeqDataAxis(C.CUDNN_SEQDATA_BEAM_DIM); return *s }

//Vect -index in Vector
//Method sets type to Vect and returns Vect value
func (s *SeqDataAxis) Vect() SeqDataAxis { *s = SeqDataAxis(C.CUDNN_SEQDATA_VECT_DIM); return *s }
func (s SeqDataAxis) c() C.cudnnSeqDataAxis_t {
	return C.cudnnSeqDataAxis_t(s)
}

//CudnnSeqDataDimCount is a flag for the number of dims.
const CudnnSeqDataDimCount = C.CUDNN_SEQDATA_DIM_COUNT

//SeqDataD holds C.cudnnSeqDataDescriptor_t
type SeqDataD struct {
	descriptor      C.cudnnSeqDataDescriptor_t
	seqlenarraysize C.size_t
	nbDims          C.int
	padding         float64
	finalizer       bool
}

func cudnnCreateSeqDataDescriptor() (seqdata *SeqDataD, err error) {
	seqdata = new(SeqDataD)
	err = Status(C.cudnnCreateSeqDataDescriptor(&seqdata.descriptor)).error("cudnnCreateSeqDataDescriptor")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		seqdata.finalizer = true
		runtime.SetFinalizer(seqdata, cudnnDestroySeqDataDescriptor)

	}

	return seqdata, err
}
func cudnnDestroySeqDataDescriptor(s *SeqDataD) error {

	err := Status(C.cudnnDestroySeqDataDescriptor(s.descriptor)).error("cudnnDestroySeqDataDescriptor")
	if err != nil {
		return err
	}
	s = nil
	return nil
}

//Destroy will destroy the descriptor
//For now since everything is on the runtime. This will run the garbage collector.
//Sometime in the future. I will have an option to turn on and off the GC.
func (s *SeqDataD) Destroy() error {
	if s.finalizer || setfinalizer {
		runtime.GC()
		return nil
	}
	return cudnnDestroySeqDataDescriptor(s)

}

//cudnnSetSeqDataDescriptor- from reading the documentation this is what it seems like how you set it up, and the possible work around with gocudnn.
//
//len(dimsA) && len(axes) needs to equal 4. len(seqLengthArray) needs to be < dimsA[(*seqDataAxis).Time()]
//
//
//dimsA - contains the dims of the buffer that holds a batch of sequence samples.  all vals need to be positive.
//
//	dimsA[(*seqDataAxis).Time()]=is the maximum allowed sequence length
//
//	dimsA[(*seqDataAxis).Batch()]= is the maximum allowed batch size
//
//	dimsA[(*seqDataAxis).Beam()]= is the number of beam in each sample
//
//	dimsA[(*seqDataAxis).Vect()]= is the vector length.
//
//
//axes- order in which the axes are in. Needs to be in order of outermost to inner most.
//Kind of like an NCHW tensor where N is the outer and w is the inner.
//
//	Example:
//
//	var s SeqDataAxis
//
//	axes:=[]SeqDataAxis{s.Batch(), s.Time(),s.Beam(),s.Vect()}
//
//seqLengthArray - Array that holds the sequence lengths of each sequence.
//paddingfill - Points to a value, of dataType, that is used to fill up the buffer beyond the sequence length of each sequence. The only supported value for paddingFill is 0.
//paddingfill is autoconverted to the datatype that it needs in the function
func (s *SeqDataD) cudnnSetSeqDataDescriptor(dtype DataType, dimsA []int32, axes []SeqDataAxis, seqLengthArray []int32, paddingfill float64) error {
	pf := cscalarbydatatype(dtype, paddingfill)
	s.nbDims = (C.int)(len(dimsA))
	s.seqlenarraysize = (C.size_t)(len(seqLengthArray))
	s.padding = paddingfill
	return Status(C.cudnnSetSeqDataDescriptor(s.descriptor, dtype.c(), s.nbDims, (*C.int)(&dimsA[0]), (*C.cudnnSeqDataAxis_t)(&axes[0]), s.seqlenarraysize, (*C.int)(&seqLengthArray[0]), pf.CPtr())).error("cudnnSetSeqDataDescriptor")
}
func (s *SeqDataD) cudnnGetSeqDataDescriptor() (dtype DataType, dimsA []int32, axes []SeqDataAxis, seqLengthArray []int32, paddingfill float64, err error) {

	dimsA = make([]int32, s.nbDims)
	axes = make([]SeqDataAxis, s.nbDims)
	seqLengthArray = make([]int32, s.seqlenarraysize)
	actualnb := s.nbDims
	actualseq := s.seqlenarraysize
	holder := 0.0
	pf := cscalarbydatatype(dtype, holder)
	err = Status(C.cudnnGetSeqDataDescriptor(s.descriptor, (*C.cudnnDataType_t)(&dtype), &actualnb, s.nbDims, (*C.int)(&dimsA[0]), (*C.cudnnSeqDataAxis_t)(&axes[0]), &actualseq, s.seqlenarraysize, (*C.int)(&seqLengthArray[0]), pf.CPtr())).error("cudnnSetSeqDataDescriptor")

	paddingfill = s.padding
	return

}
