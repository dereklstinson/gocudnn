package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

//RNNDataD is a RNNDataDescriptor
type RNNDataD struct {
	d          C.cudnnRNNDataDescriptor_t
	dtype      DataType
	seqlensize int32
	gogc       bool
}

//CreateRNNDataD creates an RNNDataD through cudnn's cudnnCreateRNNDataDescriptor
//This is put into the runtime for GC
func CreateRNNDataD() (*RNNDataD, error) {
	d := new(RNNDataD)
	err := Status(C.cudnnCreateRNNDataDescriptor(&d.d)).error("CreateRNNDataD")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		runtime.SetFinalizer(d, destroyrnndatadescriptor)
	}

	return d, nil
}

/*Set sets the RNNDataD
dataType -  The datatype of the RNN data tensor. See cudnnDataType_t.
layout - The memory layout of the RNN data tensor.
maxSeqLength - The maximum sequence length within this RNN data tensor. In the unpacked (padded) layout, this should include the padding vectors in each sequence. In the packed (unpadded) layout, this should be equal to the greatest element in seqLengthArray.
vectorSize -The vector length (i.e. embedding size) of the input or output tensor at each timestep.
seqLengthArray - An integer array the size of the mini-batch number number of elements. Describes the length (i.e. number of timesteps) of each sequence. Each element in seqLengthArray must be greater than 0 but less than or equal to maxSeqLength. In the packed layout, the elements should be sorted in descending order, similar to the layout required by the non-extended RNN compute functions.

paddingFill -  For gocudnn it will auto typecast the value into the correct datatype. Just put the value you want used as an float64.
			  From Documentation:
			  A user-defined symbol for filling the padding position in RNN output.
			  This is only effective when the descriptor is describing the RNN output, and the unpacked layout is specified.
			  The symbol should be in the host memory, and is interpreted as the same data type as that of the RNN data tensor.


*/
func (r *RNNDataD) Set(dtype DataType, layout RNNDataLayout,
	maxSeqLength, vectorsize int32, seqLengthArray []int32, paddingsymbol float64) error {
	r.dtype = dtype
	symbol := cscalarbydatatypeforsettensor(dtype, paddingsymbol)
	batchsize := len(seqLengthArray)
	seqlenarray := int32Tocint(seqLengthArray)
	return Status(C.cudnnSetRNNDataDescriptor(r.d, dtype.c(), layout.c(), (C.int)(maxSeqLength), (C.int)(batchsize), (C.int)(vectorsize), &seqlenarray[0], symbol.CPtr())).error("(*RNNDataD)Set")
}

//Get gets the parameters used in Set for RNNDataD
func (r *RNNDataD) Get() (dtype DataType, layout RNNDataLayout, maxSeqLength, vectorsize int32, seqLengthArray []int32, paddingsymbol float64, err error) {
	ps := cscalarbydatatypeforsettensor(r.dtype, paddingsymbol)
	sla := make([]C.int, r.seqlensize)
	var (
		cdtype C.cudnnDataType_t
		lo     C.cudnnRNNDataLayout_t
		msl    C.int
		bs     C.int
		vs     C.int
	)
	err = Status(C.cudnnGetRNNDataDescriptor(r.d, &cdtype, &lo, &msl, &bs, &vs, C.int(r.seqlensize), &sla[0], ps.CPtr())).error("(*RNNDATAD)Get")
	dtype = DataType(cdtype)
	layout = RNNDataLayout(lo)
	maxSeqLength = int32(msl)
	vectorsize = int32(vs)

	paddingsymbol = cutil.CScalartoFloat64(ps)

	if r.seqlensize > int32(bs) {
		seqLengthArray = cintToint32(sla[:bs])
	} else {
		seqLengthArray = cintToint32(sla)
	}

	return dtype, layout, maxSeqLength, vectorsize, seqLengthArray, paddingsymbol, err

}

//Destroy destorys descriptor unless gogc is being used in which it will just return nil
func (r *RNNDataD) Destroy() error {
	if setfinalizer || r.gogc {
		return nil
	}
	err := destroyrnndatadescriptor(r)
	if err != nil {
		return err
	}
	r = nil
	return nil
}
func destroyrnndatadescriptor(d *RNNDataD) error {
	err := Status(C.cudnnDestroyRNNDataDescriptor(d.d)).error("destroyrnndatadescriptor")
	d = nil
	return err
}

//RNNDataLayout are used for flags for data layout
type RNNDataLayout C.cudnnRNNDataLayout_t

func (r RNNDataLayout) c() C.cudnnRNNDataLayout_t {
	return C.cudnnRNNDataLayout_t(r)
}

func (r *RNNDataLayout) cptr() *C.cudnnRNNDataLayout_t {
	return (*C.cudnnRNNDataLayout_t)(r)
}

//SeqMajorUnPacked sets r to and returns CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED flag
func (r *RNNDataLayout) SeqMajorUnPacked() RNNDataLayout {
	*r = RNNDataLayout(C.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED)
	return *r
}

//SeqMajorPacked sets r to  CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED flag
func (r *RNNDataLayout) SeqMajorPacked() RNNDataLayout {
	*r = RNNDataLayout(C.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED)
	return *r
}

//BatchMajorUnPacked sets r to  CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED flag
func (r *RNNDataLayout) BatchMajorUnPacked() RNNDataLayout {
	*r = RNNDataLayout(C.CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED)
	return *r
}

func (r RNNDataLayout) String() string {
	var x string
	f := r
	switch r {
	case f.BatchMajorUnPacked():
		x = "BatchMajorUnPacked"
	case f.SeqMajorPacked():
		x = "SeqMajorPacked"
	case f.SeqMajorUnPacked():
		x = "SeqMajorUnPacked"
	default:
		x = "Unsupported Flag"
	}
	return "RNNDataLayout: " + x
}

//RNNPaddingMode is the padding mode flag
type RNNPaddingMode C.cudnnRNNPaddingMode_t

func (r RNNPaddingMode) c() C.cudnnRNNPaddingMode_t {
	return C.cudnnRNNPaddingMode_t(r)
}
func (r *RNNPaddingMode) cptr() *C.cudnnRNNPaddingMode_t {
	return (*C.cudnnRNNPaddingMode_t)(r)
}

//Disabled sets r to and returns RNNPaddingMode(C.CUDNN_RNN_PADDED_IO_DISABLED)
func (r *RNNPaddingMode) Disabled() RNNPaddingMode {
	*r = RNNPaddingMode(C.CUDNN_RNN_PADDED_IO_DISABLED)
	return *r
}

//Enabled sets r to and returns RNNPaddingMode(C.CUDNN_RNN_PADDED_IO_ENABLED)
func (r *RNNPaddingMode) Enabled() RNNPaddingMode {
	*r = RNNPaddingMode(C.CUDNN_RNN_PADDED_IO_ENABLED)
	return *r
}
func (r RNNPaddingMode) String() string {
	var x string
	f := r
	switch r {
	case f.Disabled():
		x = "Disabled"
	case f.Enabled():
		x = "Enabled"
	default:
		x = "Unsupported Flag"
	}
	return "RNNPaddingMode: " + x
}

//SetPaddingMode sets the padding mode with flag passed
func (r *RNND) SetPaddingMode(mode RNNPaddingMode) error {
	return Status(C.cudnnSetRNNPaddingMode(r.descriptor, mode.c())).error("SetRNNPaddingMode")
}

//GetPaddingMode gets padding mode for the descriptor
func (r *RNND) GetPaddingMode() (mode RNNPaddingMode, err error) {

	err = Status(C.cudnnGetRNNPaddingMode(r.descriptor, mode.cptr())).error("GetRNNPaddingMode")
	return mode, err
}

/*ForwardTrainingEx - From cudnn documentation
This routine is the extended version of the cudnnRNNForwardTraining function.
The ForwardTrainingEx allows the user to use unpacked (padded) layout for input x and output y.
In the unpacked layout, each sequence in the mini-batch is considered to be of fixed length, specified by
maxSeqLength in its corresponding RNNDataDescriptor. Each fixed-length sequence, for example,
the nth sequence in the mini-batch, is composed of a valid segment specified by the seqLengthArray[n]
in its corresponding RNNDataDescriptor; and a padding segment to make the combined sequence length equal to maxSeqLength.
With the unpacked layout, both sequence major (i.e. time major) and batch major are supported.
For backward compatibility, the packed sequence major layout is supported.
However, similar to the non-extended function cudnnRNNForwardTraining, the sequences
 in the mini-batch need to be sorted in descending order according to length.

Parameters:

handle - Input. Handle to a previously created cuDNN context.

xD - Input. A previously initialized RNN Data descriptor. The dataType, layout, maxSeqLength , batchSize, and seqLengthArray need to match that of yD.

x - Input. Data pointer to the GPU memory associated with the RNN data descriptor xD.
		   The input vectors are expected to be laid out in memory according to the layout specified by xD.
		   The elements in the tensor (including elements in the padding vector) must be densely packed, and no strides are supported.

hxD - Input. A fully packed tensor descriptor describing the initial hidden state of the RNN.
			 The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. Moreover:
			 If direction is CUDNN_UNIDIRECTIONAL then the first dimension should match the numLayers argument passed to cudnnSetRNNDescriptor.
			 If direction is CUDNN_BIDIRECTIONAL then the first dimension should match double the numLayers argument passed to cudnnSetRNNDescriptor.
			 The second dimension must match the batchSize parameter in xD.
			 The third dimension depends on whether RNN mode is CUDNN_LSTM and whether LSTM projection is enabled. Moreover:
			 If RNN mode is CUDNN_LSTM and LSTM projection is enabled, the third dimension must match the
			 recProjSize argument passed to cudnnSetRNNProjectionLayers call used to set rnnDesc.
			 Otherwise, the third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc .

hx - Input. Data pointer to GPU memory associated with the tensor descriptor hxD.
			If a NULL pointer is passed, the initial hidden state of the network will be initialized to zero.

cxD - Input. A fully packed tensor descriptor describing the initial cell state for LSTM networks.
			 The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. Moreover:
			 If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the numLayers argument passed to cudnnSetRNNDescriptor.
			 If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the numLayers argument passed to cudnnSetRNNDescriptor.
			 The second dimension must match the first dimension of the tensors described in xD.
			 The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.

cx - Input. Data pointer to GPU memory associated with the tensor descriptor cxD. If a NULL pointer
			is passed, the initial cell state of the network will be initialized to zero.

wD - Input. Handle to a previously initialized filter descriptor describing the weights for the RNN.

w- Input. Data pointer to GPU memory associated with the filter descriptor wD.

yD - Input. A previously initialized RNN data descriptor. The dataType, layout, maxSeqLength , batchSize, and seqLengthArray
	        need to match that of dyD and dxD. The parameter vectorSize depends on whether RNN mode is CUDNN_LSTM and
			whether LSTM projection is enabled and whether the network is bidirectional.
			In specific: For uni-directional network, if RNN mode is CUDNN_LSTM and LSTM projection is enabled,
			the parameter vectorSize must match the recProjSize argument passed to cudnnSetRNNProjectionLayers
			call used to set rnnDesc. If the network is bidirectional, then multiply the value by 2.
			Otherwise, for uni-directional network, the parameter vectorSize must match the
			hiddenSize argument passed to the cudnnSetRNNDescriptor call used
			to initialize rnnDesc. If the network is bidirectional, then multiply the value by 2.

y - Output. Data pointer to GPU memory associated with the RNN data descriptor yD.
			The input vectors are expected to be laid out in memory according to the layout
			specified by yD. The elements in the tensor (including elements in the padding vector)
			must be densely packed, and no strides are supported.

hyD - Input. A fully packed tensor descriptor describing the final hidden state of the RNN. The descriptor must be set exactly the same as hxD.

hy - Output. Data pointer to GPU memory associated with the tensor descriptor hyD. If a NULL pointer is passed, the final hidden state of the network will not be saved.

cyD - Input. A fully packed tensor descriptor describing the final cell state for LSTM networks. The descriptor must be set exactly the same as cxD.

cy- Output. Data pointer to GPU memory associated with the tensor descriptor cyD. If a NULL pointer is passed, the final cell state of the network will be not be saved.

wspace - Input. Data pointer to GPU memory to be used as a wspace for this call.

wspacesib - Input. Specifies the size in bytes of the provided wspace.

rspace -Input/Output. Data pointer to GPU memory to be used as a reserve space for this call.

rspacesib - Input. Specifies the size in bytes of the provided rspace
*/
func (r *RNND) ForwardTrainingEx(h *Handle,
	xD *RNNDataD, x cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	cxD *TensorD, cx cutil.Mem,
	wD *FilterD, w cutil.Mem,
	yD *RNNDataD, y cutil.Mem,
	hyD *TensorD, hy cutil.Mem,
	cyD *TensorD, cy cutil.Mem,
	wspace cutil.Mem, wspacesib uint,
	rspace cutil.Mem, rspacesib uint) error {
	if h.w != nil {
		return h.w.Work(func() error {
			return Status(C.cudnnRNNForwardTrainingEx(h.x,
				r.descriptor,
				xD.d, x.Ptr(),
				hxD.descriptor, hx.Ptr(),
				cxD.descriptor, cx.Ptr(),
				wD.descriptor, w.Ptr(),
				yD.d, y.Ptr(),
				hyD.descriptor, hy.Ptr(),
				cyD.descriptor, cy.Ptr(),
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				wspace.Ptr(),
				C.size_t(wspacesib),
				rspace.Ptr(),
				C.size_t(rspacesib))).error("(r *RNND) ForwardTrainingEx")
		})
	}
	return Status(C.cudnnRNNForwardTrainingEx(h.x,
		r.descriptor,
		xD.d, x.Ptr(),
		hxD.descriptor, hx.Ptr(),
		cxD.descriptor, cx.Ptr(),
		wD.descriptor, w.Ptr(),
		yD.d, y.Ptr(),
		hyD.descriptor, hy.Ptr(),
		cyD.descriptor, cy.Ptr(),
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		wspace.Ptr(),
		C.size_t(wspacesib),
		rspace.Ptr(),
		C.size_t(rspacesib))).error("(r *RNND) ForwardTrainingEx")

}

//ForwardTrainingExUS is like ForwardTrainingEx but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) ForwardTrainingExUS(h *Handle,
	xD *RNNDataD, x unsafe.Pointer,
	hxD *TensorD, hx unsafe.Pointer,
	cxD *TensorD, cx unsafe.Pointer,
	wD *FilterD, w unsafe.Pointer,
	yD *RNNDataD, y unsafe.Pointer,
	hyD *TensorD, hy unsafe.Pointer,
	cyD *TensorD, cy unsafe.Pointer,
	wspace unsafe.Pointer, wspacesib uint,
	rspace unsafe.Pointer, rspacesib uint) error {
	if h.w != nil {
		return h.w.Work(func() error {
			return Status(C.cudnnRNNForwardTrainingEx(h.x,
				r.descriptor,
				xD.d, x,
				hxD.descriptor, hx,
				cxD.descriptor, cx,
				wD.descriptor, w,
				yD.d, y,
				hyD.descriptor, hy,
				cyD.descriptor, cy,
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				wspace, C.size_t(wspacesib),
				rspace, C.size_t(rspacesib))).error("(r *RNND) ForwardTrainingExUS")
		})
	}
	return Status(C.cudnnRNNForwardTrainingEx(h.x,
		r.descriptor,
		xD.d, x,
		hxD.descriptor, hx,
		cxD.descriptor, cx,
		wD.descriptor, w,
		yD.d, y,
		hyD.descriptor, hy,
		cyD.descriptor, cy,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		wspace, C.size_t(wspacesib),
		rspace, C.size_t(rspacesib))).error("(r *RNND) ForwardTrainingExUS")

}

/*ForwardInferenceEx - from cudnn documentation
This routine is the extended version of the cudnnRNNForwardInference function.
The ForwardTrainingEx allows the user to use unpacked (padded) layout for input x and output y.
In the unpacked layout, each sequence in the mini-batch is considered to be of fixed length, specified by maxSeqLength in its corresponding RNNDataDescriptor.
Each fixed-length sequence, for example, the nth sequence in the mini-batch, is composed of a valid segment,
specified by the seqLengthArray[n] in its corresponding RNNDataDescriptor, and a padding segment to make the combined sequence length equal to maxSeqLength.

With unpacked layout, both sequence major (i.e. time major) and batch major are supported.
For backward compatibility, the packed sequence major layout is supported.
However, similar to the non-extended function cudnnRNNForwardInference, the sequences in the mini-batch need to be sorted in descending order according to length.

Parameters

handle - Input. Handle to a previously created cuDNN context.

xD- Input. A previously initialized RNN Data descriptor. The dataType, layout, maxSeqLength , batchSize, and seqLengthArray need to match that of yD.
x -Input. Data pointer to the GPU memory associated with the RNN data descriptor xD. The vectors are expected to be laid out in memory according to the layout specified by xD.
		  The elements in the tensor (including elements in the padding vector) must be densely packed, and no strides are supported.

hxD - Input. A fully packed tensor descriptor describing the initial hidden state of the RNN. The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc:
			 If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the numLayers argument passed to cudnnSetRNNDescriptor.
			 If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the numLayers argument passed to cudnnSetRNNDescriptor.
			 The second dimension must match the batchSize parameter described in xD.
			 The third dimension depends on whether RNN mode is CUDNN_LSTM and whether LSTM projection is enabled. In specific:
			 If RNN mode is CUDNN_LSTM and LSTM projection is enabled, the third dimension must match the recProjSize argument passed to cudnnSetRNNProjectionLayers call used to set rnnDesc.
			 Otherwise, the third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.

hx - Input. Data pointer to GPU memory associated with the tensor descriptor hxD. If a NULL pointer is passed, the initial hidden state of the network will be initialized to zero.

cxD -Input. A fully packed tensor descriptor describing the initial cell state for LSTM networks.
		    The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc:
		    If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the numLayers argument passed to cudnnSetRNNDescriptor.
		    If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the numLayers argument passed to cudnnSetRNNDescriptor.
		    The second dimension must match the batchSize parameter in xD. The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.

cx - Input. Data pointer to GPU memory associated with the tensor descriptor cxD.
	 	    If a NULL pointer is passed, the initial cell state of the network will be initialized to zero.

wD - Input. Handle to a previously initialized filter descriptor describing the weights for the RNN.

w - Input. Data pointer to GPU memory associated with the filter descriptor wD.

yD - Input. A previously initialized RNN data descriptor. The dataType, layout, maxSeqLength , batchSize, and seqLengthArray must match that of dyD and dxD.
		   The parameter vectorSize depends on whether RNN mode is CUDNN_LSTM and whether LSTM projection is enabled and whether the network is bidirectional.
		   In specific: For uni-directional network, if RNN mode is CUDNN_LSTM and LSTM projection is enabled,
		   the parameter vectorSize must match the recProjSize argument passed to cudnnSetRNNProjectionLayers call used to set rnnDesc.
		   If the network is bidirectional, then multiply the value by 2.
		   Otherwise, for uni-directional network, the parameter vectorSize must match the hiddenSize argument passed
		   to the cudnnSetRNNDescriptor call used to initialize rnnDesc. If the network is bidirectional, then multiply the value by 2.

y - Output. Data pointer to the GPU memory associated with the RNN data descriptor yD.
		   The vectors are expected to be laid out in memory according to the layout specified by yD.
		   The elements in the tensor (including elements in the padding vector) must be densely packed, and no strides are supported.

hyD - Input. A fully packed tensor descriptor describing the final hidden state of the RNN. The descriptor must be set exactly the same way as hxD.

hy - Output. Data pointer to GPU memory associated with the tensor descriptor hyD. If a NULL pointer is passed, the final hidden state of the network will not be saved.

cyD - Input. A fully packed tensor descriptor describing the final cell state for LSTM networks. The descriptor must be set exactly the same way as cxD.

cy -Output. Data pointer to GPU memory associated with the tensor descriptor cyD. If a NULL pointer is passed, the final cell state of the network will be not be saved.

wspace - Input. Data pointer to GPU memory to be used as a wspace for this call.

wspacesib - Input. Specifies the size in bytes of the provided wspace.
*/
func (r *RNND) ForwardInferenceEx(
	h *Handle,
	xD *RNNDataD, x cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	cxD *TensorD, cx cutil.Mem,
	wD *FilterD, w cutil.Mem,
	yD *RNNDataD, y cutil.Mem,
	hyD *TensorD, hy cutil.Mem,
	cyD *TensorD, cy cutil.Mem,
	wspace cutil.Mem, wspacesib uint,
) error {
	if h.w != nil {
		return h.w.Work(func() error {
			return Status(C.cudnnRNNForwardInferenceEx(h.x,
				r.descriptor,
				xD.d, x.Ptr(),
				hxD.descriptor, hx.Ptr(),
				cxD.descriptor, cx.Ptr(),
				wD.descriptor, w.Ptr(),
				yD.d, y.Ptr(),
				hyD.descriptor, hy.Ptr(),
				cyD.descriptor, cy.Ptr(),
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				wspace.Ptr(), C.size_t(wspacesib))).error(" (r *RNND) ForwardInferenceEx")
		})
	}
	return Status(C.cudnnRNNForwardInferenceEx(h.x,
		r.descriptor,
		xD.d, x.Ptr(),
		hxD.descriptor, hx.Ptr(),
		cxD.descriptor, cx.Ptr(),
		wD.descriptor, w.Ptr(),
		yD.d, y.Ptr(),
		hyD.descriptor, hy.Ptr(),
		cyD.descriptor, cy.Ptr(),
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		wspace.Ptr(), C.size_t(wspacesib))).error(" (r *RNND) ForwardInferenceEx")

}

//ForwardInferenceExUS is like ForwardInferenceEx but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) ForwardInferenceExUS(
	h *Handle,
	xD *RNNDataD, x unsafe.Pointer,
	hxD *TensorD, hx unsafe.Pointer,
	cxD *TensorD, cx unsafe.Pointer,
	wD *FilterD, w unsafe.Pointer,
	yD *RNNDataD, y unsafe.Pointer,
	hyD *TensorD, hy unsafe.Pointer,
	cyD *TensorD, cy unsafe.Pointer,
	wspace unsafe.Pointer, wspacesib uint,
) error {
	if h.w != nil {
		return h.w.Work(func() error {
			return Status(C.cudnnRNNForwardInferenceEx(h.x,
				r.descriptor,
				xD.d, x,
				hxD.descriptor, hx,
				cxD.descriptor, cx,
				wD.descriptor, w,
				yD.d, y,
				hyD.descriptor, hy,
				cyD.descriptor, cy,
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				nil,
				wspace, C.size_t(wspacesib))).error("(r *RNND) ForwardInferenceExUS")
		})
	}
	return Status(C.cudnnRNNForwardInferenceEx(h.x,
		r.descriptor,
		xD.d, x,
		hxD.descriptor, hx,
		cxD.descriptor, cx,
		wD.descriptor, w,
		yD.d, y,
		hyD.descriptor, hy,
		cyD.descriptor, cy,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		wspace, C.size_t(wspacesib))).error("(r *RNND) ForwardInferenceExUS")

}

/*BackwardDataEx - Taken from cudnn documentation
This routine is the extended version of the function cudnnRNNBackwardData.
This function cudnnRNNBackwardDataEx allows the user to use unpacked (padded) layout for input y and output dx.
In the unpacked layout, each sequence in the mini-batch is considered to be of fixed length, specified by maxSeqLength in its corresponding RNNDataDescriptor.
Each fixed-length sequence, for example, the nth sequence in the mini-batch, is composed of a valid segment specified
by the seqLengthArray[n] in its corresponding RNNDataDescriptor; and a padding segment to make the combined sequence length equal to maxSeqLength.

With the unpacked layout, both sequence major (i.e. time major) and batch major are supported.
For backward compatibility, the packed sequence major layout is supported.
However, similar to the non-extended function cudnnRNNBackwardData, the sequences in the mini-batch need to be sorted in descending order according to length.

Parameters:

handle is handle passed to all cudnn funcs. needs to be initialized before using.

yD -Input. A previously initialized RNN data descriptor.
	Must match or be the exact same descriptor previously passed into ForwardTrainingEx.

y -Input. Data pointer to the GPU memory associated with the RNN data descriptor yD.
	The vectors are expected to be laid out in memory according to the layout specified by yD.
	The elements in the tensor (including elements in the padding vector) must be densely packed, and no strides are supported.
	Must contain the exact same data previously produced by ForwardTrainingEx.

dyD -Input. A previously initialized RNN data descriptor.
	The dataType, layout, maxSeqLength , batchSize, vectorSize and seqLengthArray need to match the yD previously passed to ForwardTrainingEx.

dy -Input.Data pointer to the GPU memory associated with the RNN data descriptor dyD.
	The vectors are expected to be laid out in memory according to the layout specified by dyD.
	The elements in the tensor (including elements in the padding vector) must be densely packed, and no strides are supported.

dhyD -Input. A fully packed tensor descriptor describing the gradients at the final hidden state of the RNN.
	The first dimension of the tensor depends on the direction argument passed to the (*RNND)Set(params) call used to initialize rnnDesc.
	Moreover:
	If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the numLayers argument passed to ((*RNND)Set(params).)
	If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the numLayers argument passed to (*RNND)Set(params).
The second dimension must match the batchSize parameter in xD.

The third dimension depends on whether RNN mode is CUDNN_LSTM and whether LSTM projection is enabled. Moreover:

If RNN mode is CUDNN_LSTM and LSTM projection is enabled, the third dimension must match the recProjSize argument passed to (*RNND)SetProjectionLayers(params) call used to set rnnDesc.
Otherwise, the third dimension must match the hiddenSize argument passed to the (*RNND)Set(params) call used to initialize rnnDesc.
dhy
Input. Data pointer to GPU memory associated with the tensor descriptor dhyD. If a NULL pointer is passed, the gradients at the final hidden state of the network will be initialized to zero.

dcyD - Input. A fully packed tensor descriptor describing the gradients at the final cell state of the RNN. The first dimension of the tensor depends on the direction argument passed to the (*RNND)Set(params) call used to initialize rnnDesc. Moreover:
	If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the numLayers argument passed to (*RNND)Set(params).
	If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the numLayers argument passed to (*RNND)Set(params).
	The second dimension must match the first dimension of the tensors described in xD.

The third dimension must match the hiddenSize argument passed to the (*RNND)Set(params) call used to initialize rnnDesc. The tensor must be fully packed.

dcy - Input. Data pointer to GPU memory associated with the tensor descriptor dcyD. If a NULL pointer is passed, the gradients at the final cell state of the network will be initialized to zero.

wD -Input. Handle to a previously initialized filter descriptor describing the weights for the RNN.

w -Input. Data pointer to GPU memory associated with the filter descriptor wD.

hxD -Input. A fully packed tensor descriptor describing the initial hidden state of the RNN. Must match or be the exact same descriptor previously passed into ForwardTrainingEx.

hx -Input. Data pointer to GPU memory associated with the tensor descriptor hxD. If a NULL pointer is passed, the initial hidden state of the network will be initialized to zero. Must contain the exact same data previously passed into ForwardTrainingEx, or be NULL if NULL was previously passed to ForwardTrainingEx.

cxD - Input. A fully packed tensor descriptor describing the initial cell state for LSTM networks. Must match or be the exact same descriptor previously passed into ForwardTrainingEx.

cx -Input. Data pointer to GPU memory associated with the tensor descriptor cxD. If a NULL pointer is passed, the initial cell state of the network will be initialized to zero. Must contain the exact same data previously passed into ForwardTrainingEx, or be NULL if NULL was previously passed to ForwardTrainingEx.

dxD - Input. A previously initialized RNN data descriptor. The dataType, layout, maxSeqLength, batchSize, vectorSize and seqLengthArray need to match that of xD previously passed to ForwardTrainingEx.

dx -Output. Data pointer to the GPU memory associated with the RNN data descriptor dxD. The vectors are expected to be laid out in memory according to the layout specified by dxD. The elements in the tensor (including elements in the padding vector) must be densely packed, and no strides are supported.

dhxD -Input. A fully packed tensor descriptor describing the gradient at the initial hidden state of the RNN. The descriptor must be set exactly the same way as dhyD.

dhx- Output. Data pointer to GPU memory associated with the tensor descriptor dhxD. If a NULL pointer is passed, the gradient at the hidden input of the network will not be set.

dcxD-Input. A fully packed tensor descriptor describing the gradient at the initial cell state of the RNN. The descriptor must be set exactly the same way as dcyD.

dcx -Output. Data pointer to GPU memory associated with the tensor descriptor dcxD. If a NULL pointer is passed, the gradient at the cell input of the network will not be set.



wspace  - Input. Data pointer to GPU memory to be used as a wspace for this call.
wspacesib - Input. Specifies the size in bytes of the provided wspace.

rspace - Input/Output. Data pointer to GPU memory to be used as a reserve space for this call.
rspacesib - Input. Specifies the size in bytes of the provided rspace.
*/
func (r *RNND) BackwardDataEx(h *Handle,
	yD *RNNDataD, y cutil.Mem,
	dyD *RNNDataD, dy cutil.Mem,
	dhyD *TensorD, dhy cutil.Mem,
	dcyD *TensorD, dcy cutil.Mem,
	wD *FilterD, w cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	cxD *TensorD, cx cutil.Mem,
	dxD *RNNDataD, dx cutil.Mem,
	dhxD *TensorD, dhx cutil.Mem,
	dcxD *TensorD, dcx cutil.Mem,
	wspace cutil.Mem, wspacesib uint,
	rspace cutil.Mem, rspacesib uint) error {
	if h.w != nil {
		return h.w.Work(func() error {
			return Status(C.cudnnRNNBackwardDataEx(h.x,
				r.descriptor,
				yD.d, y.Ptr(),
				dyD.d, dy.Ptr(),
				nil, nil,
				dhyD.descriptor, dhy.Ptr(),
				dcyD.descriptor, dcy.Ptr(),
				wD.descriptor, w.Ptr(),
				hxD.descriptor, hx.Ptr(),
				cxD.descriptor, cx.Ptr(),
				dxD.d, dx.Ptr(),
				dhxD.descriptor, dhx.Ptr(),
				dcxD.descriptor, dcx.Ptr(),
				nil,
				nil,
				wspace.Ptr(), C.size_t(wspacesib),
				rspace.Ptr(), C.size_t(rspacesib))).error("(r *RNND) BackwardDataEx")
		})
	}
	return Status(C.cudnnRNNBackwardDataEx(h.x,
		r.descriptor,
		yD.d, y.Ptr(),
		dyD.d, dy.Ptr(),
		nil, nil,
		dhyD.descriptor, dhy.Ptr(),
		dcyD.descriptor, dcy.Ptr(),
		wD.descriptor, w.Ptr(),
		hxD.descriptor, hx.Ptr(),
		cxD.descriptor, cx.Ptr(),
		dxD.d, dx.Ptr(),
		dhxD.descriptor, dhx.Ptr(),
		dcxD.descriptor, dcx.Ptr(),
		nil,
		nil,
		wspace.Ptr(), C.size_t(wspacesib),
		rspace.Ptr(), C.size_t(rspacesib))).error("(r *RNND) BackwardDataEx")

}

//BackwardDataExUS is like BackwardDataEx but uses unsafe.Pointer instead of cutil.Mem
func (r *RNND) BackwardDataExUS(h *Handle,
	yD *RNNDataD, y unsafe.Pointer,
	dyD *RNNDataD, dy unsafe.Pointer,
	dhyD *TensorD, dhy unsafe.Pointer,
	dcyD *TensorD, dcy unsafe.Pointer,
	wD *FilterD, w unsafe.Pointer,
	hxD *TensorD, hx unsafe.Pointer,
	cxD *TensorD, cx unsafe.Pointer,
	dxD *RNNDataD, dx unsafe.Pointer,
	dhxD *TensorD, dhx unsafe.Pointer,
	dcxD *TensorD, dcx unsafe.Pointer,
	wspace unsafe.Pointer, wspacesib uint,
	rspace unsafe.Pointer, rspacesib uint) error {
	if h.w != nil {
		return h.w.Work(func() error {
			return Status(C.cudnnRNNBackwardDataEx(h.x,
				r.descriptor,
				yD.d, y,
				dyD.d, dy,
				nil, nil,
				dhyD.descriptor, dhy,
				dcyD.descriptor, dcy,
				wD.descriptor, w,
				hxD.descriptor, hx,
				cxD.descriptor, cx,
				dxD.d, dx,
				dhxD.descriptor, dhx,
				dcxD.descriptor, dcx,
				nil,
				nil,
				wspace, C.size_t(wspacesib),
				rspace, C.size_t(rspacesib))).error("(r *RNND) BackwardDataExUS")
		})
	}
	return Status(C.cudnnRNNBackwardDataEx(h.x,
		r.descriptor,
		yD.d, y,
		dyD.d, dy,
		nil, nil,
		dhyD.descriptor, dhy,
		dcyD.descriptor, dcy,
		wD.descriptor, w,
		hxD.descriptor, hx,
		cxD.descriptor, cx,
		dxD.d, dx,
		dhxD.descriptor, dhx,
		dcxD.descriptor, dcx,
		nil,
		nil,
		wspace, C.size_t(wspacesib),
		rspace, C.size_t(rspacesib))).error("(r *RNND) BackwardDataExUS")

}

/*BackwardWeightsEx -from cudnn documentation
This routine is the extended version of the function cudnnRNNBackwardWeights.
This function cudnnRNNBackwardWeightsEx allows the user to use unpacked (padded) layout for input x and output dw.
In the unpacked layout, each sequence in the mini-batch is considered to be of fixed length,
specified by maxSeqLength in its corresponding RNNDataDescriptor. Each fixed-length sequence,
for example, the nth sequence in the mini-batch, is composed of a valid segment specified by t
he seqLengthArray[n] in its corresponding RNNDataDescriptor; and a padding segment to
make the combined sequence length equal to maxSeqLength.
With the unpacked layout, both sequence major (i.e. time major) and batch major are supported.
For backward compatibility, the packed sequence major layout is supported.
However, similar to the non-extended function cudnnRNNBackwardWeights, the sequences in the
mini-batch need to be sorted in descending order according to length.

Parameters:

handle - Input. Handle to a previously created cuDNN context.

xD - Input. A previously initialized RNN data descriptor. Must match or
			be the exact same descriptor previously passed into ForwardTrainingEx.

x - Input. Data pointer to GPU memory associated with the tensor descriptors
		   in the array xD. Must contain the exact same data previously passed into ForwardTrainingEx.

hxD - Input. A fully packed tensor descriptor describing the initial hidden state of the RNN.
			 Must match or be the exact same descriptor previously passed into ForwardTrainingEx.

hx - Input. Data pointer to GPU memory associated with the tensor descriptor hxD.
	 If a NULL pointer is passed, the initial hidden state of the network will be initialized to zero.
	 Must contain the exact same data previously passed into ForwardTrainingEx, or be NULL if NULL was previously passed to ForwardTrainingEx.

yD - Input. A previously initialized RNN data descriptor.
               Must match or be the exact same descriptor previously passed into ForwardTrainingEx.

y -Input. Data pointer to GPU memory associated with the output tensor descriptor yD.
		  Must contain the exact same data previously produced by ForwardTrainingEx.

wspace - Input. Data pointer to GPU memory to be used as a wspace for this call.

wspacesib - Input. Specifies the size in bytes of the provided wspace.

dwD- Input. Handle to a previously initialized filter descriptor describing the gradients of the weights for the RNN.

dw - Input/Output. Data pointer to GPU memory associated with the filter descriptor dwD.

rspace - Input. Data pointer to GPU memory to be used as a reserve space for this call.

rspacesib - Input. Specifies the size in bytes of the provided rspace
*/
func (r *RNND) BackwardWeightsEx(h *Handle,
	xD *RNNDataD, x cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	yD *RNNDataD, y cutil.Mem,
	wspace cutil.Mem, wspacesib uint,
	dwD *FilterD, dw cutil.Mem,
	rspace cutil.Mem, rspacesib uint,
) error {
	if h.w != nil {
		return h.w.Work(func() error {
			return Status(C.cudnnRNNBackwardWeightsEx(
				h.x,
				r.descriptor,
				xD.d, x.Ptr(),
				hxD.descriptor, hx.Ptr(),
				yD.d, y.Ptr(),
				wspace.Ptr(), C.size_t(wspacesib),
				dwD.descriptor, dw.Ptr(),
				rspace.Ptr(), C.size_t(rspacesib),
			)).error("(r *RNND) BackwardWeightsEx")
		})
	}
	return Status(C.cudnnRNNBackwardWeightsEx(
		h.x,
		r.descriptor,
		xD.d, x.Ptr(),
		hxD.descriptor, hx.Ptr(),
		yD.d, y.Ptr(),
		wspace.Ptr(), C.size_t(wspacesib),
		dwD.descriptor, dw.Ptr(),
		rspace.Ptr(), C.size_t(rspacesib),
	)).error("(r *RNND) BackwardWeightsEx")

}

//BackwardWeightsExUS is like BackwardWeightsEx but with unsafe.Pointer instead of cutil.Mem
func (r *RNND) BackwardWeightsExUS(h *Handle,
	xD *RNNDataD, x unsafe.Pointer,
	hxD *TensorD, hx unsafe.Pointer,
	yD *RNNDataD, y unsafe.Pointer,
	wspace unsafe.Pointer, wspacesib uint,
	dwD *FilterD, dw unsafe.Pointer,
	rspace unsafe.Pointer, rspacesib uint,
) error {
	if h.w != nil {
		return h.w.Work(func() error {
			return Status(C.cudnnRNNBackwardWeightsEx(
				h.x,
				r.descriptor,
				xD.d, x,
				hxD.descriptor, hx,
				yD.d, y,
				wspace, C.size_t(wspacesib),
				dwD.descriptor, dw,
				rspace, C.size_t(rspacesib),
			)).error("(r *RNND) BackwardWeightsExUS")
		})
	}
	return Status(C.cudnnRNNBackwardWeightsEx(
		h.x,
		r.descriptor,
		xD.d, x,
		hxD.descriptor, hx,
		yD.d, y,
		wspace, C.size_t(wspacesib),
		dwD.descriptor, dw,
		rspace, C.size_t(rspacesib),
	)).error("(r *RNND) BackwardWeightsExUS")

}
