package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

/*


Descriptors


*/

//DeConvolutionD sets all the convolution info
type DeConvolutionD struct {
	descriptor C.cudnnConvolutionDescriptor_t
	dims       C.int
	gogc       bool
}

//CreateDeConvolutionDescriptor creates a deconvolution descriptor
func CreateDeConvolutionDescriptor() (*DeConvolutionD, error) {
	d := new(DeConvolutionD)
	err := Status(C.cudnnCreateConvolutionDescriptor(&d.descriptor)).error("NewConvolution2dDescriptor-create")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		runtime.SetFinalizer(d, destroydeconvolutiondescriptor)
	}
	return d, nil
}

//Set sets the convolution descriptor
//Input.Type of the filter layout format. If this input is set to CUDNN_TENSOR_NCHW, which is one of the enumerated values allowed by cudnnTensorFormat_t descriptor, then the layout of the filter is as follows:
//
//	For N=4, i.e., for a 4D filter descriptor, the filter layout is in the form of KCRS (K represents the number of output feature maps, C the number of input feature maps, R the number of rows per filter, and S the number of columns per filter.)
//
//	For N=3, i.e., for a 3D filter descriptor, the number S (number of columns per filter) is omitted.
//
//	For N=5 and greater, the layout of the higher dimensions immediately follow RS.
//
//	On the other hand, if this input is set to CUDNN_TENSOR_NHWC, then the layout of the filter is as follows:
//
//	for N=4, i.e., for a 4D filter descriptor, the filter layout is in the form of KRSC.
//
//	For N=3, i.e., for a 3D filter descriptor, the number S (number of columns per filter) is omitted, and the layout of C immediately follows R.
//
//	For N=5 and greater, the layout of the higher dimensions are inserted between S and C. See also the description for cudnnTensorFormat_t.
//
//	Note:
//
//  Length of stride, pad, and dilation need to be len(tensordims) -2.
func (c *DeConvolutionD) Set(mode ConvolutionMode, data DataType, pad, stride, dilation []int32) error {
	cdata := data.c()
	cmode := mode.c()
	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdilation := int32Tocint(dilation)
	c.dims = C.int(len(pad))
	return Status(C.cudnnSetConvolutionNdDescriptor(c.descriptor, c.dims, &cpad[0], &cstride[0], &cdilation[0], cmode, cdata)).error("NewConvolutionNdDescriptor-set")

}

//Get gets returns the values used to make the convolution descriptor
func (c *DeConvolutionD) Get() (mode ConvolutionMode, data DataType, pad []int32, stride []int32, dilation []int32, err error) {
	if c.dims == 0 {
		c.dims = C.CUDNN_DIM_MAX
	}
	padding := make([]C.int, c.dims)
	striding := make([]C.int, c.dims)
	dilationing := make([]C.int, c.dims)
	var actual C.int
	var moded C.cudnnConvolutionMode_t
	var dtype C.cudnnDataType_t

	err = Status(C.cudnnGetConvolutionNdDescriptor(c.descriptor, c.dims, &actual, &padding[0], &striding[0], &dilationing[0], &moded, &dtype)).error("GetndDescriptor")
	c.dims = actual
	return ConvolutionMode(moded), DataType(dtype), cintToint32(padding[:actual]), cintToint32(striding[:actual]), cintToint32(dilationing[:actual]), err

}

//String satisfies fmt Stringer interface.
func (c *DeConvolutionD) String() string {
	cmode, dtype, pad, stride, dilation, err := c.Get()
	if err != nil {
		var errst = "error"
		return fmt.Sprintf(
			"DeConvolutionD Values\n"+
				"---------------------\n"+
				"ConvolutionMode: %s\n"+
				"DataType: %s\n"+
				"Pad: %v,\n"+
				"Stride: %v,\n"+
				"Dilation %v,\n", errst, errst, errst, errst, errst)

	}
	return fmt.Sprintf(
		"DeConvolutionD Values\n"+
			"---------------------\n"+
			"ConvolutionMode: %s\n"+
			"DataType: %s\n"+
			"Pad: %v,\n"+
			"Stride: %v,\n"+
			"Dilation %v,\n", cmode.String(), dtype.String(), pad, stride, dilation)

}

//SetGroupCount sets the Group Count
func (c *DeConvolutionD) SetGroupCount(groupCount int32) error {

	err := Status(C.cudnnSetConvolutionGroupCount(c.descriptor, C.int(groupCount))).error("SetGroupCountandMathtype-Group")

	return err

}

//SetReorderType sets the reorder type
func (c *DeConvolutionD) SetReorderType(r Reorder) error {
	return Status(C.cudnnSetConvolutionReorderType(c.descriptor, r.c())).error("SetReorderType")
}

//GetReorderType gets the reorder type
func (c *DeConvolutionD) GetReorderType() (r Reorder, err error) {
	err = Status(C.cudnnGetConvolutionReorderType(c.descriptor, r.cptr())).error("GetReorderType")
	return r, err
}

//SetMathType sets the mathtype
func (c *DeConvolutionD) SetMathType(mathtype MathType) error {

	x := Status(C.cudnnSetConvolutionMathType(c.descriptor, C.cudnnMathType_t(mathtype)))

	return x.error("SetGroupCountandMathtype-Math")
}

//GetOutputDims is a helper function to give the size of the output of of a COnvolutionNDForward
//Each dimension of the (nbDims-2)-D images of the output tensor is computed as followed:
//
//    outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride;
//
// DeConvolution works differently than a convolution.
//
//In a normal convolution, the output channel will be the number of neurons it has.  The channel size of the nuerons will be the input channel size.
//
//For a deconvolution.  The number of neurons will be the input channel size, and the neuron channel size will be the output channel size.
func (c *DeConvolutionD) GetOutputDims(input *TensorD, filter *FilterD) ([]int32, error) {
	frmt, _, dims, _, err := input.Get()
	if err != nil {
		return nil, err
	}
	_, ffrmt, fdims, err := filter.Get()
	if err != nil {
		return nil, err
	}
	_, _, pad, stride, dilation, err := c.Get()

	if err != nil {
		return nil, err
	}
	if frmt != ffrmt {
		return nil, errors.New(" (c *DeConvolutionD) GetOutputDims () input tensor format != filter tensor format")
	}
	flg := frmt
	switch frmt {
	case flg.NCHW():
		neurons := fdims[0]
		neuronchans := fdims[1]
		batch := dims[0]
		tensorchans := dims[1]
		if neurons != tensorchans {
			return nil, errors.New("(c *DeConvolutionD) GetOutputDims(input *TensorD, filter *FilterD): neurons != inputchannel size")
		}

		outputdims := make([]int32, len(fdims))
		outputdims[0] = batch
		outputdims[1] = neuronchans
		for i := 2; i < len(outputdims); i++ {
			outputdims[i] = deconvoutputdim(dims[i], fdims[i], pad[i-2], stride[i-2], dilation[i-2])
		}
		return outputdims, nil
	case flg.NHWC():
		neurons := fdims[0]
		neuronchans := fdims[len(fdims)-1]
		batch := dims[0]
		tensorchans := dims[len(dims)-1]
		if neurons != tensorchans {
			return nil, errors.New("(c *DeConvolutionD) GetOutputDims(input *TensorD, filter *FilterD): neurons != inputchannel size")
		}

		outputdims := make([]int32, len(fdims))
		outputdims[0] = batch
		outputdims[len(outputdims)-1] = neuronchans
		for i := 1; i < len(outputdims)-1; i++ {
			outputdims[i] = deconvoutputdim(dims[i], fdims[i], pad[i-1], stride[i-1], dilation[i-1])
		}
		return outputdims, nil
	default:
		return nil, errors.New("(c *DeConvolutionD) GetOutputDims(input *TensorD, filter *FilterD): Unsupported Format")
	}

}
func deconvoutputdim(x, f, p, s, d int32) int32 {
	//output for regular convolution
	//o = 1 + ( i + 2*p - (((f-1)*d)+1) )/s;
	//switch i and o and solve for o
	//i = 1 + ( o + 2*p - (((f-1)*d)+1) )/s
	//(i-1)*s -2*p + (((f-1)*d)+1)
	return ((x - 1) * s) - (2 * p) + (((f - 1) * d) + 1)
}

//Destroy destroys the ConvolutionDescriptor. If GC is set then it only returns nil.
//Currently GC is set with no option to turn off
func (c *DeConvolutionD) Destroy() error {
	if setfinalizer || c.gogc {
		return nil
	}
	return destroydeconvolutiondescriptor(c)
}
func destroydeconvolutiondescriptor(c *DeConvolutionD) error {
	return Status(C.cudnnDestroyConvolutionDescriptor(c.descriptor)).error("DestroyConvolutionDescriptor")
}

//GetForwardWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (c *DeConvolutionD) GetForwardWorkspaceSize(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	dxD *TensorD,
	algo ConvBwdDataAlgo) (uint, error) {
	var sizebytes C.size_t
	err := Status(C.cudnnGetConvolutionBackwardDataWorkspaceSize(
		handle.x,
		wD.descriptor,
		dyD.descriptor,
		c.descriptor,
		dxD.descriptor,
		algo.c(),
		&sizebytes)).error("GetConvolutionBackwardDataWorkspaceSize")

	return uint(sizebytes), err
}

//Forward does the forward deconvolution
//
//This function computes the convolution data gradient of the tensor dy,
//where y is the output of the forward convolution in (*ConvolutionD)Forward().
//It uses the specified algo, and returns the results in the output tensor dx.
//Scaling factors alpha and beta can be used to scale the computed result or accumulate with the current dx.
func (c *DeConvolutionD) Forward(
	handle *Handle,
	alpha float64,
	xD *TensorD, x cutil.Mem,
	wD *FilterD, w cutil.Mem,
	algo DeConvFwdAlgo,
	wspace cutil.Mem, wspaceSIB uint,
	beta float64,
	yD *TensorD, y cutil.Mem) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if wspace == nil {

		return Status(C.cudnnConvolutionBackwardData(
			handle.x,
			a.CPtr(),
			wD.descriptor,
			w.Ptr(),
			xD.descriptor,
			x.Ptr(),
			c.descriptor,
			algo.c(),
			nil,
			(C.size_t)(wspaceSIB),
			b.CPtr(),
			yD.descriptor,
			y.Ptr(),
		)).error("ConvolutionBackwardData")
	}

	return Status(C.cudnnConvolutionBackwardData(
		handle.x,
		a.CPtr(),
		wD.descriptor,
		w.Ptr(),
		xD.descriptor,
		x.Ptr(),
		c.descriptor,
		algo.c(),
		wspace.Ptr(),
		(C.size_t)(wspaceSIB),
		b.CPtr(),
		yD.descriptor,
		y.Ptr(),
	)).error("ConvolutionBackwardData")
}

//ForwardUS is like BackwardData but uses unsafe.Pointer instead of cutil.Mem
func (c *DeConvolutionD) ForwardUS(
	handle *Handle,
	alpha float64,
	xD *TensorD, x unsafe.Pointer,
	wD *FilterD, w unsafe.Pointer,
	algo DeConvFwdAlgo,
	wspace unsafe.Pointer, wspacesize uint,
	beta float64,
	yD *TensorD, y unsafe.Pointer) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)

	return Status(C.cudnnConvolutionBackwardData(
		handle.x,
		a.CPtr(),
		wD.descriptor, w,
		xD.descriptor, x,
		c.descriptor,
		algo.c(),
		wspace, (C.size_t)(wspacesize),
		b.CPtr(),
		yD.descriptor, y,
	)).error("ConvolutionBackwardData")
}

//BackwardBias is used to compute the bias gradient for batch convolution db is returned
func (c *DeConvolutionD) BackwardBias(
	handle *Handle,
	alpha float64,
	dyD *TensorD,
	dy cutil.Mem,
	beta float64,
	dbD *TensorD,
	db cutil.Mem) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	return Status(C.cudnnConvolutionBackwardBias(handle.x, a.CPtr(), dyD.descriptor, dy.Ptr(), b.CPtr(), dbD.descriptor, db.Ptr())).error("ConvolutionBackwardBias")
}

//BackwardBiasUS is like BackwardBias but using unsafe.Pointer instead of cutil.Mem
func (c *DeConvolutionD) BackwardBiasUS(
	handle *Handle,
	alpha float64,
	dyD *TensorD, dy unsafe.Pointer,
	beta float64,
	dbD *TensorD, db unsafe.Pointer) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	return Status(C.cudnnConvolutionBackwardBias(handle.x, a.CPtr(), dyD.descriptor, dy, b.CPtr(), dbD.descriptor, db)).error("ConvolutionBackwardBias")
}

//GetBackwardFilterWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (c *DeConvolutionD) GetBackwardFilterWorkspaceSize(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	dwD *FilterD,
	algo DeConvBwdFiltAlgo) (uint, error) {
	var sizebytes C.size_t
	err := Status(C.cudnnGetConvolutionBackwardFilterWorkspaceSize(
		handle.x,
		xD.descriptor,
		dyD.descriptor,
		c.descriptor,
		dwD.descriptor,
		algo.c(),
		&sizebytes)).error("GetConvolutionForwardWorkspaceSize")

	return uint(sizebytes), err
}

//BackwardFilter does the backwards convolution
func (c *DeConvolutionD) BackwardFilter(
	handle *Handle,
	alpha float64,
	xD *TensorD, x cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	algo DeConvBwdFiltAlgo,
	wspace cutil.Mem, wspacesize uint,
	beta float64,
	dwD *FilterD, dw cutil.Mem,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	if wspace == nil {

		return Status(C.cudnnConvolutionBackwardFilter(
			handle.x,
			a.CPtr(),
			xD.descriptor,
			x.Ptr(),
			dyD.descriptor,
			dy.Ptr(),
			c.descriptor,
			algo.c(),
			nil,
			C.size_t(wspacesize),
			b.CPtr(),
			dwD.descriptor,
			dw.Ptr(),
		)).error("cudnnConvolutionBackwardFilter")

	}

	return Status(C.cudnnConvolutionBackwardFilter(
		handle.x,
		a.CPtr(),
		xD.descriptor,
		x.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		c.descriptor,
		algo.c(),
		wspace.Ptr(),
		C.size_t(wspacesize),
		b.CPtr(),
		dwD.descriptor,
		dw.Ptr(),
	)).error("cudnnConvolutionBackwardFilter")
}

//BackwardFilterUS is like BackwardFilter but using unsafe.Pointer instead of cutil.Mem
func (c *DeConvolutionD) BackwardFilterUS(
	handle *Handle,
	alpha float64,
	xD *TensorD, x unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	algo DeConvBwdFiltAlgo,
	wspace unsafe.Pointer, wspacesize uint,
	beta float64,
	dwD *FilterD, dw unsafe.Pointer,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)

	return Status(C.cudnnConvolutionBackwardFilter(
		handle.x,
		a.CPtr(),
		xD.descriptor, x,
		dyD.descriptor, dy,
		c.descriptor,
		algo.c(),
		wspace, C.size_t(wspacesize),
		b.CPtr(),
		dwD.descriptor, dw,
	)).error("cudnnConvolutionBackwardFilter")
}

//GetBackwardWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (c *DeConvolutionD) GetBackwardWorkspaceSize(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	dxD *TensorD,
	algo DeConvBwdDataAlgo) (uint, error) {
	var sizebytes C.size_t
	err := Status(C.cudnnGetConvolutionForwardWorkspaceSize(handle.x, dyD.descriptor, wD.descriptor, c.descriptor, dxD.descriptor, algo.c(), &sizebytes)).error("GetConvolutionForwardWorkspaceSize")

	return uint(sizebytes), err
}

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//BackwardData Function to perform the backward pass pass for batch convolution
func (c *DeConvolutionD) BackwardData(
	handle *Handle,
	alpha float64,
	wD *FilterD, w cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	algo DeConvBwdDataAlgo,
	wspace cutil.Mem, wspaceSIB uint,
	beta float64,
	dxD *TensorD, dx cutil.Mem) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	var err error
	if wspace == nil {

		err = Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), dyD.descriptor, dy.Ptr(), wD.descriptor, w.Ptr(),
			c.descriptor, algo.c(), nil, C.size_t(wspaceSIB), b.CPtr(), dxD.descriptor, dx.Ptr())).error(" (c *DeConvolutionD) BackwardData")

	} else {
		err = Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), dyD.descriptor, dy.Ptr(), wD.descriptor, w.Ptr(),
			c.descriptor, algo.c(), wspace.Ptr(), C.size_t(wspaceSIB), b.CPtr(), dxD.descriptor, dx.Ptr())).error(" (c *DeConvolutionD) BackwardData")
	}
	if cudnndebugmode {
		if err != nil {

			fmt.Println("\nError for ConvForward\n", "alpha: ", a, "\nbeta: ", b, "\nxD: ", dyD, "\nx :", dy, "\nwD :", wD, "\nw: ", w, "\nwspace: ", wspace, "\nwspacesize: ", wspaceSIB, "\nyD: ", dxD, "\ny: ", dx)
			fdt, ftf, fdim, err1 := wD.Get()
			fmt.Println("wD vals", fdt, ftf, fdim, err1)

			cmode, cdtype, pad, stride, dilation, err1 := c.Get()
			fmt.Println("Pad Settings", cmode, cdtype, pad, stride, dilation, err1)
			fmt.Println("Algo Settings", algo)
			actualwspacesize, err := c.GetBackwardWorkspaceSize(handle, wD, dyD, dxD, algo)

			fmt.Println("Workspace Size Compare passed/wanted:", wspaceSIB, actualwspacesize, err)
			panic(err)
		}
	}

	return err
}

//BackwardDataUS is like BackwardData but using unsafe.Pointer instead of cutil.Mem
func (c *DeConvolutionD) BackwardDataUS(
	handle *Handle,
	alpha float64,
	wD *FilterD, w unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	algo DeConvBwdDataAlgo,
	wspace unsafe.Pointer, wspacesize uint,
	beta float64,
	dxD *TensorD, dx unsafe.Pointer) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)

	err := Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), dyD.descriptor, dy, wD.descriptor, w,
		c.descriptor, algo.c(), wspace, C.size_t(wspacesize), b.CPtr(), dxD.descriptor, dx)).error(" (c *DeConvolutionD) BackwardDataUS")

	return err
}

/*

Flags


*/

/*
*
*
*       ConvolutionMode
*
*
 */

/*
*
*
*       ConvBwdDataPrefFlag
*
*
 */

//DeConvBwdDataPref used for flags on bwddatapref exposing them through methods
type DeConvBwdDataPref C.cudnnConvolutionFwdPreference_t

//NoWorkSpace sets c to returns ConvBwdDataPref( C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE) and returns value of c
func (c *DeConvBwdDataPref) NoWorkSpace() DeConvBwdDataPref {
	*c = DeConvBwdDataPref(C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE)
	return *c
}

//PreferFastest  sets c to ConvBwdDataPref( C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST) and returns value of c
func (c *DeConvBwdDataPref) PreferFastest() DeConvBwdDataPref {
	*c = DeConvBwdDataPref(C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
	return *c
}

//SpecifyWorkSpaceLimit  sets c to ConvBwdDataPref( C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)and returns value of c
func (c *DeConvBwdDataPref) SpecifyWorkSpaceLimit() DeConvBwdDataPref {
	*c = DeConvBwdDataPref(C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
	return *c
}

func (c DeConvBwdDataPref) c() C.cudnnConvolutionFwdPreference_t {
	return C.cudnnConvolutionFwdPreference_t(c)
}

//DeConvBwdDataAlgo flags for cudnnConvFwdAlgo_t  exposing them through methods.
//Deconvolution uses the forward pass for backward data
type DeConvBwdDataAlgo C.cudnnConvolutionFwdAlgo_t

//ImplicitGemm sets c to DeConvBwdDataAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) and returns value of c
func (c *DeConvBwdDataAlgo) ImplicitGemm() DeConvBwdDataAlgo {
	*c = DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
	return *c
}

//ImplicitPrecompGemm sets c to DeConvBwdDataAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) and returns value of c
func (c *DeConvBwdDataAlgo) ImplicitPrecompGemm() DeConvBwdDataAlgo {
	*c = DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
	return *c
}

//Gemm sets c to DeConvBwdDataAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM) and returns value of c
func (c *DeConvBwdDataAlgo) Gemm() DeConvBwdDataAlgo {
	*c = DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
	return *c
}

//Direct sets c to DeConvBwdDataAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT) and returns value of c
func (c *DeConvBwdDataAlgo) Direct() DeConvBwdDataAlgo {
	*c = DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
	return *c
}

//FFT sets c to DeConvBwdDataAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_FFT) and returns value of c
func (c *DeConvBwdDataAlgo) FFT() DeConvBwdDataAlgo {
	*c = DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT)
	return *c
}

//FFTTiling sets c to DeConvBwdDataAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) and returns value of c
func (c *DeConvBwdDataAlgo) FFTTiling() DeConvBwdDataAlgo {
	*c = DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
	return *c
}

//WinoGrad sets c to DeConvBwdDataAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD) and returns value of c
func (c *DeConvBwdDataAlgo) WinoGrad() DeConvBwdDataAlgo {
	*c = DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
	return *c
}

//WinoGradNonFused sets c to DeConvBwdDataAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED) and returns value of c
func (c *DeConvBwdDataAlgo) WinoGradNonFused() DeConvBwdDataAlgo {
	*c = DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
	return *c
}

//Count sets c to DeConvBwdDataAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT) and returns value of c
func (c *DeConvBwdDataAlgo) Count() DeConvBwdDataAlgo {
	*c = DeConvBwdDataAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
	return *c
}

func (c DeConvBwdDataAlgo) c() C.cudnnConvolutionFwdAlgo_t {
	return C.cudnnConvolutionFwdAlgo_t(c)
}

/*
*
*
*       DeConvFwdAlgo Flag
*
*
 */

//DeConvolutionForwardPref used for flags on deconvolution forward exposing them through methods
type DeConvolutionForwardPref C.cudnnConvolutionBwdDataPreference_t

//NoWorkSpace sets c to returns DeConvolutionForwardPref( C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE) and returns value of c
func (c *DeConvolutionForwardPref) NoWorkSpace() DeConvolutionForwardPref {
	*c = DeConvolutionForwardPref(C.CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE)
	return *c
}

//PreferFastest  sets c to DeConvolutionForwardPref( C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST) and returns value of c
func (c *DeConvolutionForwardPref) PreferFastest() DeConvolutionForwardPref {
	*c = DeConvolutionForwardPref(C.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST)
	return *c
}

//SpecifyWorkSpaceLimit  sets c to ConvBwdDataPref( C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)and returns value of c
func (c *DeConvolutionForwardPref) SpecifyWorkSpaceLimit() DeConvolutionForwardPref {
	*c = DeConvolutionForwardPref(C.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT)
	return *c
}

func (c DeConvolutionForwardPref) c() C.cudnnConvolutionBwdDataPreference_t {
	return C.cudnnConvolutionBwdDataPreference_t(c)
}

//DeConvFwdAlgo used for flags in the forward data algorithms  exposing them through methods
//DeConvolution does the Backward Data pass as its forward.
type DeConvFwdAlgo C.cudnnConvolutionBwdDataAlgo_t

//Algo0  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)  and returns value of c /* non-deterministic */
func (c *DeConvFwdAlgo) Algo0() DeConvFwdAlgo {
	*c = DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
	return *c
}

//Algo1  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)  and returns value of c
func (c *DeConvFwdAlgo) Algo1() DeConvFwdAlgo {
	*c = DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)
	return *c
}

//FFT  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)  and returns value of c
func (c *DeConvFwdAlgo) FFT() DeConvFwdAlgo {
	*c = DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)
	return *c
}

//FFTTiling  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)  and returns value of c
func (c *DeConvFwdAlgo) FFTTiling() DeConvFwdAlgo {
	*c = DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)
	return *c
}

//Winograd 	 sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)  and returns value of c
func (c *DeConvFwdAlgo) Winograd() DeConvFwdAlgo {
	*c = DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
	return *c
}

//WinogradNonFused  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)  and returns value of c
func (c *DeConvFwdAlgo) WinogradNonFused() DeConvFwdAlgo {
	*c = DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)
	return *c
}

//Count  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)  and returns value of c
func (c *DeConvFwdAlgo) Count() DeConvFwdAlgo {
	*c = DeConvFwdAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)
	return *c
}
func (c DeConvFwdAlgo) c() C.cudnnConvolutionBwdDataAlgo_t {
	return C.cudnnConvolutionBwdDataAlgo_t(c)
}

/*
*
*
*       DeConvBwdFilter Flags
*
*
 */

//DeConvBwdFilterPref are used for flags for the backwds filters  exposing them through methods
type DeConvBwdFilterPref C.cudnnConvolutionBwdFilterPreference_t

//NoWorkSpace sets c to  DeConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)  and returns value of c
func (c *DeConvBwdFilterPref) NoWorkSpace() DeConvBwdFilterPref {
	*c = DeConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
	return *c
}

//PreferFastest sets c to  DeConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)  and returns value of c
func (c *DeConvBwdFilterPref) PreferFastest() DeConvBwdFilterPref {
	*c = DeConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
	return *c
}

//SpecifyWorkSpaceLimit sets c to  DeConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)  and returns value of c
func (c *DeConvBwdFilterPref) SpecifyWorkSpaceLimit() DeConvBwdFilterPref {
	*c = DeConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
	return *c
}

func (c DeConvBwdFilterPref) c() C.cudnnConvolutionBwdFilterPreference_t {
	return C.cudnnConvolutionBwdFilterPreference_t(c)
}

/*
*
*
*       DeConvBwdFiltAlgo Flag
*
*
 */

//DeConvBwdFiltAlgo Used for ConvBwdFiltAlgo flags  exposing them through methods
type DeConvBwdFiltAlgo C.cudnnConvolutionBwdFilterAlgo_t

//Algo0 sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0) and returns value of c /* non-deterministic */
func (c *DeConvBwdFiltAlgo) Algo0() DeConvBwdFiltAlgo {
	*c = DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)
	return *c
}

//Algo1 sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1) and returns value of c
func (c *DeConvBwdFiltAlgo) Algo1() DeConvBwdFiltAlgo {
	*c = DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
	return *c
}

//FFT sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT) and returns value of c
func (c *DeConvBwdFiltAlgo) FFT() DeConvBwdFiltAlgo {
	*c = DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT)
	return *c
}

//Algo3 sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3) and returns value of c
func (c *DeConvBwdFiltAlgo) Algo3() DeConvBwdFiltAlgo {
	*c = DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)
	return *c
}

//Winograd 	sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD) and returns value of c
func (c *DeConvBwdFiltAlgo) Winograd() DeConvBwdFiltAlgo {
	*c = DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD)
	return *c
}

//WinogradNonFused sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED) and returns value of c
func (c *DeConvBwdFiltAlgo) WinogradNonFused() DeConvBwdFiltAlgo {
	*c = DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED)
	return *c
}

//FFTTiling sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING) and returns value of c
func (c *DeConvBwdFiltAlgo) FFTTiling() DeConvBwdFiltAlgo {
	*c = DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING)
	return *c
}

//Count sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT) and returns value of c
func (c *DeConvBwdFiltAlgo) Count() DeConvBwdFiltAlgo {
	*c = DeConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)
	return *c
}
func (c DeConvBwdFiltAlgo) c() C.cudnnConvolutionBwdFilterAlgo_t {
	return C.cudnnConvolutionBwdFilterAlgo_t(c)
}

/*
*
*
*       ConvolutionFwdPrefFlag
*
*
 */
