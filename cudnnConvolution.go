package gocudnn

/*
#include <cudnn.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/dereklstinson/cutil"
)

/*


Descriptors


*/

//ConvolutionD sets all the convolution info
type ConvolutionD struct {
	descriptor C.cudnnConvolutionDescriptor_t
	dims       C.int
	gogc       bool
}

const convolutionnd2dtestflag = true

func (c *ConvolutionD) String() string {
	cmode, dtype, pad, stride, dilation, err := c.Get()
	if err != nil {

		return fmt.Sprintf("ConvolutionD{\nError\n}\n")
	}
	return fmt.Sprintf(
		"ConvolutionD{\n%s\n%s\nPad: %v\nStride: %v\nDilation: %v\nAddress: %p\n}\n", cmode.String(), dtype.String(), pad, stride, dilation, c)

}

//CreateConvolutionDescriptor creates a convolution descriptor
func CreateConvolutionDescriptor() (*ConvolutionD, error) {
	d := new(ConvolutionD)
	err := Status(C.cudnnCreateConvolutionDescriptor(&d.descriptor)).error("NewConvolution2dDescriptor-create")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		runtime.SetFinalizer(d, destroyconvolutiondescriptor)
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
func (c *ConvolutionD) Set(mode ConvolutionMode, data DataType, pad, stride, dilation []int32) error {
	cdata := data.c()
	cmode := mode.c()
	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdilation := int32Tocint(dilation)
	c.dims = C.int(len(pad))
	return Status(C.cudnnSetConvolutionNdDescriptor(c.descriptor, c.dims, &cpad[0], &cstride[0], &cdilation[0], cmode, cdata)).error("NewConvolutionNdDescriptor-set")

}

//Get gets returns the values used to make the convolution descriptor
func (c *ConvolutionD) Get() (mode ConvolutionMode, data DataType, pad []int32, stride []int32, dilation []int32, err error) {
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

//SetGroupCount sets the Group Count
func (c *ConvolutionD) SetGroupCount(groupCount int32) error {

	err := Status(C.cudnnSetConvolutionGroupCount(c.descriptor, C.int(groupCount))).error("SetGroupCountandMathtype-Group")

	return err

}

//SetReorderType sets the reorder type
func (c *ConvolutionD) SetReorderType(r Reorder) error {
	return Status(C.cudnnSetConvolutionReorderType(c.descriptor, r.c())).error("SetReorderType")
}

//GetReorderType gets the reorder type
func (c *ConvolutionD) GetReorderType() (r Reorder, err error) {
	err = Status(C.cudnnGetConvolutionReorderType(c.descriptor, r.cptr())).error("GetReorderType")
	return r, err
}

//SetMathType sets the mathtype
func (c *ConvolutionD) SetMathType(mathtype MathType) error {

	x := Status(C.cudnnSetConvolutionMathType(c.descriptor, C.cudnnMathType_t(mathtype)))

	return x.error("SetGroupCountandMathtype-Math")
}

//GetOutputDims is a helper function to give the size of the output of of a COnvolutionNDForward
//Each dimension of the (nbDims-2)-D images of the output tensor is computed as followed:
//
//    outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride;
//
//	Note if input and filter are NHWC.  cudnn would take the formats as NCHW and output an NCHW
//  gocudnn will take that NCHW and format it to an actual NHWC.
func (c *ConvolutionD) GetOutputDims(input *TensorD, filter *FilterD) ([]int32, error) {
	cdims := make([]C.int, int32(input.dims))

	err := Status(C.cudnnGetConvolutionNdForwardOutputDim(c.descriptor, input.descriptor, filter.descriptor, input.dims, &cdims[0])).error("GetConvolutionNdForwardOutputDim")
	if err != nil {
		return nil, err
	}
	fflg := input.frmt
	dims := cintToint32(cdims)
	switch input.frmt {
	case fflg.NHWC():
		dims = compatabilityNHWCdimsCudnntoGocudnn(dims)
	}
	return dims, err

}

//Destroy destroys the ConvolutionDescriptor. If GC is set then it only returns nil.
//Currently GC is set with no option to turn off
func (c *ConvolutionD) Destroy() error {
	if setfinalizer || c.gogc {
		return nil
	}
	return destroyconvolutiondescriptor(c)
}
func destroyconvolutiondescriptor(c *ConvolutionD) error {
	return Status(C.cudnnDestroyConvolutionDescriptor(c.descriptor)).error("DestroyConvolutionDescriptor")
}

//GetBackwardDataWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (c *ConvolutionD) GetBackwardDataWorkspaceSize(
	handle *Handle,
	wD *FilterD,
	dyD *TensorD,
	dxD *TensorD,
	algo ConvBwdDataAlgo) (uint, error) {
	var sizebytes C.size_t
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetConvolutionBackwardDataWorkspaceSize(
				handle.x,
				wD.descriptor,
				dyD.descriptor,
				c.descriptor,
				dxD.descriptor,
				algo.c(),
				&sizebytes)).error("(c *ConvolutionD) GetBackwardDataWorkspaceSize")
		})
	} else {
		err = Status(C.cudnnGetConvolutionBackwardDataWorkspaceSize(
			handle.x,
			wD.descriptor,
			dyD.descriptor,
			c.descriptor,
			dxD.descriptor,
			algo.c(),
			&sizebytes)).error("(c *ConvolutionD) GetBackwardDataWorkspaceSize")
	}

	return uint(sizebytes), err
}

//BackwardData does the backwards convolution on data
//
//This function computes the convolution data gradient of the tensor dy,
//where y is the output of the forward convolution in (*ConvolutionD)Forward().
//It uses the specified algo, and returns the results in the output tensor dx.
//Scaling factors alpha and beta can be used to scale the computed result or accumulate with the current dx.
//
//Parameters:
//
//	---
//	handle(input):
//
//	previously created Handle
//	---
//	----
//	alpha, beta(input):
//
//	Pointers to scaling factors (in host memory) used to blend the computation result with prior
//	value in the output layer as follows: dstValue = alpha[0]*result + beta[0]*priorDstValue.
//	----
//	---
//	wD(input):
//
//	For previously set input tensor descriptor.
//	---
//	----
//	w(input):
//
//	Data pointer to GPU memory associated with the tensor descriptor xD.
//
//	----
//	---
//	dyD(input):
//
//	For previously set input tensor descriptor of dy.
//	---
//	----
//	dy(input):
//
//	Data pointer to GPU memory associated with the input tensor desctiptor.(Holds back propigation errors)
//	----
//	---
//	algo(input):
//
//	Enumerant that specifies which backward data convolution algorithm shoud be used to compute the results.
//	---
//	----
//	wspace, wspaceSIB(inputs):
//
//	Data pointer and size in bytes of workspace needed for algo passed. If no wspace is need nil can be passed.
//	----
//	---
//	dxD(input):
//	For previously set output tensor descriptor of dx.
//	---
//	----
//	dx(input/output):
//	Data pointer to GPU memory associated with the output tensor desctiptor.(Holds back propigation errors for layer it received its forward inputs.)
//	----
//
//Supported Configurations
//	----
//	Config: "TRUE_HALF_CONFIG (only compute capability 5.3 and later)."
//	TensorD (wD,dyD,dxD): (*DataType)Half()
//	ConvolutionD: (*DataType)Half()
//	----
//	---
//	Config: "PSEUDO_HALF_CONFIG"
//	TensorD (wD,dyD,dxD): (*DataType)Half()
//	ConvolutionD: (*DataType)Float()
//	---
//	----
//	Config: "FLOAT_CONFIG"
//	TensorD (wD,dyD,dxD): (*DataType)Float()
//	ConvolutionD: (*DataType)Float()
//	----
//	---
//	Config: "DOUBLE_CONFIG"
//	TensorD (wD,dyD,dxD): (*DataType)Double()
//	ConvolutionD: (*DataType)Double()
//	---
//
//Note:
//Specifying a separate algorithm can cause changes in performance, support and computation determinism.
//
//Table of algorithm with configs can be found at.  (gocudnn flag names are similar to cudnn)
//	https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
//
//Possible Error Returns:
//	nil:
//
//	The function launched successfully.
//
//	CUDNN_STATUS_NOT_SUPPORTED:
//
//	At least one of the following conditions are met:
//	1)	dyD or dxD have negative tensor striding
//	2)	dyD, wD or dxD has a number of dimensions that is not 4 or 5
//	3)	The chosen algo does not support the parameters provided; see above for exhaustive list of parameter support for each algo
//	4)	dyD or wD indicate an output channel count that isn't a multiple of group count (if group count has been set in ConvolutionD).
//
//	CUDNN_STATUS_BAD_PARAM:
//
//	At least one of the following conditions are met:
//	1)	At least one of the following is NULL: handle, dyD, wD, ConvolutionD, dxD, dy, w, dx, alpha, beta
//	2)	wD and dyD have a non-matching number of dimensions
//	3)	wD and dxD have a non-matching number of dimensions
//	4)	wD has fewer than three number of dimensions
//	5)	wD, dxD and dyD have a non-matching data type.
//	6)	wD and dxD have a non-matching number of input feature maps per image (or group in case of Grouped Convolutions).
//	7)	dyD's spatial sizes do not match with the expected size as determined by (*ConvolutionD)GetOutputDims().
//
//	CUDNN_STATUS_MAPPING_ERROR:
//
//	An error occurs during the texture binding of the filter data or the input differential tensor data
//
//	CUDNN_STATUS_EXECUTION_FAILED:
//
//	The function failed to launch on the GPU.
//
func (c *ConvolutionD) BackwardData(
	handle *Handle,
	alpha float64,
	wD *FilterD, w cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	algo ConvBwdDataAlgo,
	wspace cutil.Mem, wspaceSIB uint,
	beta float64,
	dxD *TensorD, dx cutil.Mem,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			if wspace == nil {

				return Status(C.cudnnConvolutionBackwardData(
					handle.x,
					a.CPtr(),
					wD.descriptor,
					w.Ptr(),
					dyD.descriptor,
					dy.Ptr(),
					c.descriptor,
					algo.c(),
					nil,
					(C.size_t)(wspaceSIB),
					b.CPtr(),
					dxD.descriptor,
					dx.Ptr(),
				)).error("(c *ConvolutionD) BackwardData")
			}

			return Status(C.cudnnConvolutionBackwardData(
				handle.x,
				a.CPtr(),
				wD.descriptor,
				w.Ptr(),
				dyD.descriptor,
				dy.Ptr(),
				c.descriptor,
				algo.c(),
				wspace.Ptr(),
				(C.size_t)(wspaceSIB),
				b.CPtr(),
				dxD.descriptor,
				dx.Ptr(),
			)).error("(c *ConvolutionD) BackwardData")
		})
	}
	if wspace == nil {

		return Status(C.cudnnConvolutionBackwardData(
			handle.x,
			a.CPtr(),
			wD.descriptor,
			w.Ptr(),
			dyD.descriptor,
			dy.Ptr(),
			c.descriptor,
			algo.c(),
			nil,
			(C.size_t)(wspaceSIB),
			b.CPtr(),
			dxD.descriptor,
			dx.Ptr(),
		)).error("(c *ConvolutionD) BackwardData")
	}

	return Status(C.cudnnConvolutionBackwardData(
		handle.x,
		a.CPtr(),
		wD.descriptor,
		w.Ptr(),
		dyD.descriptor,
		dy.Ptr(),
		c.descriptor,
		algo.c(),
		wspace.Ptr(),
		(C.size_t)(wspaceSIB),
		b.CPtr(),
		dxD.descriptor,
		dx.Ptr(),
	)).error("(c *ConvolutionD) BackwardData")
}

//BackwardDataUS is like BackwardData but uses unsafe.Pointer instead of cutil.Mem
func (c *ConvolutionD) BackwardDataUS(
	handle *Handle,
	alpha float64,
	wD *FilterD, w unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	algo ConvBwdDataAlgo,
	wspace unsafe.Pointer, wspacesize uint,
	beta float64,
	dxD *TensorD, dx unsafe.Pointer,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnConvolutionBackwardData(
				handle.x,
				a.CPtr(),
				wD.descriptor, w,
				dyD.descriptor, dy,
				c.descriptor,
				algo.c(),
				wspace, (C.size_t)(wspacesize),
				b.CPtr(),
				dxD.descriptor, dx,
			)).error("(c *ConvolutionD) BackwardDataUS")
		})
	}
	return Status(C.cudnnConvolutionBackwardData(
		handle.x,
		a.CPtr(),
		wD.descriptor, w,
		dyD.descriptor, dy,
		c.descriptor,
		algo.c(),
		wspace, (C.size_t)(wspacesize),
		b.CPtr(),
		dxD.descriptor, dx,
	)).error("(c *ConvolutionD) BackwardDataUS")
}

//Im2Col transformes the multiDim tensors into 2d tensors for speed up in calculation at the cost of memory.
func (c *ConvolutionD) Im2Col(
	handle *Handle,
	xD *TensorD,
	x cutil.Mem,
	wD *FilterD,
	buffer cutil.Mem,
) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnIm2Col(
				handle.x,
				xD.descriptor,
				x.Ptr(),
				wD.descriptor,
				c.descriptor,
				buffer.Ptr(),
			)).error("(c *ConvolutionD) Im2Col")
		})
	}
	return Status(C.cudnnIm2Col(
		handle.x,
		xD.descriptor,
		x.Ptr(),
		wD.descriptor,
		c.descriptor,
		buffer.Ptr(),
	)).error("(c *ConvolutionD) Im2Col")
}

//Im2ColUS is like IN2Col but using unsafe.Pointer instead of cutil.Mem
func (c *ConvolutionD) Im2ColUS(
	handle *Handle,
	xD *TensorD, x unsafe.Pointer,
	wD *FilterD,
	buffer unsafe.Pointer,
) error {
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnIm2Col(
				handle.x,
				xD.descriptor, x,
				wD.descriptor,
				c.descriptor,
				buffer,
			)).error("(c *ConvolutionD) Im2ColUS")
		})
	}
	return Status(C.cudnnIm2Col(
		handle.x,
		xD.descriptor, x,
		wD.descriptor,
		c.descriptor,
		buffer,
	)).error("(c *ConvolutionD) Im2ColUS")
}

//BackwardBias is used to compute the bias gradient for batch convolution db is returned
func (c *ConvolutionD) BackwardBias(
	handle *Handle,
	alpha float64,
	dyD *TensorD,
	dy cutil.Mem,
	beta float64,
	dbD *TensorD,
	db cutil.Mem) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnConvolutionBackwardBias(handle.x, a.CPtr(), dyD.descriptor, dy.Ptr(), b.CPtr(), dbD.descriptor, db.Ptr())).error("(c *ConvolutionD) BackwardBias")
		})
	}
	return Status(C.cudnnConvolutionBackwardBias(handle.x, a.CPtr(), dyD.descriptor, dy.Ptr(), b.CPtr(), dbD.descriptor, db.Ptr())).error("(c *ConvolutionD) BackwardBias")
}

//BackwardBiasUS is like BackwardBias but using unsafe.Pointer instead of cutil.Mem
func (c *ConvolutionD) BackwardBiasUS(
	handle *Handle,
	alpha float64,
	dyD *TensorD, dy unsafe.Pointer,
	beta float64,
	dbD *TensorD, db unsafe.Pointer) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnConvolutionBackwardBias(handle.x, a.CPtr(), dyD.descriptor, dy, b.CPtr(), dbD.descriptor, db)).error(" (c *ConvolutionD) BackwardBiasUS")
		})
	}
	return Status(C.cudnnConvolutionBackwardBias(handle.x, a.CPtr(), dyD.descriptor, dy, b.CPtr(), dbD.descriptor, db)).error(" (c *ConvolutionD) BackwardBiasUS")
}

//GetBackwardFilterWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (c *ConvolutionD) GetBackwardFilterWorkspaceSize(
	handle *Handle,
	xD *TensorD,
	dyD *TensorD,
	dwD *FilterD,
	algo ConvBwdFiltAlgo) (uint, error) {
	var err error
	var sizebytes C.size_t
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetConvolutionBackwardFilterWorkspaceSize(
				handle.x,
				xD.descriptor,
				dyD.descriptor,
				c.descriptor,
				dwD.descriptor,
				algo.c(),
				&sizebytes)).error("(c *ConvolutionD) GetBackwardFilterWorkspaceSize")
		})
	} else {
		err = Status(C.cudnnGetConvolutionBackwardFilterWorkspaceSize(
			handle.x,
			xD.descriptor,
			dyD.descriptor,
			c.descriptor,
			dwD.descriptor,
			algo.c(),
			&sizebytes)).error("(c *ConvolutionD) GetBackwardFilterWorkspaceSize")
	}

	return uint(sizebytes), err
}

//BackwardFilter does the backwards convolution
func (c *ConvolutionD) BackwardFilter(
	handle *Handle,
	alpha float64,
	xD *TensorD, x cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	algo ConvBwdFiltAlgo,
	wspace cutil.Mem, wspacesize uint,
	beta float64,
	dwD *FilterD, dw cutil.Mem,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			if wspace == nil {
				if cudnndebugmode {
					fmt.Println("wspace is nil")
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
					nil,
					C.size_t(wspacesize),
					b.CPtr(),
					dwD.descriptor,
					dw.Ptr(),
				)).error("(c *ConvolutionD) BackwardFilter")

			}
			if cudnndebugmode {
				fmt.Println("is not nil")
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
			)).error("(c *ConvolutionD) BackwardFilter")

		})
	} else {
		if wspace == nil {
			if cudnndebugmode {
				fmt.Println("wspace is nil")
			}
			err = Status(C.cudnnConvolutionBackwardFilter(
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
			)).error("(c *ConvolutionD) BackwardFilter")

		} else {
			if cudnndebugmode {
				fmt.Println("is not nil")
			}
			err = Status(C.cudnnConvolutionBackwardFilter(
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
			)).error("(c *ConvolutionD) BackwardFilter")
		}

	}
	if cudnndebugmode {
		if err != nil {
			fmt.Println("checking the addresses")
			fmt.Println("handle.x", handle.x)
			fmt.Println("a.CPtr()", a.CPtr())
			fmt.Println("xD.descriptor", xD.descriptor)
			fmt.Println("x.Ptr()", x.Ptr())
			fmt.Println("dyD.descriptor", dyD.descriptor)
			fmt.Println("dy.Ptr()", dy.Ptr())
			fmt.Println("c.descriptor", c.descriptor)
			fmt.Println("algo.c()", algo.c())
			if wspace == nil {
				fmt.Println("wspace", nil)
			} else {
				fmt.Println("wspace.Ptr()", wspace.Ptr())
			}
			fmt.Println("wspacesize", wspacesize)
			fmt.Println("b.Cptr()", b.CPtr())
			fmt.Println("dwD.descriptor", dwD.descriptor)
			fmt.Println("dw.Ptr()", dw.Ptr())
			//going to check the output
			fmt.Printf("\nAlgo: %v", algo)
			fmt.Printf("\n%v,\nxD: %v,\ndyD: \n%v,\ndwD: %v", c, xD, dyD, dwD)
			fmt.Println(c.GetOutputDims(xD, dwD))
		}

	}
	return err
}

//BackwardFilterUS is like BackwardFilter but using unsafe.Pointer instead of cutil.Mem
func (c *ConvolutionD) BackwardFilterUS(
	handle *Handle,
	alpha float64,
	xD *TensorD, x unsafe.Pointer,
	dyD *TensorD, dy unsafe.Pointer,
	algo ConvBwdFiltAlgo,
	wspace unsafe.Pointer, wspacesize uint,
	beta float64,
	dwD *FilterD, dw unsafe.Pointer,
) error {
	a := cscalarbydatatype(dyD.dtype, alpha)
	b := cscalarbydatatype(dyD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
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
			)).error("(c *ConvolutionD) BackwardFilterUS")
		})
	}
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
	)).error("(c *ConvolutionD) BackwardFilterUS")
}

//GetForwardWorkspaceSize is a helper function that will return the minimum Size of the workspace to be passed by the convolution given an algo.
func (c *ConvolutionD) GetForwardWorkspaceSize(
	handle *Handle,
	xD *TensorD,
	wD *FilterD,
	yD *TensorD,
	algo ConvFwdAlgo) (uint, error) {
	var sizebytes C.size_t
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			return Status(C.cudnnGetConvolutionForwardWorkspaceSize(handle.x, xD.descriptor, wD.descriptor, c.descriptor, yD.descriptor, algo.c(), &sizebytes)).error("(c *ConvolutionD) GetForwardWorkspaceSize")
		})

	} else {
		err = Status(C.cudnnGetConvolutionForwardWorkspaceSize(handle.x, xD.descriptor, wD.descriptor, c.descriptor, yD.descriptor, algo.c(), &sizebytes)).error("(c *ConvolutionD) GetForwardWorkspaceSize")
	}
	return uint(sizebytes), err
}

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

//Forward Function to perform the forward pass for batch convolution
func (c *ConvolutionD) Forward(
	handle *Handle,
	alpha float64,
	xD *TensorD, x cutil.Mem,
	wD *FilterD, w cutil.Mem,
	algo ConvFwdAlgo,
	wspace cutil.Mem, wspacesize uint,
	beta float64,
	yD *TensorD, y cutil.Mem) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	var err error
	if handle.w != nil {
		err = handle.w.Work(func() error {
			if wspace == nil {

				return Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(),
					c.descriptor, algo.c(), nil, C.size_t(wspacesize), b.CPtr(), yD.descriptor, y.Ptr())).error("(c *ConvolutionD) Forward")
			}

			return Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(),
				c.descriptor, algo.c(), wspace.Ptr(), C.size_t(wspacesize), b.CPtr(), yD.descriptor, y.Ptr())).error("(c *ConvolutionD) Forward")
		})
	} else {
		if wspace == nil {

			return Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(),
				c.descriptor, algo.c(), nil, C.size_t(wspacesize), b.CPtr(), yD.descriptor, y.Ptr())).error("(c *ConvolutionD) Forward")
		}

		return Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), xD.descriptor, x.Ptr(), wD.descriptor, w.Ptr(),
			c.descriptor, algo.c(), wspace.Ptr(), C.size_t(wspacesize), b.CPtr(), yD.descriptor, y.Ptr())).error("(c *ConvolutionD) Forward")
	}

	if cudnndebugmode {
		if err != nil {
			fmt.Println("\nError for ConvForward\n", "alpha: ", a, "\nbeta: ", b, "\nxD: ", xD, "\nx :", x, "\nwD :", wD, "\nw: ", w, "\nwspace: ", wspace, "\nwspacesize: ", wspacesize, "\nyD: ", yD, "\ny: ", y)

			fmt.Printf("\n%v\n", wD)
			fmt.Printf("\n%v\n", c)
			fmt.Printf("\n%v", algo)
			actualwspacesize, err := c.GetForwardWorkspaceSize(handle, xD, wD, yD, algo)

			fmt.Println("Workspace Size Compare passed/wanted:", wspacesize, actualwspacesize, err)
			panic(err)
		}
	}

	return err
}

//ForwardUS is like Forward but using unsafe.Pointer instead of cutil.Mem
func (c *ConvolutionD) ForwardUS(
	handle *Handle,
	alpha float64,
	xD *TensorD, x unsafe.Pointer,
	wD *FilterD, w unsafe.Pointer,
	algo ConvFwdAlgo,
	wspace unsafe.Pointer, wspacesize uint,
	beta float64,
	yD *TensorD, y unsafe.Pointer) error {
	a := cscalarbydatatype(yD.dtype, alpha)
	b := cscalarbydatatype(yD.dtype, beta)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), xD.descriptor, x, wD.descriptor, w,
				c.descriptor, algo.c(), wspace, C.size_t(wspacesize), b.CPtr(), yD.descriptor, y)).error("(c *ConvolutionD) ForwardUS")
		})
	}
	return Status(C.cudnnConvolutionForward(handle.x, a.CPtr(), xD.descriptor, x, wD.descriptor, w,
		c.descriptor, algo.c(), wspace, C.size_t(wspacesize), b.CPtr(), yD.descriptor, y)).error("(c *ConvolutionD) ForwardUS")

}

//BiasActivationForward info can be found at:
//
//https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBiasActivationForward
//
//Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias )
func (c *ConvolutionD) BiasActivationForward(
	handle *Handle,
	alpha1 float64,
	xD *TensorD, x cutil.Mem,
	wD *FilterD, w cutil.Mem,
	algo ConvFwdAlgo,
	wspace cutil.Mem,
	wspacesize uint,
	alpha2 float64,
	zD *TensorD, z cutil.Mem,
	biasD *TensorD, bias cutil.Mem,
	aD *ActivationD,
	yD *TensorD, y cutil.Mem,
) error {
	a1 := cscalarbydatatype(yD.dtype, alpha1)
	a2 := cscalarbydatatype(yD.dtype, alpha2)
	if handle.w != nil {
		return handle.w.Work(func() error {
			if wspace == nil {

				return Status(
					C.cudnnConvolutionBiasActivationForward(
						handle.x,
						a1.CPtr(),
						xD.descriptor,
						x.Ptr(),
						wD.descriptor,
						w.Ptr(),
						c.descriptor,
						algo.c(),
						nil,
						C.size_t(wspacesize),
						a2.CPtr(),
						zD.descriptor,
						z.Ptr(),
						biasD.descriptor,
						bias.Ptr(),
						aD.descriptor,
						yD.descriptor,
						y.Ptr(),
					)).error("(c *ConvolutionD) BiasActivationForward")
			}

			return Status(
				C.cudnnConvolutionBiasActivationForward(
					handle.x,
					a1.CPtr(),
					xD.descriptor,
					x.Ptr(),
					wD.descriptor,
					w.Ptr(),
					c.descriptor,
					algo.c(),
					wspace.Ptr(),
					C.size_t(wspacesize),
					a2.CPtr(),
					zD.descriptor,
					z.Ptr(),
					biasD.descriptor,
					bias.Ptr(),
					aD.descriptor,
					yD.descriptor,
					y.Ptr(),
				)).error("(c *ConvolutionD) BiasActivationForward")
		})
	}
	if wspace == nil {

		return Status(
			C.cudnnConvolutionBiasActivationForward(
				handle.x,
				a1.CPtr(),
				xD.descriptor,
				x.Ptr(),
				wD.descriptor,
				w.Ptr(),
				c.descriptor,
				algo.c(),
				nil,
				C.size_t(wspacesize),
				a2.CPtr(),
				zD.descriptor,
				z.Ptr(),
				biasD.descriptor,
				bias.Ptr(),
				aD.descriptor,
				yD.descriptor,
				y.Ptr(),
			)).error("(c *ConvolutionD) BiasActivationForward")
	}

	return Status(
		C.cudnnConvolutionBiasActivationForward(
			handle.x,
			a1.CPtr(),
			xD.descriptor,
			x.Ptr(),
			wD.descriptor,
			w.Ptr(),
			c.descriptor,
			algo.c(),
			wspace.Ptr(),
			C.size_t(wspacesize),
			a2.CPtr(),
			zD.descriptor,
			z.Ptr(),
			biasD.descriptor,
			bias.Ptr(),
			aD.descriptor,
			yD.descriptor,
			y.Ptr(),
		)).error("(c *ConvolutionD) BiasActivationForward")
}

//BiasActivationForwardUS is like BiasActivationForward but using unsafe.Pointer instead of cutil.Mem
func (c *ConvolutionD) BiasActivationForwardUS(
	handle *Handle,
	alpha1 float64,
	xD *TensorD, x unsafe.Pointer,
	wD *FilterD, w unsafe.Pointer,
	algo ConvFwdAlgo,
	wspace unsafe.Pointer, wspacesize uint,
	alpha2 float64,
	zD *TensorD, z unsafe.Pointer,
	biasD *TensorD, bias unsafe.Pointer,
	aD *ActivationD,
	yD *TensorD, y unsafe.Pointer,
) error {
	a1 := cscalarbydatatype(yD.dtype, alpha1)
	a2 := cscalarbydatatype(yD.dtype, alpha2)
	if handle.w != nil {
		return handle.w.Work(func() error {
			return Status(
				C.cudnnConvolutionBiasActivationForward(
					handle.x,
					a1.CPtr(),
					xD.descriptor, x,
					wD.descriptor, w,
					c.descriptor,
					algo.c(),
					wspace, C.size_t(wspacesize),
					a2.CPtr(),
					zD.descriptor, z,
					biasD.descriptor, bias,
					aD.descriptor,
					yD.descriptor, y,
				)).error("(c *ConvolutionD) BiasActivationForwardUS")
		})
	}
	return Status(
		C.cudnnConvolutionBiasActivationForward(
			handle.x,
			a1.CPtr(),
			xD.descriptor, x,
			wD.descriptor, w,
			c.descriptor,
			algo.c(),
			wspace, C.size_t(wspacesize),
			a2.CPtr(),
			zD.descriptor, z,
			biasD.descriptor, bias,
			aD.descriptor,
			yD.descriptor, y,
		)).error("(c *ConvolutionD) BiasActivationForwardUS")
}

/*

Flags


*/
/*
//ConvolutionFlags is used to store the different Convolution Flags.  Hopefully it can make it easier when
//Using something like VSCode.
type ConvolutionFlags struct {
	Mode ConvolutionMode
	Bwd  ConvolutionBwdFlags
	Fwd  ConvolutionFwdFlags
}

//ConvolutionFwdFlags holds the different flags used for the convlution fwd
type ConvolutionFwdFlags struct {
	Pref ConvolutionForwardPref
	Algo ConvFwdAlgo
}

//ConvolutionBwdFlags holds the different type of BwdConvolutionFlags
type ConvolutionBwdFlags struct {
	DataPref ConvBwdDataPref
	DataAlgo ConvBwdDataAlgo
	FltrPref ConvBwdFilterPref
	FltrAlgo ConvBwdFiltAlgo
}
*/
/*
*
*
*       ConvolutionMode
*
*
 */

//ConvolutionMode is the type to describe the convolution mode flags
type ConvolutionMode C.cudnnConvolutionMode_t

//Convolution sets and returns value of c to ConvolutionMode(C.CUDNN_CONVOLUTION)
func (c *ConvolutionMode) Convolution() ConvolutionMode {
	*c = ConvolutionMode(C.CUDNN_CONVOLUTION)
	return *c
}

// CrossCorrelation n sets and returns value of c to  ConvolutionMode(C.CUDNN_CROSS_CORRELATION)
func (c *ConvolutionMode) CrossCorrelation() ConvolutionMode {
	*c = ConvolutionMode(C.CUDNN_CROSS_CORRELATION)
	return *c
}
func (c ConvolutionMode) String() string {
	var x string
	cflg := c
	switch c {
	case cflg.CrossCorrelation():
		x = "CrossCorrelation"
	case cflg.Convolution():
		x = "Convolution"
	default:
		x = "Unsupported Flag"
	}
	return "ConvolutionMode: " + x
}

func (c ConvolutionMode) c() C.cudnnConvolutionMode_t { return C.cudnnConvolutionMode_t(c) }

/*
*
*
*       ConvBwdDataPrefFlag
*
*
 */

//ConvBwdDataPref used for flags on bwddatapref exposing them through methods
type ConvBwdDataPref C.cudnnConvolutionBwdDataPreference_t

//NoWorkSpace sets c to returns ConvBwdDataPref( C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE) and returns value of c
func (c *ConvBwdDataPref) NoWorkSpace() ConvBwdDataPref {
	*c = ConvBwdDataPref(C.CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE)
	return *c
}

//PreferFastest  sets c to ConvBwdDataPref( C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST) and returns value of c
func (c *ConvBwdDataPref) PreferFastest() ConvBwdDataPref {
	*c = ConvBwdDataPref(C.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST)
	return *c
}

//SpecifyWorkSpaceLimit  sets c to ConvBwdDataPref( C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)and returns value of c
func (c *ConvBwdDataPref) SpecifyWorkSpaceLimit() ConvBwdDataPref {
	*c = ConvBwdDataPref(C.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT)
	return *c
}

func (c ConvBwdDataPref) c() C.cudnnConvolutionBwdDataPreference_t {
	return C.cudnnConvolutionBwdDataPreference_t(c)
}

/*
*
*
*       ConvBwdDataAlgoFlag
*
*
 */

//ConvBwdDataAlgo used for flags in the bacward data algorithms  exposing them through methods
type ConvBwdDataAlgo C.cudnnConvolutionBwdDataAlgo_t

//Algo0  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)  and returns value of c /* non-deterministic */
func (c *ConvBwdDataAlgo) Algo0() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
	return *c
}

//Algo1  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)  and returns value of c
func (c *ConvBwdDataAlgo) Algo1() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)
	return *c
}

//FFT  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)  and returns value of c
func (c *ConvBwdDataAlgo) FFT() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)
	return *c
}

//FFTTiling  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)  and returns value of c
func (c *ConvBwdDataAlgo) FFTTiling() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)
	return *c
}

//Winograd 	 sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)  and returns value of c
func (c *ConvBwdDataAlgo) Winograd() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
	return *c
}

//WinogradNonFused  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)  and returns value of c
func (c *ConvBwdDataAlgo) WinogradNonFused() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)
	return *c
}

//Count  sets c to  ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)  and returns value of c
func (c *ConvBwdDataAlgo) Count() ConvBwdDataAlgo {
	*c = ConvBwdDataAlgo(C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)
	return *c
}
func (c ConvBwdDataAlgo) c() C.cudnnConvolutionBwdDataAlgo_t {
	return C.cudnnConvolutionBwdDataAlgo_t(c)
}

/*
*
*
*       ConvBwdFilterPrefFlag
*
*
 */

//ConvBwdFilterPref are used for flags for the backwds filters  exposing them through methods
type ConvBwdFilterPref C.cudnnConvolutionBwdFilterPreference_t

//NoWorkSpace sets c to  ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)  and returns value of c
func (c *ConvBwdFilterPref) NoWorkSpace() ConvBwdFilterPref {
	*c = ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
	return *c
}

//PreferFastest sets c to  ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)  and returns value of c
func (c *ConvBwdFilterPref) PreferFastest() ConvBwdFilterPref {
	*c = ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
	return *c
}

//SpecifyWorkSpaceLimit sets c to  ConvBwdFilterPref( C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)  and returns value of c
func (c *ConvBwdFilterPref) SpecifyWorkSpaceLimit() ConvBwdFilterPref {
	*c = ConvBwdFilterPref(C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
	return *c
}

func (c ConvBwdFilterPref) c() C.cudnnConvolutionBwdFilterPreference_t {
	return C.cudnnConvolutionBwdFilterPreference_t(c)
}

/*
*
*
*       ConvBwdFiltAlgoFlag
*
*
 */

//ConvBwdFiltAlgo Used for ConvBwdFiltAlgo flags  exposing them through methods
type ConvBwdFiltAlgo C.cudnnConvolutionBwdFilterAlgo_t

//Algo0 sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0) and returns value of c /* non-deterministic */
func (c *ConvBwdFiltAlgo) Algo0() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)
	return *c
}

//Algo1 sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1) and returns value of c
func (c *ConvBwdFiltAlgo) Algo1() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
	return *c
}

//FFT sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT) and returns value of c
func (c *ConvBwdFiltAlgo) FFT() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT)
	return *c
}

//Algo3 sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3) and returns value of c
func (c *ConvBwdFiltAlgo) Algo3() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)
	return *c
}

//Winograd 	sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD) and returns value of c
func (c *ConvBwdFiltAlgo) Winograd() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD)
	return *c
}

//WinogradNonFused sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED) and returns value of c
func (c *ConvBwdFiltAlgo) WinogradNonFused() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED)
	return *c
}

//FFTTiling sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING) and returns value of c
func (c *ConvBwdFiltAlgo) FFTTiling() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING)
	return *c
}

//Count sets c to  ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT) and returns value of c
func (c *ConvBwdFiltAlgo) Count() ConvBwdFiltAlgo {
	*c = ConvBwdFiltAlgo(C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)
	return *c
}
func (c ConvBwdFiltAlgo) c() C.cudnnConvolutionBwdFilterAlgo_t {
	return C.cudnnConvolutionBwdFilterAlgo_t(c)
}

/*
*
*
*       ConvolutionFwdPrefFlag
*
*
 */

// ConvolutionForwardPref used for flags  exposing them through methods
type ConvolutionForwardPref C.cudnnConvolutionFwdPreference_t

func (c ConvolutionForwardPref) c() C.cudnnConvolutionFwdPreference_t {
	return C.cudnnConvolutionFwdPreference_t(c)
}

//NoWorkSpace sets c to ConvolutionForwardPref( C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE) and returns value of c
func (c *ConvolutionForwardPref) NoWorkSpace() ConvolutionForwardPref {
	*c = ConvolutionForwardPref(C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE)
	return *c
}

//PreferFastest returns ConvolutionForwardPref( C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
func (c *ConvolutionForwardPref) PreferFastest() ConvolutionForwardPref {
	*c = ConvolutionForwardPref(C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
	return *c
}

//SpecifyWorkSpaceLimit returns ConvolutionForwardPref( C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
func (c *ConvolutionForwardPref) SpecifyWorkSpaceLimit() ConvolutionForwardPref {
	*c = ConvolutionForwardPref(C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
	return *c
}

/*
*
*
*       ConvFwdAlgoFlag
*
*
 */

//ConvFwdAlgo flags for cudnnConvFwdAlgo_t  exposing them through methods
type ConvFwdAlgo C.cudnnConvolutionFwdAlgo_t

//ImplicitGemm sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) and returns value of c
func (c *ConvFwdAlgo) ImplicitGemm() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
	return *c
}

//ImplicitPrecompGemm sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) and returns value of c
func (c *ConvFwdAlgo) ImplicitPrecompGemm() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
	return *c
}

//Gemm sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM) and returns value of c
func (c *ConvFwdAlgo) Gemm() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
	return *c
}

//Direct sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT) and returns value of c
func (c *ConvFwdAlgo) Direct() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
	return *c
}

//FFT sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_FFT) and returns value of c
func (c *ConvFwdAlgo) FFT() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT)
	return *c
}

//FFTTiling sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) and returns value of c
func (c *ConvFwdAlgo) FFTTiling() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
	return *c
}

//WinoGrad sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD) and returns value of c
func (c *ConvFwdAlgo) WinoGrad() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
	return *c
}

//WinoGradNonFused sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED) and returns value of c
func (c *ConvFwdAlgo) WinoGradNonFused() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
	return *c
}

//Count sets c to ConvFwdAlgo( C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT) and returns value of c
func (c *ConvFwdAlgo) Count() ConvFwdAlgo {
	*c = ConvFwdAlgo(C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
	return *c
}

func (c ConvFwdAlgo) c() C.cudnnConvolutionFwdAlgo_t {
	return C.cudnnConvolutionFwdAlgo_t(c)
}

//Reorder is a flag that is changed through its methods
type Reorder C.cudnnReorderType_t

//Default Sets Reorder for inference
func (r *Reorder) Default() Reorder {
	*r = (Reorder)(C.CUDNN_DEFAULT_REORDER)
	return *r
}

//NoReorder changes the flag to noreorder
func (r *Reorder) NoReorder() Reorder {
	*r = (Reorder)(C.CUDNN_NO_REORDER)
	return *r
}
func (r Reorder) c() C.cudnnReorderType_t {
	return C.cudnnReorderType_t(r)
}
func (r *Reorder) cptr() *C.cudnnReorderType_t {
	return (*C.cudnnReorderType_t)(r)
}
func (r Reorder) String() string {
	var x string
	f := r
	switch r {
	case f.Default():
		x = "Default"
	case f.NoReorder():
		x = "NoReorder"
	default:
		x = "Unsupported Flag"

	}
	return "Reorder: " + x
}

/*
//Set2D sets convolution descriptor to 2D
func (c *ConvolutionD) Set2D(mode ConvolutionMode, data DataType, pad, stride, dialation []int32) error {
	if len(pad) != len(stride) || len(pad) != len(dialation) || len(pad) != 2 {
		return errors.New("pad stride and dialation need to be size 2")
	}
	cdata := data.c()
	cmode := mode.c()
	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdialation := int32Tocint(dialation)
	c.isconv2d = true
	if convolutionnd2dtestflag {
		dims := C.int(len(pad))
		return Status(C.cudnnSetConvolutionNdDescriptor(c.descriptor, dims, &cpad[0], &cstride[0], &cdialation[0], cmode, cdata)).error("NewConvolutionNdDescriptor-Set2D")

	}
	return Status(C.cudnnSetConvolution2dDescriptor(c.descriptor, cpad[0], cpad[1], cstride[0], cstride[1], cdialation[0], cdialation[1], cmode, cdata)).error("NewConvolution2dDescriptor-set")
}

//SetND sets the convolution descriptor to ND
func (c *ConvolutionD) SetND(mode ConvolutionMode, data DataType, pad, stride, dialation []int32) error {
	cdata := data.c()
	cmode := mode.c()
	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdialation := int32Tocint(dialation)
	dims := C.int(len(pad))
	return Status(C.cudnnSetConvolutionNdDescriptor(c.descriptor, dims, &cpad[0], &cstride[0], &cdialation[0], cmode, cdata)).error("NewConvolutionNdDescriptor-set")

}
*/
/*
//GetConvolution2dForwardOutputDim is a helper func that will output the shape of the convolution
func (c *ConvolutionD) GetConvolution2dForwardOutputDim(input *TensorD, filter *FilterD) ([]int32, error) {
	var shape [4]C.int
	x := Status(C.cudnnGetConvolution2dForwardOutputDim(c.descriptor, input.descriptor, filter.descriptor,
		&shape[0], &shape[1], &shape[2], &shape[3]))
	retshap := cintToint32(shape[:4])

	return retshap, x.error("GetConvolution2dForwardOutputDim")

}
*/
