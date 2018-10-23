package gocudnn

import (
	"errors"
	"fmt"
)

//Reshape is failing ---- changed all to private

//XTransposeD holds the kernel function
type XTransposeD struct {
	kern *Kernel
}

//CreateTransposeDesc creates a struct that holds the kernel for Transpos operation.  Might get rid of it
func (xt Xtra) CreateTransposeDesc(handle *XHandle) (*XTransposeD, error) {
	var cu Cuda
	kern, err := cu.MakeKernel("Transpose", handle.mod)
	return &XTransposeD{kern: kern}, err
}

//GetChannelTransposeOutputProperties  will output a transposed descriptor and the permutation array for a NCHW to NHWC and vice versa
func (t *XTransposeD) GetChannelTransposeOutputProperties(src *TensorD) (TensorFormat, DataType, []int32, []int32, error) {
	dtype, dims, _, err := src.GetDescrptor()
	if err != nil {
		return 255, 255, nil, nil, err
	}
	var dflag DataTypeFlag
	if dtype != dflag.Float() {
		return 255, 255, nil, nil, errors.New("Only Supported Format is float32")
	}
	frmt, err := src.GetFormat()
	if err != nil {
		return 255, 255, nil, nil, err
	}

	var frmtflg TensorFormatFlag

	if frmt == frmtflg.NCHW() {
		perm := []int32{0, 2, 3, 1}
		outdims := findnewdims(dims, perm)

		return frmtflg.NHWC(), dtype, outdims, perm, nil

	} else if frmt == frmtflg.NHWC() {
		perm := []int32{0, 3, 1, 2}
		outdims := findnewdims(dims, perm)

		return frmtflg.NCHW(), dtype, outdims, perm, nil
	}
	return 255, 255, nil, nil, errors.New("Unsupported Tensor Format")

}

func findnewdims(original, perm []int32) []int32 {
	newdims := make([]int32, len(original))
	for i := 0; i < len(original); i++ {
		newdims[i] = original[perm[i]]
	}
	return newdims
}

//Transpose will transpose the values of src to dest . only works on non sliding Tensors
func (t *XTransposeD) Transpose(handle *XHandle, perm []int32, srcdesc *TensorD, src Memer, destdesc *TensorD, dest Memer) error {

	xfrmt, err := srcdesc.GetFormat()
	if err != nil {
		return errors.New("TransPoseTensor doesn't support sliding Tensors")
	}
	yfrmt, err := destdesc.GetFormat()
	if err != nil {
		return errors.New("TransPoseTensor doesn't support sliding Tensors")
	}
	if xfrmt == yfrmt {
		return errors.New("TransPoseTensor not neccessary")
	}
	var frmt TensorFormatFlag
	if xfrmt == frmt.NCHWvectC() || yfrmt == frmt.NCHWvectC() {
		return errors.New("TransPoseTensor NCHWvectC not supported")
	}

	dtype, xdims, _, err := srcdesc.GetDescrptor()
	var dflag DataTypeFlag
	if dtype != dflag.Float() {
		return errors.New("Only Supported Format is float32")
	}
	if err != nil {
		return err
	}
	_, ydims, _, err := destdesc.GetDescrptor()
	if err != nil {
		return err
	}
	if findvol(xdims) != findvol(ydims) {
		return errors.New("Transpose-The tensor volumes x and y need to be the same N*C*W*H (any order)")
	}
	xstrides := stridecalc(xdims)
	ystrides := stridecalc(ydims)
	dimslegnth := int32(len(xdims))
	buffer := make([]int32, 3*dimslegnth)
	for i := int32(0); i < dimslegnth; i++ {
		buffer[i] = xstrides[i]
		buffer[i+dimslegnth] = ystrides[i]
		buffer[i+(2*dimslegnth)] = perm[i]
	}
	buffptr, err := MakeGoPointer(buffer)
	if err != nil {
		return err
	}
	var lflg LocationFlag

	if src.Stored() == lflg.Unified() {
		var devbuffer *Malloced
		devbuffer, err = UnifiedMangedGlobal(buffptr.ByteSize())
		if err != nil {
			return err
		}
		err = CudaMemCopy(devbuffer, buffptr, buffptr.ByteSize(), MemcpyKindFlag{}.Default())
		if err != nil {
			return err
		}
		handle.s.Sync()
		xsizeinbytes, err := srcdesc.GetSizeInBytes()
		if err != nil {
			devbuffer.Free()
			return err
		}
		elements := FindLength(xsizeinbytes, dtype)
		config := handle.LaunchConfig(int32(elements))

		err = t.kern.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, handle.s, config.Elements, src, devbuffer, len(xdims), dest)
		if err != nil {
			devbuffer.Free()
			return err
		}

		return devbuffer.Free()
	}
	var devbuffer *Malloced
	devbuffer, err = Malloc(buffptr.ByteSize())
	if err != nil {
		return err
	}
	err = CudaMemCopy(devbuffer, buffptr, buffptr.ByteSize(), MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		return err
	}

	xsizeinbytes, err := srcdesc.GetSizeInBytes()
	if err != nil {
		devbuffer.Free()
		return err
	}
	elements := FindLength(xsizeinbytes, dtype)
	config := handle.LaunchConfig(int32(elements))
	err = t.kern.Launch(config.BlockCount, 1, 1, config.ThreadPerBlock, 1, 1, 0, handle.s, config.Elements, src, devbuffer, len(xdims), dest)
	if err != nil {
		devbuffer.Free()
		return err
	}
	handle.s.Sync()

	return devbuffer.Free()
}

func stridecalc(dims []int32) []int32 {
	strides := make([]int32, len(dims))
	stride := int32(1)
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= dims[i]
	}
	return strides
}

//XResizeD is a struct that holds the reshape functions
type XResizeD struct {
	nearestfwdnhwc *Kernel
	nearestbwdnhwc *Kernel
	nearestfwdnchw *Kernel
	nearestbwdnchw *Kernel
	aligncorners   bool
}

//CreateResizeDesc creates a descriptor that holds the reshpaes
func (xt Xtra) CreateResizeDesc(handle *XHandle, aligncorners bool) (*XResizeD, error) {
	nearestfwdnhwc, err := Cuda{}.MakeKernel("nearestneighborNHWC", handle.mod)
	if err != nil {
		return nil, err
	}
	nearestbwdnhwc, err := Cuda{}.MakeKernel("nearestneighborNHWCBack", handle.mod)
	if err != nil {
		return nil, err
	}
	nearestfwdnchw, err := Cuda{}.MakeKernel("nearestneighborNCHW", handle.mod)
	if err != nil {
		return nil, err
	}
	nearestbwdnchw, err := Cuda{}.MakeKernel("nearestneighborNCHWBack", handle.mod)
	if err != nil {
		return nil, err
	}
	return &XResizeD{
		nearestfwdnhwc: nearestfwdnhwc,
		nearestbwdnhwc: nearestbwdnhwc,
		nearestfwdnchw: nearestfwdnchw,
		nearestbwdnchw: nearestbwdnchw,
		aligncorners:   aligncorners,
	}, nil
}

//ResizeForward does the reshape operation
func (s *XResizeD) ResizeForward(handle *XHandle, xdesc *TensorD, x Memer, ydesc *TensorD, y Memer) error {

	_, dimsx, _, err := xdesc.GetDescrptor()
	if err != nil {
		return err
	}
	_, dimsy, _, err := ydesc.GetDescrptor()
	if err != nil {
		return err
	}
	var fmtflag TensorFormatFlag
	frmtx, err := xdesc.GetFormat()
	if err != nil {
		return err
	}
	frmty, err := ydesc.GetFormat()
	if err != nil {
		return err
	}
	if frmtx != frmty {
		return errors.New("ResizeForward - tensors must match")
	}

	switch frmtx {
	case fmtflag.NHWC():
		if dimsx[0] != dimsy[0] || dimsx[3] != dimsy[3] {
			return errors.New("x and y dims n to n or c to c do not equal")
		}
		ratioh := float32(dimsx[1]) / float32(dimsy[1])
		ratiow := float32(dimsx[2]) / float32(dimsy[2])
		outputvol := findvol(dimsy)
		conf := handle.LaunchConfig(outputvol)
		var aligned int32
		if s.aligncorners == true {
			aligned = 1
		}
		return s.nearestfwdnhwc.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, x, dimsx[1], dimsx[2], dimsx[3], dimsy[1], dimsy[2], ratioh, ratiow, y)
	case fmtflag.NCHW():

		if dimsx[0] != dimsy[0] || dimsx[1] != dimsy[1] {
			return errors.New("x and y dims n to n or c to c do not equal")
		}
		ratioh := float32(dimsx[2]) / float32(dimsy[2])
		ratiow := float32(dimsx[3]) / float32(dimsy[3])
		outputvol := findvol(dimsy)
		conf := handle.LaunchConfig(outputvol)
		var aligned int32
		if s.aligncorners == true {
			aligned = 1
		}
		return s.nearestfwdnchw.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, x, dimsx[2], dimsx[3], dimsx[1], dimsy[2], dimsy[3], ratioh, ratiow, y)

	}
	return errors.New("Not Supported Tensor Format")
}

//ResizeBackward does a reshape backwards but it will add the errors on the backprop.
func (s *XResizeD) ResizeBackward(handle *XHandle, dxdesc *TensorD, dx Memer, dydesc *TensorD, dy Memer) error {
	_, dimsx, _, err := dxdesc.GetDescrptor()
	if err != nil {
		return err
	}
	_, dimsy, _, err := dydesc.GetDescrptor()
	if err != nil {
		return err
	}

	var fmtflag TensorFormatFlag
	frmtx, err := dxdesc.GetFormat()
	if err != nil {
		return err
	}
	frmty, err := dydesc.GetFormat()
	if err != nil {
		return err
	}
	if frmtx != frmty {
		return errors.New("ResizeForward - tensors must match")
	}

	switch frmtx {
	case fmtflag.NHWC():
		if dimsx[0] != dimsy[0] || dimsx[3] != dimsy[3] {
			return errors.New("dx and dy dims n to n or c to c do not equal")
		}
		ratioh := float32(dimsx[1]) / float32(dimsy[1])
		ratiow := float32(dimsx[2]) / float32(dimsy[2])
		inputvol := findvol(dimsx)
		conf := handle.LaunchConfig(inputvol)
		var aligned int32
		if s.aligncorners == true {
			aligned = 1
		}
		mx, ok := dx.(*Malloced)
		if ok {
			err = mx.Set(0)
			dx = mx
			if err != nil {
				return err

			}
		} else {
			return errors.New("Unsupported Memer")

		}

		return s.nearestbwdnhwc.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, dx, dimsx[1], dimsx[2], dimsx[3], dimsy[1], dimsy[2], ratioh, ratiow, dy)
	case fmtflag.NCHW():
		if dimsx[0] != dimsy[0] || dimsx[1] != dimsy[1] {
			return errors.New("x and y dims n to n or c to c do not equal")
		}
		ratioh := float32(dimsx[2]) / float32(dimsy[2])
		ratiow := float32(dimsx[3]) / float32(dimsy[3])
		outputvol := findvol(dimsy)
		conf := handle.LaunchConfig(outputvol)
		var aligned int32
		if s.aligncorners == true {
			aligned = 1
		}
		mx, ok := dx.(*Malloced)
		if ok {
			err = mx.Set(0)
			dx = mx
			if err != nil {
				return err

			}
		} else {
			return errors.New("Unsupported Memer")

		}

		return s.nearestbwdnchw.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, dx, dimsx[1], dimsx[2], dimsx[3], dimsy[1], dimsy[2], ratioh, ratiow, dy)
	}
	return errors.New("Not Supported Tensor Format")
}

//XShapetoBatchD holds the kernel function
type XShapetoBatchD struct {
	nhwc *Kernel
	nchw *Kernel
}

//CreateShapetoBatchDesc creates a shape to batch desc
func (xt Xtra) CreateShapetoBatchDesc(handle *XHandle) (*XShapetoBatchD, error) {
	nhwc, err := Cuda{}.MakeKernel("ShapetoBatch4DNHWC", handle.mod)
	if err != nil {
		return nil, err
	}
	nchw, err := Cuda{}.MakeKernel("ShapetoBatch4DNCHW", handle.mod)
	return &XShapetoBatchD{nhwc: nhwc, nchw: nchw}, err
}

//ShapeToBatch4d seperates chunks fo memory to blocks, so each window is the size of the block passed, and that those will becomoe the new batches.
//if S2B is true then it does the "Forward". Where the x values will be placed into the y tensor
//if S2B is false the y values will be placed into the x tensor. The C channel is the only thing that needs to be the same between tensor x and y.
//Any values that don't fit will get the zero value
//To get the y tensor please use FindShapetoBatchoutputTensor.
func (s *XShapetoBatchD) ShapeToBatch4d(handle *XHandle, xDesc *TensorD, x Memer, yDesc *TensorD, y Memer, S2B bool) error {

	dtype, xdims, _, err := xDesc.GetDescrptor()
	var dflag DataTypeFlag
	if dtype != dflag.Float() {
		return errors.New("Only Supported dtype is float32")
	}

	dtype, ydims, _, err := yDesc.GetDescrptor()
	if dtype != dflag.Float() {
		return errors.New("Only Supported dytype is float32")
	}
	var tflag TensorFormatFlag
	frmt, err := xDesc.GetFormat()
	if err != nil {
		return err
	}
	frmt2, err := yDesc.GetFormat()
	if err != nil {
		return err
	}
	if frmt != frmt2 {
		return errors.New("TensorFormats Must be the same")
	}
	switch frmt {
	case tflag.NHWC():
		n1 := divideandroundup(xdims[1], ydims[1])
		n2 := divideandroundup(xdims[2], ydims[2])
		if int32(n1*n2) != ydims[0] || ydims[3] != xdims[3] {
			return errors.New("N values or C values don't match up please use FindShapetoBatchoutputTensor to get TensorD")
		}
		oHH := xdims[1]
		oHW := xdims[2]
		OriginalBatches := xdims[0]
		OriginalVol := findvol(xdims[1:])
		BatchedVol := int32(n1) * int32(n2) * ydims[1] * ydims[2] * ydims[3]

		if err != nil {
			return err
		}

		cfg := handle.LaunchConfig3d(ydims[1], ydims[2], ydims[3])

		fmt.Println(cfg)
		zero := int32(0)
		for i := zero; i < OriginalBatches; i++ {
			err = s.nhwc.Launch(cfg.BlockCountx,
				cfg.BlockCounty,
				cfg.BlockCountz,
				cfg.ThreadPerBlockx,
				cfg.ThreadPerBlocky,
				cfg.ThreadPerBlockz,
				0, handle.s, cfg.Dimx, cfg.Dimy, cfg.Dimz, oHH, oHW, (i * BatchedVol), (i * OriginalVol), n1, n2, x, y, S2B)
			//(i * BatchedVol), (i * OriginalVol) are offset values if there are multiple batches that come from x
			if err != nil {
				return err
			}
		}

		return nil
	case tflag.NCHW():
		n1 := divideandroundup(xdims[2], ydims[2])
		n2 := divideandroundup(xdims[3], ydims[3])
		if int32(n1*n2) != ydims[0] || ydims[1] != xdims[1] {
			return errors.New("N values or C values don't match up please use FindShapetoBatchoutputTensor to get TensorD")
		}
		oHH := xdims[2]
		oHW := xdims[3]
		OriginalBatches := xdims[0]
		OriginalVol := findvol(xdims[1:])
		BatchedVol := int32(n1) * int32(n2) * ydims[1] * ydims[2] * ydims[3]

		if err != nil {
			return err
		}

		cfg := handle.LaunchConfig3d(ydims[1], ydims[2], ydims[3])

		fmt.Println(cfg)
		zero := int32(0)
		for i := zero; i < OriginalBatches; i++ {
			err = s.nchw.Launch(cfg.BlockCountx,
				cfg.BlockCounty,
				cfg.BlockCountz,
				cfg.ThreadPerBlockx,
				cfg.ThreadPerBlocky,
				cfg.ThreadPerBlockz,
				0, handle.s, cfg.Dimx, cfg.Dimy, cfg.Dimz, oHH, oHW, (i * BatchedVol), (i * OriginalVol), n1, n2, x, y, S2B)
			//(i * BatchedVol), (i * OriginalVol) are offset values if there are multiple batches that come from x
			if err != nil {
				return err
			}
		}

		return nil

	}
	return errors.New("Unsupported Tensor Format")
}

func copytogpumalloc(x *GoPointer) (*Malloced, error) {

	y, err := Malloc(x.ByteSize())
	if err != nil {
		return nil, err
	}
	err = CudaMemCopy(y, x, x.ByteSize(), MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		y.Free()
		return nil, err
	}
	return y, nil
}
func copytogpuunified(x *GoPointer) (*Malloced, error) {

	y, err := UnifiedMangedGlobal(x.ByteSize())
	if err != nil {
		return nil, err
	}
	err = CudaMemCopy(y, x, x.ByteSize(), MemcpyKindFlag{}.Default())
	if err != nil {
		y.Free()
		return nil, err
	}
	return y, nil
}

//GetShapetoBatchOutputProperties creates a tensordescriptor for the segmeented size
func (s *XShapetoBatchD) GetShapetoBatchOutputProperties(descX *TensorD, h, w int32) (TensorFormat, DataType, []int32, error) {
	dtype, dims, _, err := descX.GetDescrptor()
	if err != nil {
		return 255, 255, nil, err
	}
	var dflag DataTypeFlag
	if dtype != dflag.Float() {
		return 255, 255, nil, errors.New("Only Supported Format is float32")
	}
	var frmt TensorFormatFlag
	xfrmt, err := descX.GetFormat()

	switch xfrmt {
	case frmt.NCHW():

		n1 := int32(divideandroundup(dims[2], h))
		n2 := int32(divideandroundup(dims[3], w))

		return frmt.NCHW(), dtype, []int32{n1 * n2 * dims[0], dims[1], h, w}, nil

	case frmt.NHWC():
		n1 := int32(divideandroundup(dims[1], h))
		n2 := int32(divideandroundup(dims[2], w))
		return frmt.NHWC(), dtype, []int32{n1 * n2 * dims[0], h, w, dims[3]}, nil

	default:
		return 255, 255, nil, errors.New("NHWC-Vec Not supported")
	}

}

func findvol(dims []int32) int32 {
	mult := int32(1)
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	return mult
}
func divideandroundup(den, num int32) uint32 {

	return uint32(((den - 1) / num) + 1)

}
