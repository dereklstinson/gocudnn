package xtra

import (
	"errors"

	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cuda"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/cutil"
)

//Reshape is failing ---- changed all to private

//XTransposeD holds the kernel function
type XTransposeD struct {
	kern *cuda.Kernel
}

/*
//CreateTransposeDesc creates a struct that holds the kernel for Transpos operation.  Might get rid of it
func CreateTransposeDesc(handle *Handle) (*XTransposeD, error) {

	kern, err := cuda.MakeKernel("Transpose", handle.mod)
	return &XTransposeD{kern: kern}, err
}

//GetChannelTransposeOutputProperties  will output a transposed descriptor and the permutation array for a NCHW to NHWC and vice versa
func (t *XTransposeD) GetChannelTransposeOutputProperties(src *gocudnn.TensorD) (gocudnn.TensorFormat, gocudnn.DataType, []int32, []int32, error) {
	dtype, dims, _, err := src.Get()
	if err != nil {
		return 255, 255, nil, nil, err
	}
	var dflag gocudnn.DataTypeFlag
	if dtype != dflag.Float() {
		return 255, 255, nil, nil, errors.New("Only Supported Format is float32")
	}
	frmt, err := src.GetFormat()
	if err != nil {
		return 255, 255, nil, nil, err
	}

	var frmtflg gocudnn.TensorFormatFlag

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



//Transpose will transpose the values of src to dest . only works on non sliding Tensors
func (t *XTransposeD) Transpose(handle *Handle, perm []int32, srcdesc *gocudnn.TensorD, src cutil.Mem, destdesc *gocudnn.TensorD, dest cutil.Mem) error {

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
	var frmt gocudnn.TensorFormatFlag
	if xfrmt == frmt.NCHWvectC() || yfrmt == frmt.NCHWvectC() {
		return errors.New("TransPoseTensor NCHWvectC not supported")
	}

	dtype, xdims, _, err := srcdesc.Get()
	var dflag gocudnn.DataTypeFlag
	if dtype != dflag.Float() {
		return errors.New("Only Supported Format is float32")
	}
	if err != nil {
		return err
	}
	_, ydims, _, err := destdesc.Get()
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
		var devbuffer cutil.Mem
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
	var devbuffer cutil.Mem
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




*/
func findnewdims(original, perm []int32) []int32 {
	newdims := make([]int32, len(original))
	for i := 0; i < len(original); i++ {
		newdims[i] = original[perm[i]]
	}
	return newdims
}

//XResizeD is a struct that holds the reshape functions
type XResizeD struct {
	nearestfwdnhwc *cuda.Kernel
	nearestbwdnhwc *cuda.Kernel
	nearestfwdnchw *cuda.Kernel
	nearestbwdnchw *cuda.Kernel
	nearestfwdnhwcfp16 *cuda.Kernel
	nearestbwdnhwcfp16 *cuda.Kernel
	nearestfwdnchwfp16 *cuda.Kernel
	nearestbwdnchwfp16 *cuda.Kernel
	aligncorners   bool
}

//CreateResizeDesc creates a descriptor that holds the reshpaes
func CreateResizeDesc(handle *Handle, aligncorners bool) (*XResizeD, error) {
	nearestfwdnhwc, err := cuda.MakeKernel("NearestNeighborNHWC", handle.mod)
	if err != nil {
		return nil, err
	}
	nearestbwdnhwc, err := cuda.MakeKernel("NearestNeighborNHWCBack", handle.mod)
	if err != nil {
		return nil, err
	}
	nearestfwdnchw, err := cuda.MakeKernel("NearestNeighborNCHW", handle.mod)
	if err != nil {
		return nil, err
	}
	nearestbwdnchw, err := cuda.MakeKernel("NearestNeighborNCHWBack", handle.mod)
	if err != nil {
		return nil, err
	}
	nearestfwdnhwcfp16, err := cuda.MakeKernel("NearestNeighborNHWCFP16", handle.mod)
	if err != nil {
		return nil, err
	}
	nearestbwdnhwcfp16, err := cuda.MakeKernel("NearestNeighborNHWCBackFP16", handle.mod)
	if err != nil {
		return nil, err
	}
	nearestfwdnchwfp16, err := cuda.MakeKernel("NearestNeighborNCHWFP16", handle.mod)
	if err != nil {
		return nil, err
	}
	nearestbwdnchwfp16, err := cuda.MakeKernel("NearestNeighborNCHWBackFP16", handle.mod)
	if err != nil {
		return nil, err
	}
	return &XResizeD{
		nearestfwdnhwc: nearestfwdnhwc,
		nearestbwdnhwc: nearestbwdnhwc,
		nearestfwdnchw: nearestfwdnchw,
		nearestbwdnchw: nearestbwdnchw,
		nearestfwdnhwcfp16 :nearestfwdnhwcfp16 ,
		nearestbwdnhwcfp16 :nearestbwdnhwcfp16 ,
		nearestfwdnchwfp16 :nearestfwdnchwfp16 ,
		nearestbwdnchwfp16 :nearestbwdnchwfp16 ,
		aligncorners:   aligncorners,
	}, nil
}

//ResizeForward does the reshape operation
func (s *XResizeD) ResizeForward(handle *Handle, xdesc *gocudnn.TensorD, x cutil.Mem, ydesc *gocudnn.TensorD, y cutil.Mem) error {

	frmtx, dtypex, dimsx, _, err := xdesc.Get()
	if err != nil {
		return err
	}
	frmty, _, dimsy, _, err := ydesc.Get()
	if err != nil {
		return err
	}
	var fmtflag gocudnn.TensorFormat
	var dtypeflg gocudnn.DataType
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

		if dtypex==dtypeflg.Float(){
			return s.nearestfwdnhwc.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, x, dimsx[1], dimsx[2], dimsx[3], dimsy[1], dimsy[2], ratioh, ratiow, y)
		}else if  dtypex==dtypeflg.Half(){
			return s.nearestfwdnhwcfp16.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, x, dimsx[1], dimsx[2], dimsx[3], dimsy[1], dimsy[2], ratioh, ratiow, y)
		}

		
		
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
		if dtypex==dtypeflg.Float(){
			return s.nearestfwdnchw.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, x, dimsx[2], dimsx[3], dimsx[1], dimsy[2], dimsy[3], ratioh, ratiow, y)
		}else if  dtypex==dtypeflg.Half(){
			return s.nearestfwdnchwfp16.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, x, dimsx[2], dimsx[3], dimsx[1], dimsy[2], dimsy[3], ratioh, ratiow, y)
		}


	}
	return errors.New("Not Supported Tensor Format")
}

//ResizeBackward does a reshape backwards but it will add the errors on the backprop.
func (s *XResizeD) ResizeBackward(handle *Handle, dxdesc *gocudnn.TensorD, dx cutil.Mem, dydesc *gocudnn.TensorD, dy cutil.Mem) error {
	frmtx, dtypex, dimsx, _, err := dxdesc.Get()
	if err != nil {
		return err
	}

	frmty, _, dimsy, _, err := dydesc.Get()
	if err != nil {
		return err
	}

	var fmtflag gocudnn.TensorFormat
	var dtypeflg gocudnn.DataType
	if frmtx != frmty {
		return errors.New("ResizeForward - tensors must match")
	}
	sizeinbytes, err := dxdesc.GetSizeInBytes()
	if err != nil {
		return err
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

		cudart.Memset(dx, 0, sizeinbytes)
		if dtypex==dtypeflg.Float(){
			return s.nearestbwdnhwc.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, dx, dimsx[1], dimsx[2], dimsx[3], dimsy[1], dimsy[2], ratioh, ratiow, dy)
		}else if  dtypex==dtypeflg.Half(){
			return s.nearestbwdnhwcfp16.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, dx, dimsx[1], dimsx[2], dimsx[3], dimsy[1], dimsy[2], ratioh, ratiow, dy)
		}
	
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
		cudart.Memset(dx, 0, sizeinbytes)
		if dtypex==dtypeflg.Float(){
			return s.nearestbwdnchw.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, dx, dimsx[1], dimsx[2], dimsx[3], dimsy[1], dimsy[2], ratioh, ratiow, dy)
		}else if  dtypex==dtypeflg.Half(){
			return s.nearestbwdnchwfp16.Launch(conf.BlockCount, 1, 1, conf.ThreadPerBlock, 1, 1, 0, handle.s, aligned, conf.Elements, dx, dimsx[1], dimsx[2], dimsx[3], dimsy[1], dimsy[2], ratioh, ratiow, dy)
		}
		
	}
	return errors.New("Not Supported Tensor Format")
}

//XShapetoBatchD holds the kernel function
type XShapetoBatchD struct {
	nhwc *cuda.Kernel
	nchw *cuda.Kernel
}

//CreateShapetoBatchDesc creates a shape to batch desc
func CreateShapetoBatchDesc(handle *Handle) (*XShapetoBatchD, error) {
	nhwc, err := cuda.MakeKernel("ShapetoBatch4DNHWC", handle.mod)
	if err != nil {
		return nil, err
	}
	nchw, err := cuda.MakeKernel("ShapetoBatch4DNCHW", handle.mod)
	return &XShapetoBatchD{nhwc: nhwc, nchw: nchw}, err
}

//ShapeToBatch4d seperates chunks fo memory to blocks, so each window is the size of the block passed, and that those will becomoe the new batches.
//if S2B is true then it does the "Forward". Where the x values will be placed into the y tensor
//if S2B is false the y values will be placed into the x tensor. The C channel is the only thing that needs to be the same between tensor x and y.
//Any values that don't fit will get the zero value
//To get the y tensor please use FindShapetoBatchoutputTensor.
func (s *XShapetoBatchD) ShapeToBatch4d(handle *Handle, xDesc *gocudnn.TensorD, x cutil.Mem, yDesc *gocudnn.TensorD, y cutil.Mem, hstride int32, wstride int32, S2B bool) error {

	frmt, dtype, xdims, _, err := xDesc.Get()
	var dflag gocudnn.DataType
	if dtype != dflag.Float() {
		return errors.New("Only Supported dtype is float32")
	}

	frmt2, dtype, ydims, _, err := yDesc.Get()
	if dtype != dflag.Float() {
		return errors.New("Only Supported dytype is float32")
	}
	var tflag gocudnn.TensorFormat

	if frmt != frmt2 {
		return errors.New("TensorFormats Must be the same")
	}

	if !S2B { //When going backwards (aka B2S or !S2B) there can be overlap due to adding slide to the function.
		// This accumulates the values instead of just placing the values like S2B.
		// So, you have to set all the values of x to zero.
		sizeinbytes, err := xDesc.GetSizeInBytes()
		cudart.Memset(x, 0, sizeinbytes)
		if err != nil {
			return err
		}
		handle.s.Sync()
	}

	switch frmt {
	case tflag.NHWC():
		var n1 int32
		var n2 int32
		var HOverScan int32
		var WOverScan int32
		if ydims[1] == hstride {
			n1 = int32(divideandroundup(xdims[1]-ydims[1], hstride) + 1)
			HOverScan = 255

		} else {
			n1 = int32(divideandroundup(xdims[1]-ydims[1], hstride))
		}
		if ydims[2] == wstride {
			n2 = int32(divideandroundup(xdims[2]-ydims[2], wstride) + 1)
			WOverScan = 255
		} else {
			n2 = int32(divideandroundup(xdims[2]-ydims[2], wstride))
		}

		//n1 := divideandroundup(xdims[1]-ydims[1], hstride) + 1
		//	n2 := divideandroundup(xdims[2]-ydims[2], wstride) + 1
		if int32(n1*n2)*xdims[0] != ydims[0] || ydims[3] != xdims[3] {
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

		return s.nhwc.Launch(
			cfg.BlockCountx, cfg.BlockCounty, cfg.BlockCountz,
			cfg.ThreadPerBlockx, cfg.ThreadPerBlocky, cfg.ThreadPerBlockz,
			0, handle.s, cfg.Dimx, cfg.Dimy, cfg.Dimz, oHH, oHW, OriginalBatches, BatchedVol, OriginalVol, n1, n2, hstride, wstride, x, y, HOverScan, WOverScan, S2B)

	case tflag.NCHW():
		var n1 int32
		var n2 int32
		var HOverScan int32
		var WOverScan int32
		if ydims[2] == hstride {
			n1 = int32(divideandroundup(xdims[2]-ydims[2], hstride) + 1)
			HOverScan = 255

		} else {
			n1 = int32(divideandroundup(xdims[2]-ydims[2], hstride))
		}
		if ydims[3] == wstride {
			n2 = int32(divideandroundup(xdims[3]-ydims[3], wstride) + 1)
			WOverScan = 255
		} else {
			n2 = int32(divideandroundup(xdims[3]-ydims[3], wstride))
		}

		if int32(n1*n2)*xdims[0] != ydims[0] || ydims[1] != xdims[1] {
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

		return s.nchw.Launch(cfg.BlockCountx,
			cfg.BlockCounty,
			cfg.BlockCountz,
			cfg.ThreadPerBlockx,
			cfg.ThreadPerBlocky,
			cfg.ThreadPerBlockz,
			0, handle.s, cfg.Dimx, cfg.Dimy, cfg.Dimz, oHH, oHW, OriginalBatches, BatchedVol, OriginalVol, n1, n2, hstride, wstride, x, y, HOverScan, WOverScan, S2B)

	}
	return errors.New("Unsupported Tensor Format")
}

/*
func copytogpumalloc(x *GoPointer) (cutil.Mem, error) {

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
func copytogpuunified(x *GoPointer) (cutil.Mem, error) {

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
*/

//GetBatchtoShapeOutputProperties will place the batches into the shape.  It will only work if xdims[0]/(h*w) doesn't have a remainder.
func (s *XShapetoBatchD) GetBatchtoShapeOutputProperties(descX *gocudnn.TensorD, h, w, hstride, wstride int32) (gocudnn.TensorFormat, gocudnn.DataType, []int32, error) {
	xfrmt, dtype, dims, _, err := descX.Get()
	if err != nil {
		return 255, 255, nil, err
	}
	var dflag gocudnn.DataType
	if dtype != dflag.Float() {
		return 255, 255, nil, errors.New("Only Supported Format is float32")
	}
	var frmt gocudnn.TensorFormat

	if dims[0]%(h*w) != 0 {
		return 255, 255, nil, errors.New("descx batches/(h*w) must not have a remainder")
	}
	switch xfrmt {
	case frmt.NCHW():

		oh := dims[2] * h
		ow := dims[3] * w
		n1 := ((oh - dims[2]) / hstride) + 1
		n2 := ((ow - dims[3]) / wstride) + 1
		n := dims[0] / (n1 * n2)

		return frmt.NCHW(), dtype, []int32{n, dims[1], oh, ow}, nil

	case frmt.NHWC():
		oh := dims[1] * h
		ow := dims[2] * w
		n1 := ((oh - dims[1]) / hstride) + 1
		n2 := ((ow - dims[2]) / wstride) + 1
		n := dims[0] / (n1 * n2)
		return frmt.NHWC(), dtype, []int32{n, oh, ow, dims[3]}, nil

	default:
		return 255, 255, nil, errors.New("NHWC-Vec Not supported")
	}

}

//GetShapetoBatchOutputProperties returns properties to make a new descriptor
func (s *XShapetoBatchD) GetShapetoBatchOutputProperties(descX *gocudnn.TensorD, h, w, hstride, wstride int32) (gocudnn.TensorFormat, gocudnn.DataType, []int32, error) {
	xfrmt, dtype, dims, _, err := descX.Get()
	if err != nil {
		return 255, 255, nil, err
	}
	var dflag gocudnn.DataType
	if dtype != dflag.Float() {
		return 255, 255, nil, errors.New("Only Supported Format is float32")
	}
	var frmt gocudnn.TensorFormat

	switch xfrmt {
	case frmt.NCHW():

		var n1 int32
		var n2 int32
		if h == hstride {
			n1 = int32(divideandroundup(dims[2]-h, hstride) + 1)

		} else {
			n1 = int32(divideandroundup(dims[2]-h, hstride))
		}
		if w == wstride {
			n2 = int32(divideandroundup(dims[3]-w, wstride) + 1)

		} else {
			n2 = int32(divideandroundup(dims[3]-w, wstride))
		}

		return frmt.NCHW(), dtype, []int32{n1 * n2 * dims[0], dims[1], h, w}, nil

	case frmt.NHWC():
		var n1 int32
		var n2 int32
		if h == hstride {
			n1 = int32(divideandroundup(dims[1]-h, hstride) + 1)

		} else {
			n1 = int32(divideandroundup(dims[1]-h, hstride))
		}
		if w == wstride {
			n2 = int32(divideandroundup(dims[2]-w, wstride) + 1)

		} else {
			n2 = int32(divideandroundup(dims[2]-w, wstride))
		}

		return frmt.NHWC(), dtype, []int32{n1 * n2 * dims[0], h, w, dims[3]}, nil

	default:
		return 255, 255, nil, errors.New("NHWC-Vec Not supported")
	}

}

//GetShapetoBatchOutputPropertiesPLUS returns properties to make a new descriptor. PLUS the N1,N2 used to resize the dims
func (s *XShapetoBatchD) GetShapetoBatchOutputPropertiesPLUS(descX *gocudnn.TensorD, h, w, hstride, wstride int32) (gocudnn.TensorFormat, gocudnn.DataType, []int32, []int32, error) {
	xfrmt, dtype, dims, _, err := descX.Get()
	if err != nil {
		return 255, 255, nil, nil, err
	}
	var dflag gocudnn.DataType
	if dtype != dflag.Float() {
		return 255, 255, nil, nil, errors.New("Only Supported Format is float32")
	}
	var frmt gocudnn.TensorFormat

	switch xfrmt {
	case frmt.NCHW():

		var n1 int32
		var n2 int32
		if h == hstride {
			n1 = int32(divideandroundup(dims[2]-h, hstride) + 1)

		} else {
			n1 = int32(divideandroundup(dims[2]-h, hstride))
		}
		if w == wstride {
			n2 = int32(divideandroundup(dims[3]-w, wstride) + 1)

		} else {
			n2 = int32(divideandroundup(dims[3]-w, wstride))
		}

		return frmt.NCHW(), dtype, []int32{n1 * n2 * dims[0], dims[1], h, w}, []int32{n1, n2}, nil

	case frmt.NHWC():
		var n1 int32
		var n2 int32
		if h == hstride {
			n1 = int32(divideandroundup(dims[1]-h, hstride) + 1)

		} else {
			n1 = int32(divideandroundup(dims[1]-h, hstride))
		}
		if w == wstride {
			n2 = int32(divideandroundup(dims[2]-w, wstride) + 1)

		} else {
			n2 = int32(divideandroundup(dims[2]-w, wstride))
		}
		return frmt.NHWC(), dtype, []int32{n1 * n2 * dims[0], h, w, dims[3]}, []int32{n1, n2}, nil

	default:
		return 255, 255, nil, nil, errors.New("NHWC-Vec Not supported")
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
