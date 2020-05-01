package xtra

import (
	"errors"
	"fmt"
	"github.com/dereklstinson/gocudnn/cudart"
	"runtime"
	"sync"

	"github.com/dereklstinson/half"

	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocudnn/cuda"
	"github.com/dereklstinson/gocudnn/kernels"
	"github.com/dereklstinson/cutil"
)

//ConcatEx holds the concat kernels
type ConcatEx struct {
	fp32    *concat
	fp16    *concat
	streams []*cudart.Stream
}
type concat struct {
	nhwc *cuda.Kernel
	nchw *cuda.Kernel
}

//CreateConcatEx holds does concat for nchw and nhwc half and float tensors
func CreateConcatEx(h *Handle) (c *ConcatEx, err error) {
	c = new(ConcatEx)
	c.fp32 = new(concat)
	c.fp16 = new(concat)
	var ktf kernels.XtraKerns
	if h.w != nil {
		err = h.w.Work(func() error {
			c.fp32.nhwc, err = cuda.MakeKernel(ktf.ConcatNHWCEX(), h.mod)
			if err != nil {
				return err
			}
			c.fp32.nchw, err = cuda.MakeKernel(ktf.ConcatNCHWEX(), h.mod)
			if err != nil {
				return err
			}

			c.fp16.nhwc, err = cuda.MakeKernel(ktf.ConcatNHWCEXHalf(), h.mod)
			if err != nil {
				return err
			}
			c.fp16.nchw, err = cuda.MakeKernel(ktf.ConcatNCHWEXHalf(), h.mod)
			if err != nil {
				return err
			}
			return nil
		})

	} else {
		c.fp32.nhwc, err = cuda.MakeKernel(ktf.ConcatNHWCEX(), h.mod)
		if err != nil {
			return nil, err
		}
		c.fp32.nchw, err = cuda.MakeKernel(ktf.ConcatNCHWEX(), h.mod)
		if err != nil {
			return nil, err
		}

		c.fp16.nhwc, err = cuda.MakeKernel(ktf.ConcatNHWCEXHalf(), h.mod)
		if err != nil {
			return nil, err
		}
		c.fp16.nchw, err = cuda.MakeKernel(ktf.ConcatNCHWEXHalf(), h.mod)
		if err != nil {
			return nil, err
		}
	}
	if err != nil {
		return nil, err
	}

	return c, err
}

//GetOutputDimsFromInputDims gets the outputdims from inputdims passed
func (c *ConcatEx) GetOutputDimsFromInputDims(srcs [][]int32, frmt gocudnn.TensorFormat) (outputdims []int32, err error) {
	if srcs == nil {
		return nil, errors.New("(c *ConcatEx) GetOutputDimsFromInputDims(srcs [][]int32, format gocudnn.TensorFormat) : srcs can't be nil")
	}
	fflg := frmt
	var pdims []int32
	for i, dims := range srcs {
		if i == 0 {
			outputdims = make([]int32, len(dims))
			copy(outputdims, dims)

		} else {

			switch frmt {
			case fflg.NCHW():
				if checkminuschanneldim(dims, pdims, 1) {
					fmt.Println("dims", dims, "pdims", pdims)
					return nil, errors.New("(c *ConcatEx) GetOutputDimsFromInputDims(srcs [][]int32, frmt gocudnn.TensorFormat) : dims excluding the channel dim are not the same")
				}
				outputdims[1] += dims[1]
			case fflg.NHWC():
				if checkminuschanneldim(dims, pdims, len(dims)-1) {
					fmt.Println("dims", dims, "pdims", pdims)
					return nil, errors.New("(c *ConcatEx) GetOutputDimsFromInputDims(srcs [][]int32, frmt gocudnn.TensorFormat) : dims excluding the channel dim are not the same")
				}
				outputdims[len(dims)-1] += dims[len(dims)-1]
			default:
				return nil, errors.New("(c *ConcatEx) GetOutputDimsFromInputDims -  unsupported format")

			}
		}
		pdims = dims
	}
	if len(c.streams) < len(srcs) {
		streamdif := len(srcs) - len(c.streams)
		for i := 0; i < streamdif; i++ {
			stream, err := cudart.CreateNonBlockingStream()
			if err != nil {
				return nil, err
			}
			c.streams = append(c.streams, stream)
		}
	}
	return outputdims, nil
}

//checkdimsminus1dim if the same return false. if not the same return true.  this checks to see if values are the same minus the channel dim.
func checkminuschanneldim(dims1, dims2 []int32, skippeddim int) bool {
	if len(dims1) != len(dims2) {
		return true
	}
	for i := range dims1 {
		if i != skippeddim {
			if dims1[i] != dims2[i] {
				return true
			}
		}
	}
	return false
}

//GetOutputdims gets the concat tensor dims for the output tensor
func (c *ConcatEx) GetOutputdims(srcs []*gocudnn.TensorD) (outdims []int32, err error) {
	var prevD gocudnn.DataType
	var prevF gocudnn.TensorFormat
	var prevbatch int32
	fflg := prevF
	for i := range srcs {

		frmt, dtype, dims, _, err := srcs[i].Get()
		if err != nil {
			println("error in srcs[i].Get()")
			return nil, err
		}
		if i == 0 {
			outdims = make([]int32, len(dims))
			copy(outdims, dims)
		} else {
			if prevD != dtype || prevF != frmt || prevbatch != dims[0] {
				return nil, errors.New("(c *ConcatEx) GetOutputdims --- prevD != dtype || prevF != frmt || prevbatch!=dims[0]")
			}
			switch frmt {
			case fflg.NCHW():
				outdims[1] += dims[1]

			case fflg.NHWC():
				outdims[len(dims)-1] += dims[len(dims)-1]

			default:
				return nil, errors.New("(c *ConcatEx) GetOutputdims -  unsupported format")
			}
		}
		prevD, prevF, prevbatch = dtype, frmt, dims[0]
	}
	if len(c.streams) < len(srcs) {
		streamdif := len(srcs) - len(c.streams)
		for i := 0; i < streamdif; i++ {
			stream, err := cudart.CreateNonBlockingStream()
			if err != nil {
				return nil, err
			}
			c.streams = append(c.streams, stream)
		}
	}
	return outdims, err
}

//Op takes all the values in the srcs and concats them together into dest
func (c *ConcatEx) Op(h *Handle, srcs []*gocudnn.TensorD, srcsmem []cutil.Mem, alpha float64, dest *gocudnn.TensorD, destmem cutil.Mem, beta float64, forward bool) error {
	if h.w != nil {
		return h.w.Work(func() error {
			return c.op(h, srcs, srcsmem, alpha, dest, destmem, beta, forward)
		})
	}
	return c.op(h, srcs, srcsmem, alpha, dest, destmem, beta, forward)
}

//Op takes all the values in the srcs and concats them together into dest
func (c *ConcatEx) op(h *Handle, srcs []*gocudnn.TensorD, srcsmem []cutil.Mem, alpha float64, dest *gocudnn.TensorD, destmem cutil.Mem, beta float64, forward bool) error {

	dfrmt, ddtype, ddims, _, err := dest.Get()
	if err != nil {
		return nil
	}

	batches := ddims[0]
	destbatchvol := findvol(ddims[1:])

	if len(c.streams) < len(srcs) {
		streamdif := len(srcs) - len(c.streams)
		for i := 0; i < streamdif; i++ {
			stream, err := cudart.CreateNonBlockingStream()
			if err != nil {
				return err
			}
			c.streams = append(c.streams, stream)
		}
	}
	nstreams := len(c.streams)
	errs := make([]error, len(srcs))
	var wg sync.WaitGroup
	srcchanoffset := int32(0)
	for i := range srcs {

		pfflg := dfrmt
		srcdims := srcs[i].Dims()
		srcbatchvol := findvol(srcdims[1:])
		wg.Add(1)
		go func(i, nstreams int, srcchanoffset, srcbatchvol int32, ddtype gocudnn.DataType, dfrmt gocudnn.TensorFormat) {
			runtime.LockOSThread()
			dflg := ddtype
			fflg := dfrmt

			sfrmt, sdtype, sdims, _, err := srcs[i].Get()
			if err != nil {

				errs[i] = err //return err
			}
			if sfrmt != dfrmt || sdtype != ddtype {
				errs[i] = errors.New("(c *ConcatEx) Forward --- sfrmt!=dfrmt || sdtype!=ddtype")
				//	return errors.New("(c *ConcatEx) Forward --- sfrmt!=dfrmt || sdtype!=ddtype")
			}

			//	srctotalvol := findvol(sdims)
			switch ddtype {
			case dflg.Float():
				a := float32(alpha)
				b := float32(beta)
				switch dfrmt {
				case fflg.NCHW():
					config := h.LaunchConfig(srcbatchvol)
					err = c.fp32.nchw.Launch(config.BlockCount, 1, 1,
						config.ThreadPerBlock, 1, 1, 0, c.streams[i],
						config.Elements, batches, destbatchvol, srcchanoffset, srcsmem[i], a, srcbatchvol, destmem, b, forward)
					if err != nil {

						errs[i] = err //return err
					}

				case fflg.NHWC():
					config := h.LaunchConfig3d(sdims[1], sdims[2], sdims[3])
					err = c.fp32.nhwc.Launch(config.BlockCountx, config.BlockCounty, config.BlockCountz,
						config.ThreadPerBlockx, config.ThreadPerBlocky, config.ThreadPerBlockz, 0, c.streams[i],
						config.Dimx, config.Dimy, config.Dimz,
						batches, destbatchvol, ddims[len(ddims)-1],
						srcchanoffset, srcsmem[i], a, srcbatchvol, destmem, b, forward)
					if err != nil {

						errs[i] = err //return err
					}

				}
			case dflg.Half():
				a := half.NewFloat16(float32(alpha))
				b := half.NewFloat16(float32(beta))
				switch dfrmt {
				case fflg.NCHW():
					config := h.LaunchConfig(srcbatchvol)
					err = c.fp16.nchw.Launch(config.BlockCount, 1, 1,
						config.ThreadPerBlock, 1, 1, 0, c.streams[i],
						config.Elements, batches, destbatchvol, srcchanoffset,
						srcsmem[i], a, srcbatchvol, destmem, b, forward)
					if err != nil {
						errs[i] = err //return err
					}

				case fflg.NHWC():
					config := h.LaunchConfig3d(sdims[1], sdims[2], sdims[3])
					err = c.fp16.nhwc.Launch(config.BlockCountx, config.BlockCounty, config.BlockCountz,
						config.ThreadPerBlockx, config.ThreadPerBlocky, config.ThreadPerBlockz, 0, c.streams[i],
						config.Dimx, config.Dimy, config.Dimz, batches, destbatchvol,
						ddims[len(ddims)-1], srcchanoffset, srcsmem[i], a, srcbatchvol, destmem, b, forward)
					if err != nil {
						errs[i] = err //return err
					}

				}
			default:
				errs[i] = errors.New("(c *ConcatEx) Forward --unsupported TensorFormat")
				//		return errors.New("(c *ConcatEx) Forward --unsupported TensorFormat")
			}

			wg.Done()
			//c.streams[i].Sync()
			runtime.UnlockOSThread()
		}(i, nstreams, srcchanoffset, srcbatchvol, ddtype, dfrmt)
		switch dfrmt {
		case pfflg.NCHW():
			srcchanoffset += srcbatchvol
		case pfflg.NHWC():
			srcchanoffset += srcdims[len(srcdims)-1]
		}
	}

	wg.Wait()
	for i := range errs {
		if errs[i] != nil {
			return errs[i]
		}
	}
	return nil
}
