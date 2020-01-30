package xtra

import (
	"errors"
	"fmt"
	"github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cuda"
	"github.com/dereklstinson/GoCudnn/kernels"
	"github.com/dereklstinson/cutil"
)

//ConcatEx holds the concat kernels
type ConcatEx struct {
	fp32 *concat
	fp16 *concat
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

	return c, err
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
	return outdims, err
}

//Op takes all the values in the srcs and concats them together into dest
func (c *ConcatEx) Op(h *Handle, srcs []*gocudnn.TensorD, srcsmem []cutil.Mem, dest *gocudnn.TensorD, destmem cutil.Mem, forward bool) error {
	dfrmt, ddtype, ddims, _, err := dest.Get()
	dflg := ddtype
	batches := ddims[0]
	destbatchvol := findvol(ddims[1:])
	if err != nil {
		return nil
	}
	fflg := dfrmt
	srcchanoffset := int32(0)
	for i := range srcs {
		sfrmt, sdtype, sdims, _, err := srcs[i].Get()
		if err != nil {
			return err
		}
		if sfrmt != dfrmt || sdtype != ddtype {
			return errors.New("(c *ConcatEx) Forward --- sfrmt!=dfrmt || sdtype!=ddtype")
		}

		srcbatchvol := findvol(sdims[1:])
		//	srctotalvol := findvol(sdims)
		println("src chan offset:", srcchanoffset)
		switch ddtype {
		case dflg.Float():
			switch dfrmt {
			case fflg.NCHW():
				config := h.LaunchConfig(srcbatchvol)
				err = c.fp32.nchw.Launch(config.BlockCount, 1, 1,
					config.ThreadPerBlock, 1, 1, 0, h.s,
					config.Elements, batches, destbatchvol, srcchanoffset, srcsmem[i], srcbatchvol, destmem, forward)
				if err != nil {
					return err
				}

				srcchanoffset += srcbatchvol

			case fflg.NHWC():
				config := h.LaunchConfig3d(sdims[1], sdims[2], sdims[3])
				fmt.Println(sdims[1], sdims[2], sdims[3])
				err = c.fp32.nhwc.Launch(config.BlockCountx, config.BlockCounty, config.BlockCountz,
					config.ThreadPerBlockx, config.ThreadPerBlocky, config.ThreadPerBlockz, 0, h.s,
					config.Dimx, config.Dimy, config.Dimz,
					batches, destbatchvol, ddims[len(ddims)-1],
					srcchanoffset, srcsmem[i], srcbatchvol, destmem, forward)
				if err != nil {
					return err
				}
				srcchanoffset += sdims[len(sdims)-1]

			}
		case dflg.Half():
			switch dfrmt {
			case fflg.NCHW():
				config := h.LaunchConfig(srcbatchvol)
				err = c.fp16.nchw.Launch(config.BlockCount, 1, 1,
					config.ThreadPerBlock, 1, 1, 0, h.s,
					config.Elements, batches, destbatchvol, srcchanoffset, srcsmem[i], srcbatchvol, destmem, forward)
				if err != nil {
					return err
				}
				srcchanoffset += srcbatchvol
			case fflg.NHWC():
				config := h.LaunchConfig3d(sdims[1], sdims[2], sdims[3])
				err = c.fp16.nhwc.Launch(config.BlockCountx, config.BlockCounty, config.BlockCountz,
					config.ThreadPerBlockx, config.ThreadPerBlocky, config.ThreadPerBlockz, 0, h.s,
					config.Dimx, config.Dimy, config.Dimz, batches, destbatchvol, ddims[len(ddims)-1], srcchanoffset, srcsmem[i], srcbatchvol, destmem, forward)
				if err != nil {
					return err
				}
				srcchanoffset += sdims[len(sdims)-1]
			}
		default:
			return errors.New("(c *ConcatEx) Forward -- unsupported datatype")
		}
	}

	return nil
}
