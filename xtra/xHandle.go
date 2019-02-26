package xtra

import (
	"github.com/dereklstinson/GoCudnn/cuda"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/kernels"
)

//Xtra is a holder for Xtra functions that are made by me, and not cuda or cudnn
type Xtra struct {
	notdefaultkernallocation bool
	kernellocation           string
}

//KernelLocation will set the direct kernel location and make for it kernel location
func (xtra *Xtra) KernelLocation(kernalfilelocation string) {
	xtra.notdefaultkernallocation = true
	xtra.kernellocation = kernalfilelocation
}

/*
//Handler to use this it must return nil on the ones it is not. error will be saying that is not it. Helpful when making new packages
type Handler interface {
	SetStream(s *Stream) error
}
*/

//XHandle is a handle for xtra functions. Right now all functions that use XHandle are strictly float32.
// Because I use gtx 1080ti(s) and there is basically no motivation to expand the capability.  Maybe if someone wants to get me
//A RTX2080ti I will do something about that. heh heh heh
type XHandle struct {
	mod                          *cuda.Module
	ptx                          string
	s                            gocu.Streamer
	c                            *cuda.Context
	maxblockthreads              int32
	muliproccessorcount          int32
	maxthreadspermultiproccessor int32
	maxblockdimsxyz              []int32
	maxgriddimsxyz               []int32
}

//SetStream sets a stream to be used by the handler
func (x *XHandle) SetStream(s gocu.Streamer) error {
	x.s = s
	return nil
}

//MakeXHandleV2 takes the kernel directory already made and in kernels and returns a XHandle
func (xtra Xtra) MakeXHandleV2(dev *cuda.Device) (*XHandle, error) {
	directory := "__default__"
	if xtra.notdefaultkernallocation == true {
		directory = xtra.kernellocation
	}
	return xtra.MakeXHandle(directory, dev)
}

//MakeXHandle makes one of them there "Xtra" Handles used for the xtra functions I added to gocudnn. You use MakeXHandleV2 if you want to use the default location
func (xtra Xtra) MakeXHandle(trainingfloatdir string, dev *cudart.Device) (*XHandle, error) {

	x := kernels.MakeMakeFile(trainingfloatdir, "gocudnnxtra", dev)
	//kerncode := kernels.LoadPTXFile(trainingfloatdir, x)
	mod, err := cuda.NewModule(trainingfloatdir + x)
	if err != nil {
		//fmt.Println(kerncode)
		return nil, err
	}
	mtpb, err := dev.MaxThreadsPerBlock()
	if err != nil {

		return nil, err
	}
	mmpt, err := dev.MaxThreadsPerMultiProcessor()
	if err != nil {

		return nil, err
	}

	nummp, err := dev.MultiProcessorCount()
	if err != nil {

		return nil, err
	}
	blockxyz, err := dev.MaxBlockDimXYZ()
	if err != nil {

		return nil, err
	}
	gridxyz, err := dev.MaxGridDimXYZ()
	if err != nil {

		return nil, err
	}
	//	kern,err:=cu.MakeKernel()
	return &XHandle{
		mod:                          mod,
		maxblockthreads:              mtpb,
		maxthreadspermultiproccessor: mmpt,
		muliproccessorcount:          nummp,
		maxblockdimsxyz:              blockxyz,
		maxgriddimsxyz:               gridxyz,
	}, nil
}

//Config is for a 1d kernel launch
type Config struct {
	Elements       int32
	ThreadPerBlock uint32
	BlockCount     uint32
}

//LaunchConfig returns a config struct that is used to configure some kernel launches
func (x *XHandle) LaunchConfig(elements int32) Config {
	ptc := min((x.muliproccessorcount * x.maxthreadspermultiproccessor), elements)
	threadperblock := min(1024, x.maxblockthreads)
	innerbcount := kernels.DivUp(ptc, threadperblock)
	bcount := min(innerbcount, x.muliproccessorcount)

	//blockcount:=math.Min()
	return Config{
		Elements:       elements,
		ThreadPerBlock: uint32(threadperblock),
		BlockCount:     uint32(bcount),
	}
}

//Config2d are parameters for the kernel launch
type Config2d struct {
	Dimx            int32
	Dimy            int32
	ThreadPerBlockx uint32
	ThreadPerBlocky uint32
	BlockCountx     uint32
	BlockCounty     uint32
}

//LaunchConfig2d returns configs for the kernel launch
func (x *XHandle) LaunchConfig2d(xdim, ydim int32) Config2d {
	if xdim < 1 || ydim < 1 {
		return Config2d{}
	}
	kthreadsperblock := int32(256)
	gputhreads := (x.muliproccessorcount * x.maxthreadspermultiproccessor)
	blockx := min(xdim, kthreadsperblock)
	blocky := max(kthreadsperblock/blockx, 1)
	maxblocks := max(gputhreads/kthreadsperblock, 1)
	conf := Config2d{}
	conf.Dimx = xdim
	conf.Dimy = ydim
	conf.ThreadPerBlockx = uint32(blockx)
	conf.ThreadPerBlocky = uint32(blocky)
	ratiox := divideandroundup(xdim, blockx)
	gridx := uint32(min(int32(ratiox), maxblocks))
	conf.BlockCountx = gridx
	conf.BlockCounty = uint32(min(maxblocks/int32(gridx), max(ydim/blocky, 1)))
	return conf
}

//Config3d are parameters for the kernel launch
type Config3d struct {
	Dimx            int32
	Dimy            int32
	Dimz            int32
	ThreadPerBlockx uint32
	ThreadPerBlocky uint32
	ThreadPerBlockz uint32
	BlockCountx     uint32
	BlockCounty     uint32
	BlockCountz     uint32
}

//LaunchConfig3d returns configs for the kernel launch
func (x *XHandle) LaunchConfig3d(xdim, ydim, zdim int32) Config3d {
	if xdim < 1 || ydim < 1 {
		return Config3d{}
	}
	kthreadsperblock := int32(256)

	gputhreads := (x.muliproccessorcount * x.maxthreadspermultiproccessor)
	blockx := min3(xdim, kthreadsperblock, x.maxblockdimsxyz[0])
	blocky := min3(ydim, max(kthreadsperblock/blockx, 1), x.maxblockdimsxyz[1])
	blockz := min3(zdim, max(kthreadsperblock/(blockx*blocky), 1), x.maxblockdimsxyz[2])
	maxblocks := max(gputhreads/kthreadsperblock, 1)
	ratiox := divideandroundup(xdim, blockx)
	gridx := uint32(min3(maxblocks, int32(ratiox), x.maxgriddimsxyz[0]))
	ratioy := divideandroundup(ydim, blocky)
	ratioy2 := divideandroundup(maxblocks, int32(gridx))
	gridy := uint32(min3(int32(ratioy), int32(ratioy2), x.maxgriddimsxyz[1]))
	ratioz := divideandroundup(maxblocks, int32(gridx*gridy))
	ratioz2 := divideandroundup(zdim, blockz)
	gridz := uint32(min3(int32(ratioz), int32(ratioz2), x.maxgriddimsxyz[2]))
	conf := Config3d{}
	conf.Dimx = xdim
	conf.Dimy = ydim
	conf.Dimz = zdim
	conf.ThreadPerBlockx = uint32(blockx)
	conf.ThreadPerBlocky = uint32(blocky)
	conf.ThreadPerBlockz = uint32(blockz)

	conf.BlockCountx = gridx
	conf.BlockCounty = gridy
	conf.BlockCountz = gridz
	return conf
}
func max(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}
func max3(a, b, c int32) int32 {
	return max(max(a, b), c)
}
func min3(a, b, c int32) int32 {
	return min(min(a, b), c)
}
func min(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

/*
func (t *XHandle) GetCudnnHandle() (*Handle, error) {
	return nil, errors.New("Not a CudnnHandle")
}
func (t *XHandle) GetCudaContext() (*Context, error) {
	return nil, errors.New("Not a CudaContext")
}
func (t *XHandle) GetXHandle() (*XHandle, error) {
	return t, nil
}
*/
