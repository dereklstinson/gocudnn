package gocu

//Attributes are the attributes a device needs to return.
type Attributes interface {
	MaxThreadsPerMultiProcessor() (int32, error)
	MultiProcessorCount() (int32, error)
	MaxThreadsPerBlock() (int32, error)
	MaxGridDimXYZ() ([]int32, error)
	MaxBlockDimXYZ() ([]int32, error)
}

//Config holds configuration values for launching a kernel
type Config struct {
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

//ConfigHelper creates Config structs through methods.
type ConfigHelper struct {
	maxblockthreads              int32
	muliproccessorcount          int32
	maxthreadspermultiproccessor int32
	maxblockdimsxyz              []int32
	maxgriddimsxyz               []int32
}

//CreateConfigHelper a launch config helper.
func CreateConfigHelper(device Attributes) (*ConfigHelper, error) {
	mtpb, err := device.MaxThreadsPerBlock()
	if err != nil {

		return nil, err
	}
	mmpt, err := device.MaxThreadsPerMultiProcessor()
	if err != nil {

		return nil, err
	}

	nummp, err := device.MultiProcessorCount()
	if err != nil {

		return nil, err
	}
	blockxyz, err := device.MaxBlockDimXYZ()
	if err != nil {

		return nil, err
	}
	gridxyz, err := device.MaxGridDimXYZ()
	if err != nil {

		return nil, err
	}
	return &ConfigHelper{
		maxblockthreads:              mtpb,
		maxthreadspermultiproccessor: mmpt,
		muliproccessorcount:          nummp,
		maxblockdimsxyz:              blockxyz,
		maxgriddimsxyz:               gridxyz,
	}, nil
}

//CreateLaunchConfig creates a launch configurator.  It is used when launching kernels
func (l *ConfigHelper) CreateLaunchConfig(x, y, z int32) (launch Config) {

	if x > 1 && y == 1 && z == 1 {
		ptc := min((l.muliproccessorcount * l.maxthreadspermultiproccessor), x)
		threadperblock := min(1024, l.maxblockthreads)
		innerbcount := divup(ptc, threadperblock)
		bcount := min(innerbcount, l.muliproccessorcount)
		return Config{
			Dimx:            x,
			Dimy:            1,
			Dimz:            1,
			ThreadPerBlockx: (uint32)(threadperblock),
			ThreadPerBlocky: 1,
			ThreadPerBlockz: 1,
			BlockCountx:     (uint32)(bcount),
			BlockCounty:     1,
			BlockCountz:     1,
		}
	} else if x > 1 && y > 1 && z == 1 {
		kthreadsperblock := int32(256)
		gputhreads := (l.muliproccessorcount * l.maxthreadspermultiproccessor)
		blockx := min(x, kthreadsperblock)
		blocky := max(kthreadsperblock/blockx, 1)
		maxblocks := max(gputhreads/kthreadsperblock, 1)
		ratiox := divup(x, blockx)
		gridx := uint32(min((ratiox), maxblocks))
		gridy := uint32(min(maxblocks/int32(gridx), max(y/blocky, 1)))
		return Config{
			Dimx:            x,
			Dimy:            y,
			Dimz:            1,
			ThreadPerBlockx: uint32(blockx),
			ThreadPerBlocky: uint32(blocky),
			ThreadPerBlockz: 1,
			BlockCountx:     gridx,
			BlockCounty:     gridy,
			BlockCountz:     1,
		}
	} else if x > 1 && y > 1 && z > 1 {
		kthreadsperblock := int32(256)
		gputhreads := (l.muliproccessorcount * l.maxthreadspermultiproccessor)
		blockx := min3(x, kthreadsperblock, l.maxblockdimsxyz[0])
		blocky := min3(y, max(kthreadsperblock/blockx, 1), l.maxblockdimsxyz[1])
		blockz := min3(z, max(kthreadsperblock/(blockx*blocky), 1), l.maxblockdimsxyz[2])
		maxblocks := max(gputhreads/kthreadsperblock, 1)
		ratiox := divup(x, blockx)
		gridx := uint32(min3(maxblocks, (ratiox), l.maxgriddimsxyz[0]))
		ratioy := divup(y, blocky)
		ratioy2 := divup(maxblocks, int32(gridx))
		gridy := uint32(min3((ratioy), (ratioy2), l.maxgriddimsxyz[1]))
		ratioz := divup(maxblocks, int32(gridx*gridy))
		ratioz2 := divup(z, blockz)
		gridz := uint32(min3((ratioz), (ratioz2), l.maxgriddimsxyz[2]))
		return Config{
			Dimx:            x,
			Dimy:            y,
			Dimz:            z,
			ThreadPerBlockx: uint32(blockx),
			ThreadPerBlocky: uint32(blocky),
			ThreadPerBlockz: uint32(blockz),
			BlockCountx:     gridx,
			BlockCounty:     gridy,
			BlockCountz:     gridz,
		}
	}

	return Config{}
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

//SimpleGridCalculator will output the size of the grid given the size of the block. size is the number of values. blocksizelllll should be multiples of 32!
func simplegridcalculator(blocksize uint32, size uint32) uint32 {
	return ((size - 1) / blocksize) + 1
}

//DivUp is same function from tensorflow
func divup(a, b int32) int32 {
	return (a + b - 1) / b
}

//DivUpUint32 does it in uint 32
func divupuint32(a, b uint32) uint32 {
	return (a + b - 1) / b
}
