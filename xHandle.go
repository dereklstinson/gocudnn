package gocudnn

import (
	"github.com/dereklstinson/GoCudnn/kernels"
)

//Xtra is a holder for Xtra functions that are made by me, and not cuda or cudnn
type Xtra struct {
}

//Handler to use this it must return nil on the ones it is not. error will be saying that is not it. Helpful when making new packages
type Handler interface {
	SetStream(s *Stream) error
}

//XHandle is a handle for xtra functions
type XHandle struct {
	mod                          *Module
	ptx                          string
	s                            *Stream
	c                            *Context
	maxblockthreads              int32
	muliproccessorcount          int32
	maxthreadspermultiproccessor int32
}

//SetStream sets a stream to be used by the handler
func (t *XHandle) SetStream(s *Stream) error {
	t.s = s
	return nil
}

//MakeXHandle makes one of them there "Xtra" Handles used for the xtra functions I added to gocudnn
func (xtra Xtra) MakeXHandle(trainingfloatdir string, dev *Device) (*XHandle, error) {
	var cu Cuda
	x := kernels.MakeMakeFile(trainingfloatdir, "gocudnnxtra", dev)
	//kerncode := kernels.LoadPTXFile(trainingfloatdir, x)
	mod, err := cu.NewModule(trainingfloatdir + x)
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
	//	kern,err:=cu.MakeKernel()
	return &XHandle{
		mod:                          mod,
		maxblockthreads:              mtpb,
		maxthreadspermultiproccessor: mmpt,
		muliproccessorcount:          nummp,
	}, nil
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
