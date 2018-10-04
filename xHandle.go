package gocudnn

import (
	"github.com/dereklstinson/GoCudnn/kernels"
)

//Xtra is a holder for Xtra functions that are made by me, and not cuda or cudnn
type Xtra struct {
}

//Contexter to use this it must return nil on the ones it is not. error will be saying that is not it. Helpful when making new packages
type Handler interface {
	SetStream(s *Stream) error
}

//XHandle is a handle for xtra functions
type XHandle struct {
	mod *Module
	ptx string
	s   *Stream
	c   *Context
}

func (t *XHandle) SetStream(s *Stream) error {
	t.s = s
	return nil
}

func (xtra Xtra) MakeXHandle(trainingfloatdir string, dev *Device) (*XHandle, error) {
	var cu Cuda
	x := kernels.MakeMakeFile(trainingfloatdir, "gocudnnxtra", dev)
	//kerncode := kernels.LoadPTXFile(trainingfloatdir, x)
	mod, err := cu.NewModule(trainingfloatdir + x)
	if err != nil {
		//fmt.Println(kerncode)
		return nil, err
	}
	//	kern,err:=cu.MakeKernel()
	return &XHandle{
		mod: mod,
		//	ptx: kerncode,
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
