package gocudnn

/*
#include <cuda.h>
#include <cuda_runtime_api.h>
const void ** voiddptrnull = NULL;

*/
import "C"
import (
	"errors"
	"unsafe"
)

//Kernel is a function stored on the gpu.
type Kernel struct {
	name string
	f    C.CUfunction
	m    *Module
}

//Module are used to hold kernel functions on the device that is in use
type Module struct {
	m      C.CUmodule
	loaded bool
}

func (m *Module) c() C.CUmodule { return m.m }

//extern __host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);

//NewModule creates a module in the current context.  The current context needs to be a cuda context not a cudnn handle.
func (cu Cuda) NewModule(filename string) (*Module, error) {
	var mod C.CUmodule
	fname := C.CString(filename)
	x := C.cuModuleLoad(&mod, fname)
	return &Module{
		m:      mod,
		loaded: true,
	}, newErrorDriver("NewModule", x)
}

//UnLoad Loads a Module
func (m *Module) UnLoad() error {
	x := C.cuModuleUnload(m.m)
	m.loaded = false
	return newErrorDriver("UnLoadModule", x)

}

//Load Loads a Module with a ptx file name
func (m *Module) Load(filename string) error {
	if m.loaded == true {
		return errors.New("(*Module)Load: Goside: Already Loaded")
	}
	fname := C.CString(filename)
	x := C.cuModuleLoad(&m.m, fname)
	return newErrorDriver("Load", x)
}

//MakeKernel makes a kernel.  (basically a function in cuda) If module is unloaded, then the kernel returned won't work
func (cu Cuda) MakeKernel(kname string, m *Module) (*Kernel, error) {
	var kern C.CUfunction
	if m.loaded == false {
		return nil, errors.New("MakeKernel: Module Not Loaded")
	}
	name := C.CString(kname)
	err := newErrorDriver("MakeKernel", C.cuModuleGetFunction(&kern, m.m, name))
	if err != nil {
		return nil, err
	}
	return &Kernel{
		name: kname,
		m:    m,
		f:    kern,
	}, nil
}

//Launch launches the kernel that is stored in the hidden module in this struct.
//gx,gy,gz are the grid values.
//bx,by,bz are the block values.
//shared is the shared memory size
//stream is the cuda stream this module will use.
//args are the arguments that are used for the kernel. kernels are written in a .cu file.  You will need to make a .ptx file in order to use it.
//you can check out the kernels package to get an idea of how to make the stuff.
func (k *Kernel) Launch(gx, gy, gz, bx, by, bz, shared uint32, stream *Stream, args ...interface{}) error {

	unsafearray, err := ifacetounsafe(args)
	if err != nil {
		return err
	}
	var shold C.cudaStream_t

	if stream == nil {
		shold = nil
	} else {
		shold = stream.stream
	}
	return newErrorDriver("cuLaunchKernel",
		C.cuLaunchKernel(k.f,
			C.uint(gx),
			C.uint(gy),
			C.uint(gz),
			C.uint(bx),
			C.uint(by),
			C.uint(bz),
			C.uint(shared),
			shold,
			&unsafearray[0],
			nil,
		))
}

//KernelArguments need to be loaded before using LaunchV2.  Handy if most of the parameters are not changing. Also, if you want to parallelize it then you can use this.
type KernelArguments struct {
	gx, gy, gz CUInt
	bx, by, bz CUInt
	shared     CUInt
	stream     *Stream
	args       []interface{}
}

//SetGrid sets the grid dims
func (k *KernelArguments) SetGrid(gx, gy, gz uint32) {
	k.gx, k.gy, k.gz = CUInt(gx), CUInt(gy), CUInt(gz)

}

//GetGrid returns the grid values
func (k *KernelArguments) GetGrid() (uint32, uint32, uint32) {
	return uint32(k.gx), uint32(k.gy), uint32(k.gz)
}

//SetBlock sets the block dims
func (k *KernelArguments) SetBlock(bx, by, bz uint32) {
	k.bx, k.by, k.bz = CUInt(bx), CUInt(by), CUInt(bz)

}

//GetBlock returns the block values
func (k *KernelArguments) GetBlock() (uint32, uint32, uint32) {
	return uint32(k.bx), uint32(k.by), uint32(k.bz)

}

//SetShared sets the shared dims
func (k *KernelArguments) SetShared(sharedsize uint32) {
	k.shared = CUInt(sharedsize)
}

//GetShared returns the shared memory size value
func (k *KernelArguments) GetShared() uint32 {
	return uint32(k.shared)
}

//SetArguments sets the arguments
func (k *KernelArguments) SetArguments(args ...interface{}) {
	k.args = args
}

//GetArguments returns the empty interface array of arguments
func (k *KernelArguments) GetArguments() []interface{} {
	return k.args
}

//LaunchV2 is like launch but it takes KernelArgument struct.
func (k *Kernel) LaunchV2(p KernelArguments) error {
	unsafearray, err := ifacetounsafe(p.args)
	if err != nil {
		return err
	}
	var shold C.cudaStream_t

	if p.stream == nil {
		shold = nil
	} else {
		shold = p.stream.stream
	}
	return newErrorDriver("cuLaunchKernel",
		C.cuLaunchKernel(k.f,
			p.gx.c(),
			p.gy.c(),
			p.gz.c(),
			p.bx.c(),
			p.by.c(),
			p.bz.c(),
			p.shared.c(),
			shold,
			&unsafearray[0],
			nil,
		))
}
func ifacetounsafe(args ...interface{}) ([]unsafe.Pointer, error) {
	array := make([]unsafe.Pointer, len(args))
	for i := 0; i < len(args); i++ {
		switch x := args[i].(type) {
		case Malloced:
			if x.ptr == nil {
				return nil, errors.New("Memory Doesn't Have A Pointer")
			}
			array[i] = x.ptr
		default:
			scalar := CScalarConversion(x)
			if scalar == nil {
				return nil, errors.New("Not a supported value")
			}
			array[i] = scalar.CPtr()
		}

	}
	return array, nil
}
