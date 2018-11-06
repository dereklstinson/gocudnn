package gocudnn

/*
#include <cuda.h>
#include <cuda_runtime_api.h>
const void ** voiddptrnull = NULL;
const size_t ptrSize = sizeof(void *);
const size_t maxArgSize = 8;
const CUjit_option * nullJitOptions = NULL;


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

//NewModuleEx takes a string of the ptx data
func (cu Cuda) NewModuleEx(Ptx string) (*Module, error) {
	var mod C.CUmodule
	cptx := unsafe.Pointer(C.CString(Ptx))
	x := C.cuModuleLoadDataEx(&mod, cptx, 0, C.nullJitOptions, C.voiddptrnull)
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

//LoadEx loads the ptx string straightup
func (m *Module) LoadEx(ptx string) error {
	if m.loaded == true {
		return errors.New("(*Module)LoadEx: Goside: Already Loaded")
	}
	cptx := unsafe.Pointer(C.CString(ptx))

	x := C.cuModuleLoadDataEx(&m.m, cptx, 0, C.nullJitOptions, C.voiddptrnull)
	return newErrorDriver("LoadEx", x)
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

const pointerSize = 8

func safeUintToC(x uint32) C.uint {
	if x > uint32(^C.uint(0)) {
		panic("uint value out of bounds")
	}
	return C.uint(x)
}

func safeIntToC(x int32) C.int {
	if x > int32(C.int(^C.uint(0)/2)) {
		panic("int value out of bounds")
	} else if x < int32((-C.int(^C.uint(0)/2))-1) {
		panic("int value out of bounds")
	}
	return C.int(x)
}
func offSet(ptr unsafe.Pointer, i int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(ptr) + pointerSize*uintptr(i))
}

//Launch will launch a kernal that is in it
func (k *Kernel) Launch(gx, gy, gz, bx, by, bz, shared uint32, stream *Stream, args ...interface{}) error {

	//	kernelParams, err := ifacetounsafe(args)
	kernelParams := make([]unsafe.Pointer, len(args))
	err := ifacetounsafe(args, kernelParams)
	if err != nil {
		return err
	}
	//fmt.Println("Length of params:", len(kernelParams))
	//Start straight up took this from gorgonia/cu
	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))

	for i := 0; i < len(kernelParams); i++ {
		//fmt.Println(i)
		*((*unsafe.Pointer)(offSet(argp, i))) = offSet(argv, i) // argp[i] = &argv[i]
		//fmt.Println("kernel address "+strconv.Itoa(i), ":", kernelParams[i])
		//holder :=
		*((*uint64)(offSet(argv, i))) = *((*uint64)(kernelParams[i])) //holder // argv[i] = *kernelParams[i]
	}
	defer C.free(argv)
	defer C.free(argp)

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
			(*unsafe.Pointer)(argp),
			C.voiddptrnull,
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
func ifacetounsafe(args []interface{}, array []unsafe.Pointer) error {
	//array := make([]unsafe.Pointer, len(args))
	for i := 0; i < len(args); i++ {

		switch x := args[i].(type) {
		case nil:
			array[i] = unsafe.Pointer(C.voiddptrnull)
		case *Malloced:
			if x == nil {
				array[i] = unsafe.Pointer(C.voiddptrnull)
			}
			if x.ptr == nil {
				return errors.New("Memory Doesn't Have A Pointer")
			}
			array[i] = unsafe.Pointer(&x.ptr)
		case *GoPointer:
			return errors.New("*GoPointer not supported")
		default:
			scalar := CScalarConversion(x)
			if scalar == nil {
				return errors.New("Not a supported value")
			}
			array[i] = scalar.CPtr()
		}

	}
	return nil
}

/*

func cleanKernelArguments(args []interface{}, newArgs []unsafe.Pointer,
	f func(args []unsafe.Pointer) error) error {
	if len(args) == 0 {
		return f(newArgs)
	}

	if buf, ok := args[0].(Buffer); ok {
		var res error
		buf.WithPtr(func(ptr unsafe.Pointer) {
			tempArgs := append([]interface{}{ptr}, args[1:]...)
			res = cleanKernelArguments(tempArgs, newArgs, f)
		})
		return res
	}

	valPtr := unsafe.Pointer(C.malloc(C.maxArgSize))
	defer C.free(valPtr)

	switch x := args[0].(type) {
	case uint:
		val := safeUintToC(x)
		C.memcpy(valPtr, unsafe.Pointer(&val), 4)
	case int:
		val := safeIntToC(x)
		C.memcpy(valPtr, unsafe.Pointer(&val), 4)
	case float32:
		val := C.float(x)
		C.memcpy(valPtr, unsafe.Pointer(&val), 4)
	case float64:
		val := C.double(x)
		C.memcpy(valPtr, unsafe.Pointer(&val), 8)
	case unsafe.Pointer:
		C.memcpy(valPtr, unsafe.Pointer(&x), C.ptrSize)
	}



*/
/*
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
*/
/*
func ifacetocunsafeprototype(args []interface{}) ([]unsafe.Pointer, error) {

	//	fmt.Println("arguments passed", args)
	//fmt.Println("Length of Args", len(args))
	unzippedargs := make([]interface{}, 0)
	for i := range args {
		switch x := args[i].(type) {
		case []int32:
			for j := range x {
				unzippedargs = append(unzippedargs, x[j])
			}
		case []float64:
			for j := range x {
				unzippedargs = append(unzippedargs, x[j])
			}
		case []float32:
			for j := range x {
				unzippedargs = append(unzippedargs, x[j])
			}
		case []int:
			for j := range x {
				unzippedargs = append(unzippedargs, x[j])
			}
		case []uint32:
			for j := range x {
				unzippedargs = append(unzippedargs, x[j])
			}
		case []uint:
			for j := range x {
				unzippedargs = append(unzippedargs, x[j])
			}
		case []byte:
			for j := range x {
				unzippedargs = append(unzippedargs, x[j])
			}
		default:
			unzippedargs = append(unzippedargs, args[i])
		}
	}
	array := make([]unsafe.Pointer, len(unzippedargs))

	for i := range unzippedargs {

		switch x := unzippedargs[i].(type) {
		case *Malloced:
			if x.ptr == nil {
				return nil, errors.New("Memory Doesn't Have A Pointer")
			}
			array[i] = unsafe.Pointer(&x.ptr)
		default:
			scalar := CScalarConversion(x)
			if scalar == nil {
				fmt.Println(unzippedargs[i])
				return nil, errors.New("Not a supported value")
			}
			array[i] = scalar.CPtr()
		}

	}
	return array, nil
}

*/
