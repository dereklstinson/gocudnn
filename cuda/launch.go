package cuda

/*
#include <string.h>
#include <cuda.h>


const void ** voiddptrnull = NULL;
const size_t ptrSize = sizeof(void *);
const size_t maxArgSize = 8;
const CUjit_option * nullJitOptions = NULL;


*/
import "C"
import (
	"errors"
	"fmt"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/cutil"
	"github.com/dereklstinson/half"
	"runtime"
	"sync"
	"unsafe"
)

//Kernel is a function stored on the gpu.
type Kernel struct {
	name string
	f    C.CUfunction
	m    *Module
	//notconcurentargs *launchargs
	//concurrentargs   []*launchargs
	mux sync.Mutex
}

//Module are used to hold kernel functions on the device that is in use
type Module struct {
	m      C.CUmodule
	loaded bool
}
type launchargs struct {
	args []unsafe.Pointer
}

func (m *Module) c() C.CUmodule { return m.m }

//NewModule creates a module in the current context.
func NewModule(filename string) (module *Module, err error) {
	var mod C.CUmodule
	fname := C.CString(filename)
	defer C.free((unsafe.Pointer)(fname))
	x := C.cuModuleLoad(&mod, fname)
	module = &Module{
		m:      mod,
		loaded: true,
	}
	return module, newErrorDriver("NewModule", x)
}

//NewModuleData takes a string of the ptx data
func NewModuleData(ptx string) (*Module, error) {
	var mod C.CUmodule
	cptx := C.CString(ptx)
	defer C.free((unsafe.Pointer)(cptx))
	x := C.cuModuleLoadData(&mod, (unsafe.Pointer)(cptx))
	//x := C.cuModuleLoadDataEx(&mod, (unsafe.Pointer)(&data[0]), 0, C.nullJitOptions, C.voiddptrnull)
	return &Module{

		m:      mod,
		loaded: true,
	}, newErrorDriver("NewModuleData", x)
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
	defer C.free((unsafe.Pointer)(fname))
	x := C.cuModuleLoad(&m.m, fname)
	m.loaded = true
	return newErrorDriver("Load", x)
}

//LoadData loads a string in ptx form.
func (m *Module) LoadData(ptx string) error {
	/*
		if m.loaded == true {
			return errors.New("(*Module)LoadEx: Goside: Already Loaded")
		}
	*/
	cptx := unsafe.Pointer(C.CString(ptx))
	defer C.free((unsafe.Pointer)(cptx))
	x := C.cuModuleLoadData(&m.m, cptx)
	m.loaded = true
	return newErrorDriver("LoadData", x)
}

//LoadEx loads the ptx string straightup
func (m *Module) LoadEx(ptx string) error {
	/*
		if m.loaded == true {
			return errors.New("(*Module)LoadEx: Goside: Already Loaded")
		}
	*/
	cptx := unsafe.Pointer(C.CString(ptx))
	defer C.free((unsafe.Pointer)(cptx))
	x := C.cuModuleLoadDataEx(&m.m, cptx, 0, C.nullJitOptions, C.voiddptrnull)
	m.loaded = true

	return newErrorDriver("LoadEx", x)
}

//MakeKernel makes a kernel.  (basically a function in cuda) If module is unloaded, then the kernel returned won't work
func MakeKernel(kname string, m *Module) (k *Kernel, err error) {
	var kern C.CUfunction
	if m.loaded == false {
		return nil, errors.New("MakeKernel: Module Not Loaded")
	}

	name := C.CString(kname)
	defer C.free((unsafe.Pointer)(name))
	err = newErrorDriver("MakeKernel", C.cuModuleGetFunction(&kern, m.m, name))
	if err != nil {
		return nil, err
	}
	k = &Kernel{
		name: kname,
		m:    m,
		f:    kern,
	}

	runtime.SetFinalizer(k, destroycudakernel)

	return k, nil
}

//Launch launches the kernel that is stored in the hidden module in this struct.
//gx,gy,gz are the grid values.
//bx,by,bz are the block values.
//shared is the shared memory size
//stream is the cuda stream this module will use.
//args are the arguments that are used for the kernel. kernels are written in a .cu file.  You will need to make a .ptx file in order to use it.
//you can check out the kernels package to get an idea of how to make the stuff.

const pointerSize = 8

func offSet(ptr unsafe.Pointer, i int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(ptr) + pointerSize*uintptr(i))
}

//Launch will launch a kernal that is in it.
func (k *Kernel) Launch(gx, gy, gz, bx, by, bz, shared uint32, stream gocu.Streamer, args ...interface{}) error {
	var shold unsafe.Pointer
	if stream == nil {
		shold = nil
	} else {
		shold = stream.Ptr()
	}

	cargs := makelaunchargs(len(args))
	err := k.ifacetounsafecomplete(args, cargs)
	if err != nil {
		return err
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
			(C.CUstream)(shold),
			&cargs.args[0],
			C.voiddptrnull,
		))

}

/*
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
	cargs = args
}

//GetArguments returns the empty interface array of arguments
func (k *KernelArguments) GetArguments() []interface{} {
	return cargs
}
*/
func isconvertable(gotype interface{}) bool {
	switch gotype.(type) {
	case float64:
		return true
	case float32:
		return true
	case int32:
		return true
	case int8:
		return true
	case uint8:
		return true
	case uint32:
		return true
	case half.Float16:
		return true
	default:
		return false
	}
}

func makelaunchargs(nelements int) (largs *launchargs) {

	if nelements < 1 {
		return nil
	}
	largs = new(launchargs)
	largs.args = make([]unsafe.Pointer, nelements)
	for i := range largs.args {
		largs.args[i] = (unsafe.Pointer)(C.malloc(C.maxArgSize))
	}

	runtime.SetFinalizer(largs, destroylargs)
	return largs
}
func (k *Kernel) ifacetounsafefirst(args []interface{}) (largs *launchargs, err error) {

	largs.args = make([]unsafe.Pointer, len(args))
	runtime.SetFinalizer(largs, destroylargs)
	for i := range args {
		largs.args[i] = unsafe.Pointer(C.malloc(C.maxArgSize))

		switch x := args[i].(type) {
		case nil:
			C.memcpy(largs.args[i], unsafe.Pointer(&x), C.ptrSize) //This might need to be (C.voiddptrnull)

		case cutil.Mem:
			if x == nil {
				C.memcpy(largs.args[i], unsafe.Pointer(&x), C.ptrSize)
			}
			C.memcpy(largs.args[i], unsafe.Pointer(x.DPtr()), C.ptrSize)
		case []cutil.Mem:
			{
				if x == nil {
					C.memcpy(largs.args[i], unsafe.Pointer(nil), C.ptrSize)
				}
				C.memcpy(largs.args[i], unsafe.Pointer(x[0].DPtr()), C.ptrSize)
			}
		case bool:
			if x {
				val := C.int(255)
				C.memcpy(largs.args[i], unsafe.Pointer(&val), C.size_t(4))
			} else {
				val := C.int(0)
				C.memcpy(largs.args[i], unsafe.Pointer(&val), C.size_t(4))
			}
		case int:
			val := C.int(x)
			C.memcpy(largs.args[i], unsafe.Pointer(&val), C.size_t(4))
		case uint:
			val := C.uint(x)
			C.memcpy(largs.args[i], unsafe.Pointer(&val), C.size_t(4))
		case []int32:
			C.memcpy(largs.args[i], unsafe.Pointer(&x[0]), C.size_t(4))
		default:
			scalar := cutil.CScalarConversion(x)
			if scalar == nil {
				return nil, fmt.Errorf("Kernel Launch - type %T not supported .. %+v", x, x)
			}

			/*
				val := reflect.ValueOf(x)

				sizeof := reflect.TypeOf(x).Size()
				y := unsafe.Pointer(val.Pointer())

				C.memcpy(cargs[i], y, (C.size_t)(sizeof))
			*/
			C.memcpy(largs.args[i], scalar.CPtr(), C.size_t(scalar.SIB()))
		}

	}

	return largs, nil
}
func (k *Kernel) ifacetounsafecomplete(args []interface{}, largs *launchargs) (err error) {

	for i := range args {

		switch x := args[i].(type) {
		case nil:
			C.memcpy(largs.args[i], unsafe.Pointer(&x), C.ptrSize) //This might need to be (C.voiddptrnull)

		case cutil.Mem:
			if x == nil {
				C.memcpy(largs.args[i], unsafe.Pointer(&x), C.ptrSize)
			}
			C.memcpy(largs.args[i], unsafe.Pointer(x.DPtr()), C.ptrSize)
		case []cutil.Mem:
			{
				if x == nil {
					C.memcpy(largs.args[i], unsafe.Pointer(nil), C.ptrSize)
				}
				C.memcpy(largs.args[i], unsafe.Pointer(x[0].Ptr()), C.ptrSize)
			}
		case bool:
			if x {
				val := C.int(255)
				C.memcpy(largs.args[i], unsafe.Pointer(&val), C.size_t(4))
			} else {
				val := C.int(0)
				C.memcpy(largs.args[i], unsafe.Pointer(&val), C.size_t(4))
			}
		case int:
			val := C.int(x)
			C.memcpy(largs.args[i], unsafe.Pointer(&val), C.size_t(4))
		case uint:
			val := C.uint(x)
			C.memcpy(largs.args[i], unsafe.Pointer(&val), C.size_t(4))
		case []int32:
			C.memcpy(largs.args[i], unsafe.Pointer(&x[0]), C.size_t(4))
		default:
			/*
					val := reflect.ValueOf(x)
					sizeof := reflect.TypeOf(x).Size()
					y := unsafe.Pointer(val.Pointer())

					C.memcpy(cargs[i], y, (C.size_t)(sizeof))

				}
			*/
			scalar := cutil.CScalarConversion(x)
			if scalar == nil {
				return fmt.Errorf("Kernel Launch - type %T not supported .. %+v", x, x)
			}

			C.memcpy(largs.args[i], scalar.CPtr(), C.size_t(scalar.SIB()))
		}
	}
	return nil
}
func destroylargs(l *launchargs) {
	for i := range l.args {
		C.free(l.args[i])
	}
}

/*
func destroyargs(args []unsafe.Pointer) {
	for i := range args {
		if args[i] != nil {
			C.free(args[i])
		}

	}
}
*/
/*
func (k *Kernel) destroyargs() {
	for i := range cargs {
		C.free(cargs[i])
	}
}
*/
func destroycudakernel(k *Kernel) {

	//k.destroyargs()
}
