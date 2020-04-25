package cudart

/*
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
typedef struct cudaPointerAttributes cudaPointerAttributes;
typedef enum cudaMemoryType cudaMemoryType;
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/cutil"
)

//Array is a cudaArray_t
type Array struct {
	c C.cudaArray_t
}

//Extent is a cuda struct cudaExtent
type Extent C.struct_cudaExtent

func (e Extent) c() C.struct_cudaExtent {
	return C.struct_cudaExtent(e)
}

//MakeCudaExtent -returns a cudaExtent based on input parameters.
func MakeCudaExtent(w, h, d uint) Extent {
	return (Extent)(C.make_cudaExtent((C.size_t)(w), (C.size_t)(h), (C.size_t)(d)))
}

//Width returns e.width
func (e Extent) Width() uint {
	return (uint)(e.width)
}

//Height returns e.height
func (e Extent) Height() uint {
	return (uint)(e.height)
}

//Depth returns e.depth
func (e Extent) Depth() uint {
	return (uint)(e.depth)
}

//Pos is a cuda struct cudaPos
type Pos C.struct_cudaPos

func (p Pos) c() C.struct_cudaPos {
	return C.struct_cudaPos(p)
}

//MakeCudaPos returns a cudaPos based on input parameters.
func MakeCudaPos(x, y, z uint) Pos {
	return (Pos)(C.make_cudaPos((C.size_t)(x), (C.size_t)(y), (C.size_t)(z)))
}

//X returns x position
func (p Pos) X() uint {
	return (uint)(p.x)
}

//Y returns y position
func (p Pos) Y() uint {
	return (uint)(p.y)
}

//Z returns z position
func (p Pos) Z() uint {
	return (uint)(p.z)
}

//PitchedPtr is a cudaPitchedPtr
type PitchedPtr C.struct_cudaPitchedPtr

//MakeCudaPitchedPtr makes a pitched pointer
func MakeCudaPitchedPtr(ptr cutil.Pointer, pitch, xsize, ysize uint) PitchedPtr {
	return (PitchedPtr)(C.make_cudaPitchedPtr(ptr.Ptr(), (C.size_t)(pitch), (C.size_t)(xsize), (C.size_t)(ysize)))
}

func (p PitchedPtr) c() C.struct_cudaPitchedPtr {
	return (C.struct_cudaPitchedPtr)(p)
}

//Pointer returns the ptiched pointer
func (p PitchedPtr) Pointer() cutil.Pointer {
	return gocu.WrapUnsafe(p.ptr)
}
func (p *PitchedPtr) cptr() *C.struct_cudaPitchedPtr {
	return (*C.struct_cudaPitchedPtr)(p)
}

//Ptr satisfies the cutil.Pointer interface
func (p *PitchedPtr) Ptr() unsafe.Pointer {
	return p.ptr
}

//Pitch returns the pitch
func (p PitchedPtr) Pitch() uint {
	return (uint)(p.pitch)
}

//Xsize returns the xsize
func (p PitchedPtr) Xsize() uint {
	return (uint)(p.xsize)
}

//Ysize returns the ysize
func (p PitchedPtr) Ysize() uint {
	return (uint)(p.ysize)
}

//ChannelFormatDesc describes a channels format
type ChannelFormatDesc C.struct_cudaChannelFormatDesc

//CreateChannelFormatDesc - Returns a channel descriptor with format f and number of bits of each component x, y, z, and w.
//
//So a float needs to be 32bits.
//
//unsigned is 8 ,32 bits
//
//signed is 8,32  bits
func CreateChannelFormatDesc(x, y, z, w int32, f ChannelFormatKind) ChannelFormatDesc {
	return (ChannelFormatDesc)(C.cudaCreateChannelDesc(
		(C.int)(x),
		(C.int)(y),
		(C.int)(z),
		(C.int)(w),
		f.c()))

}
func (c ChannelFormatDesc) c() C.struct_cudaChannelFormatDesc {
	return (C.struct_cudaChannelFormatDesc)(c)
}
func (c *ChannelFormatDesc) cptr() *C.struct_cudaChannelFormatDesc {
	return (*C.struct_cudaChannelFormatDesc)(c)
}

//ArrayFlag are flags used for array
type ArrayFlag C.uint

func (a ArrayFlag) c() C.uint {
	return (C.uint)(a)
}

//Default - This flag's value is defined to be 0 and provides default array allocation
func (a *ArrayFlag) Default() ArrayFlag {
	*a = (ArrayFlag)(C.cudaArrayDefault)
	return *a
}

//Layered - Allocates a layered CUDA array, with the depth extent indicating the number of layers
func (a *ArrayFlag) Layered() ArrayFlag {
	*a = (ArrayFlag)(C.cudaArrayLayered)
	return *a
}

//Cubemap - Allocates a cubemap CUDA array. Width must be equal to height, and depth must be six.
//If the cudaArrayLayered flag is also set, depth must be a multiple of six.
func (a *ArrayFlag) Cubemap() ArrayFlag {
	*a = (ArrayFlag)(C.cudaArrayCubemap)
	return *a
}

//SurfaceLoadStore - Allocates a CUDA array that could be read from or written to using a surface reference.
func (a *ArrayFlag) SurfaceLoadStore() ArrayFlag {
	*a = (ArrayFlag)(C.cudaArraySurfaceLoadStore)
	return *a
}

//TextureGather -  This flag indicates that texture gather operations will be performed on the CUDA array.
//Texture gather can only be performed on 2D CUDA arrays.
func (a *ArrayFlag) TextureGather() ArrayFlag {
	*a = (ArrayFlag)(C.cudaArrayTextureGather)
	return *a
}

//ChannelFormatKind is the kind of format the channel is in
type ChannelFormatKind C.enum_cudaChannelFormatKind

//Signed - sets the channel format to Signed
func (c *ChannelFormatKind) Signed() ChannelFormatKind {
	*c = (ChannelFormatKind)(C.cudaChannelFormatKindSigned)
	return *c
}

//UnSigned - sets the channel format to UnSigned
func (c *ChannelFormatKind) UnSigned() ChannelFormatKind {
	*c = (ChannelFormatKind)(C.cudaChannelFormatKindUnsigned)
	return *c
}

//Float - sets the channel format to Float
func (c *ChannelFormatKind) Float() ChannelFormatKind {
	*c = (ChannelFormatKind)(C.cudaChannelFormatKindFloat)
	return *c
}

// cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, or cudaChannelFormatKindFloat
func (c ChannelFormatKind) c() C.enum_cudaChannelFormatKind {
	return (C.enum_cudaChannelFormatKind)(c)
}

//Malloc3dArray - Allocate an array on the device.
func Malloc3dArray(a *Array, desc *ChannelFormatDesc, e Extent, flag ArrayFlag) error {

	err := newErrorRuntime("Malloc3dArray()",
		C.cudaMalloc3DArray(&a.c, (*C.struct_cudaChannelFormatDesc)(desc), e.c(), flag.c()))
	if err != nil {
		return err
	}
	runtime.SetFinalizer(a, freeArray)
	return nil
}

//MallocArray - Allocate an array on the device.
func MallocArray(a *Array, desc *ChannelFormatDesc, width, height uint, flag ArrayFlag) error {
	var cw, ch C.size_t
	cw = (C.size_t)(width)
	ch = (C.size_t)(height)
	err := newErrorRuntime("MallocArray()", C.cudaMallocArray(&a.c, desc.cptr(), cw, ch, flag.c()))
	if err != nil {
		return err
	}
	runtime.SetFinalizer(a, freeArray)
	return nil
}

//Malloc3D -	Allocates logical 1D, 2D, or 3D memory objects on the device.
func Malloc3D(p *PitchedPtr, e Extent) error {
	return newErrorRuntime("Malloc3D()", C.cudaMalloc3D(p.cptr(), e.c()))
}

//MallocManagedHost uses the Unified memory mangement system and starts it off in the host. Memory is set to 0.
//It will also set a finalizer on the memory for GC
func MallocManagedHost(mem cutil.Mem, size uint) error {
	var err error
	err = newErrorRuntime("MallocManagedHost()", C.cudaMallocManaged(mem.DPtr(), C.size_t(size), C.cudaMemAttachHost))
	if err != nil {
		return err
	}
	err = Memset(mem, 0, (size))
	if err != nil {
		return err
	}

	return nil
}

//MallocManagedHostEx is like MallocManagedHost but it takes a worker and memory allocated to mem will be allocated to the context being used on that host thread. If w is nil then it will behave like MallocManagedHost
func MallocManagedHostEx(w *gocu.Worker, mem cutil.Mem, size uint) error {
	var err error
	if w != nil {
		err = w.Work(func() error {
			err = newErrorRuntime("MallocManagedHostEx()", C.cudaMallocManaged(mem.DPtr(), C.size_t(size), C.cudaMemAttachHost))
			if err != nil {
				return err
			}
			return Memset(mem, 0, (size))
		})
	} else {
		err = newErrorRuntime("MallocManagedHostEx()", C.cudaMallocManaged(mem.DPtr(), C.size_t(size), C.cudaMemAttachHost))
		if err != nil {
			return err
		}
		err = Memset(mem, 0, (size))

	}
	if err != nil {
		return err
	}
	runtime.SetFinalizer(mem, hostfreemem)
	return nil
}
func freeArray(a *Array) error {

	err := newErrorRuntime("freeArray", C.cudaFreeArray(a.c))
	if err != nil {
		return err
	}
	a = nil
	return nil
}

/*
//MallocManagedHostUS is like MallocManaged but using unsafe.Pointer
func MallocManagedHostUS(mem unsafe.Pointer, size uint) error {
	err := newErrorRuntime("MallocManaged", C.cudaMallocManaged(&mem, C.size_t(size), C.uint(2)))
	runtime.SetFinalizer(mem, hostfreememUS)
	return err

}
*/

//MallocManagedGlobal Allocates memory on current devices.
func MallocManagedGlobal(mem cutil.Mem, size uint) error {
	err := newErrorRuntime("MallocManagedGlobal", C.cudaMallocManaged(mem.DPtr(), C.size_t(size), C.cudaMemAttachGlobal))
	if err != nil {
		return err
	}
	err = Memset(mem, 0, (size))
	if err != nil {
		return err
	}
	runtime.SetFinalizer(mem, devicefreemem)
	return nil
}

//MallocManagedGlobalEx is like MallocManagedGlobal but it takes a worker and memory allocated
//to mem will be allocated to the context being used on that host thread.
//If w is nil then it will behave like MallocManagedGlobal
func MallocManagedGlobalEx(w *gocu.Worker, mem cutil.Mem, size uint) error {
	var err error
	if w != nil {
		err = w.Work(func() error {
			err = newErrorRuntime("MallocManagedGlobalEx", C.cudaMallocManaged(mem.DPtr(), C.size_t(size), C.cudaMemAttachGlobal))
			if err != nil {
				return err
			}
			return Memset(mem, 0, (size))
		})
	} else {
		err = newErrorRuntime("MallocManagedGlobalEx", C.cudaMallocManaged(mem.DPtr(), C.size_t(size), C.cudaMemAttachGlobal))
		if err != nil {
			return err
		}
		err = Memset(mem, 0, (size))
	}

	if err != nil {
		return err
	}
	runtime.SetFinalizer(mem, devicefreemem)
	return nil
}

/*
//MallocManagedGlobalUS is like MallocManagedGlobal but uses unsafe.Pointer
func MallocManagedGlobalUS(mem unsafe.Pointer, size uint) error {
	err := newErrorRuntime("MallocManaged", C.cudaMallocManaged(&mem, C.size_t(size), C.uint(1)))
	runtime.SetFinalizer(mem, devicefreememUS)
	return err
}
*/

//Malloc will allocate memory to the device the size that was passed.
//It will also set the finalizer for GC
func Malloc(mem cutil.Mem, sizet uint) error {
	err := newErrorRuntime("Malloc", C.cudaMalloc(mem.DPtr(), C.size_t(sizet)))
	if err != nil {
		return err
	}
	err = Memset(mem, 0, (sizet))
	if err != nil {
		return err
	}
	runtime.SetFinalizer(mem, devicefreemem)
	return nil
}

//MallocEx is like Malloc but it takes a worker and memory allocated
//to mem will be allocated to the context being used on that host thread.
//If w is nil then it will behave like Malloc
func MallocEx(w *gocu.Worker, mem cutil.Mem, sizet uint) error {
	var err error
	if w != nil {
		err = w.Work(func() error {
			err = newErrorRuntime("MallocEx", C.cudaMalloc(mem.DPtr(), C.size_t(sizet)))
			if err != nil {
				return err
			}
			return Memset(mem, 0, (sizet))
		})
	} else {
		err = newErrorRuntime("MallocEx", C.cudaMalloc(mem.DPtr(), C.size_t(sizet)))
		if err != nil {
			return err
		}
		err = Memset(mem, 0, (sizet))
	}

	if err != nil {
		return err
	}
	runtime.SetFinalizer(mem, devicefreemem)
	return nil

}

//MallocHost will allocate memory on the host for cuda use.
//
func MallocHost(mem cutil.Mem, sizet uint) error {
	err := newErrorRuntime("MallocHost", C.cudaMalloc(mem.DPtr(), C.size_t(sizet)))
	if err != nil {
		return err
	}
	err = Memset(mem, 0, (sizet))
	if err != nil {
		return err
	}
	runtime.SetFinalizer(mem, hostfreemem)
	return err
}

//MallocHostEx is like MallocHost but it takes a worker and memory allocated
//to mem will be allocated to the context being used on that host thread.
//If w is nil then it will behave like MallocHost
func MallocHostEx(w *gocu.Worker, mem cutil.Mem, sizet uint) error {

	var err error
	if w != nil {
		err = w.Work(func() error {
			err = newErrorRuntime("MallocHostEx", C.cudaMallocHost(mem.DPtr(), C.size_t(sizet)))
			if err != nil {
				return err
			}
			return Memset(mem, 0, (sizet))
		})
	} else {
		err = newErrorRuntime("MallocHostEx", C.cudaMallocHost(mem.DPtr(), C.size_t(sizet)))
		if err != nil {
			return err
		}
		err = Memset(mem, 0, (sizet))
	}

	if err != nil {
		return err
	}
	runtime.SetFinalizer(mem, hostfreemem)
	return err
}

//PointerGetAttributes returns the atributes
func PointerGetAttributes(mem cutil.Pointer) (Atribs, error) {
	var x C.cudaPointerAttributes
	cuerr := C.cudaPointerGetAttributes(&x, mem.Ptr())
	err := newErrorRuntime("Attributes", cuerr)
	if err != nil {
		return Atribs{}, err
	}
	var managed bool
	if x.isManaged > C.int(0) {
		managed = true
	}

	return Atribs{
		Type:    MemType(x.memoryType),
		Device:  int32(x.device),
		DPtr:    unsafe.Pointer(x.devicePointer),
		HPtr:    unsafe.Pointer(x.hostPointer),
		Managed: managed,
	}, nil
}

//MemsetUS is like Memset but with unsafe.pointer
func MemsetUS(mem unsafe.Pointer, value int32, count uint) error {
	err := C.cudaMemset(mem, C.int(value), C.size_t(count))

	return newErrorRuntime("cudaMemset", err)
}

//Memset sets the value for each byte in device memory
func Memset(mem cutil.Mem, value int32, count uint) error {
	err := C.cudaMemset(mem.Ptr(), C.int(value), C.size_t(count))

	return newErrorRuntime("cudaMemset", err)
}

//Atribs are a memories attributes on the device side
type Atribs struct {
	Type    MemType
	Device  int32
	DPtr    unsafe.Pointer
	HPtr    unsafe.Pointer
	Managed bool
}

//MemType is a typedefed C.cudaMemoryType
type MemType C.cudaMemoryType

/*

finalizer functions

*/

func devicefreemem(mem cutil.Mem) error {

	err := newErrorRuntime("devicefree", C.cudaFree(mem.Ptr()))
	if err != nil {
		return nil
	}
	mem = nil
	return nil
}
func devicefreememUS(mem unsafe.Pointer) error {

	err := newErrorRuntime("devicefree", C.cudaFree(mem))
	if err != nil {
		return nil
	}
	mem = nil
	return nil
}
func hostfreememUS(mem unsafe.Pointer) error {
	err := newErrorRuntime("hostfree", C.cudaFreeHost(mem))
	if err != nil {
		return err
	}
	mem = nil
	return nil
}
func hostfreemem(mem cutil.Mem) error {
	err := newErrorRuntime("hostfree", C.cudaFreeHost(mem.Ptr()))
	if err != nil {
		return err
	}
	mem = nil
	return nil
}
