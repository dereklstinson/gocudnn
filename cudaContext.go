package gocudnn

/*
#include <cuda.h>
*/
import "C"

//Context holds a CUcontext.
type Context struct {
	ctx C.CUcontext
}

/*
CtxCreate creates a context.  --Flags are listed below just pass the equivelant uint32 for the flag.  I know it is lazy, but cuda was kind of lazy in this respect, too.  Default to zero if you don't know what the stuff does.
CU_CTX_SCHED_AUTO = 0x00 -> uint32(0)
	Automatic scheduling

CU_CTX_SCHED_SPIN = 0x01 -> uint32(1)
    Set spin as default scheduling

CU_CTX_SCHED_YIELD = 0x02 -> uint32(2)
    Set yield as default scheduling

CU_CTX_SCHED_BLOCKING_SYNC = 0x04 -> uint32(4)
    Set blocking synchronization as default scheduling

CU_CTX_SCHED_MASK = 0x07  -> uint32(7)

CU_CTX_MAP_HOST = 0x08  -> uint32(8)
    Support mapped pinned allocations

CU_CTX_LMEM_RESIZE_TO_MAX = 0x10 -> uint32(16)
    Keep local memory allocation after launch

CU_CTX_FLAGS_MASK = 0x1f -> uint32(31)

*/

//CtxCreate creates a context with the flags on the device passed if -1 is passed in flags then it sets the default CU_CTX_SCHED_BLOCKING_SYNC = 0x04
func (cu Cuda) CtxCreate(flags int32, device *Device) (*Context, error) {

	var ctx C.CUcontext
	if flags == -1 {
		x := C.cuCtxCreate(&ctx, C.uint(4), device.id)

		err := newErrorDriver("cuCtxCreate", x)
		if err != nil {
			return nil, err
		}
		return &Context{
			ctx: ctx,
		}, nil
	}
	x := C.cuCtxCreate(&ctx, C.uint(flags), device.id)

	err := newErrorDriver("cuCtxCreate", x)
	if err != nil {
		return nil, err
	}
	return &Context{
		ctx: ctx,
	}, nil
}

//GetCurrentContext returns the context bound to the calling thread
func (cu Cuda) GetCurrentContext() (*Context, error) {
	var ctx C.CUcontext
	x := C.cuCtxGetCurrent(&ctx)

	err := newErrorDriver("cuCtxGetCurrent", x)
	if err != nil {
		return nil, err
	}
	return &Context{
		ctx: ctx,
	}, nil

}

//CurrentContextFlags will return the flags that are being used by the current host thread
func (cu Cuda) CurrentContextFlags() (uint32, error) {
	var flg C.uint
	err := newErrorDriver("cuCtxGetFlags", C.cuCtxGetFlags(&flg))
	return uint32(flg), err
}

//CtxPopCurrent -Pops the current CUDA context from the current CPU thread, and returns it.
//So, if you want to use it on another thread then that is ok.
//the previous context using this thread will now be the current context for the thread
func (cu Cuda) CtxPopCurrent() (*Context, error) {
	var ctx C.CUcontext
	err := newErrorDriver("cuCtxPopCurrent", C.cuCtxPopCurrent(&ctx))
	if err != nil {
		return nil, err
	}
	return &Context{
		ctx: ctx,
	}, nil
}

//CtxSynchronize synchronizes the current context
func (cu Cuda) CtxSynchronize() error {
	return newErrorDriver("cuCtxSynchronize", C.cuCtxSynchronize())
}

//Set sets/binds the context to the calling cpu thread. I think this pretty much performs a pop and push. w/o a returned popped context.
func (c *Context) Set() error {
	return newErrorDriver("cuCtxSetCurrent", C.cuCtxSetCurrent(c.ctx))
}

//Push pushes the context onto the cpu thread, and makes it the current context for that thread.
func (c *Context) Push() error {
	return newErrorDriver("cuCtxPushCurrent", C.cuCtxPushCurrent(c.ctx))
}

//Destroy destroys the context
func (c *Context) Destroy() error {

	return newErrorDriver("cuCtxDestroy", C.cuCtxDestroy(c.ctx))
}
