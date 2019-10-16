package main

import (
	"fmt"
	"github.com/dereklstinson/GoCudnn/cuda"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/nvrtc"
	"github.com/dereklstinson/cutil"
	"github.com/dereklstinson/half"
)

const cudapath = "/usr/local/cuda/include"
const cudaname = "cuda_fp16.h"

func main() {

	check := func(e error) {
		if e != nil {
			panic(e)
		}
	}
	cuda.LockHostThread()

	//dev,err:=cudart.CreateDevice(0)
	//check(err)
	devs, err := cuda.GetDeviceList()
	check(err)
	ctx, err := cuda.CtxCreate(-1, devs[0])
	ctx.Set()
	check(err)
	fmt.Println(devs[0].Major())

	p := program()
	err = p.AddNameExpression("testfma")
	check(err)
	err = p.Compile()

	if err != nil {
		log, err2 := p.GetLog()
		check(err2)
		fmt.Println("Log:" + log)
		panic(err)
	}
	ln, err := p.GetLoweredName("testfma")
	check(err)
	fmt.Println(ln)
	ptx, err := p.PTX()
	check(err)
	ptx, err = p.PTX()
	check(err)
	//fmt.Println(string(ptx))

	m1, err := cuda.NewModuleData(ptx)
	check(err)
	kern1, err := cuda.MakeKernel("testfma", m1)
	check(err)
	fmt.Println(kern1)
	p2 := program2()
	err = p2.AddNameExpression("testfma2")
	check(err)
	err = p2.Compile("--gpu-architecture=compute_75", "-I/usr/local/cuda/include")
	if err != nil {
		log, err2 := p2.GetLog()
		check(err2)
		fmt.Println("Log:" + log)
		panic(err)
	}
	ln2, err := p2.GetLoweredName("testfma2")
	check(err)
	fmt.Println(ln2)
	ptx2, err := p2.PTX()
	check(err)
	err = m1.LoadData(ptx2)
	check(err)
	kern2, err := cuda.MakeKernel("testfma2", m1)
	check(err)
	fmt.Println(ptx2)
	fmt.Println(kern2)
	arraylen := 1024 * 1024 * 10
	one := newArray(arraylen, 1)
	two := newArray(arraylen, 2)
	three := newArray(arraylen, 3)
	equals := newArray(arraylen, 0)

	hone := half.NewFloat16Array(one)
	htwo := half.NewFloat16Array(two)
	hthree := half.NewFloat16Array(three)
	hequals := half.NewFloat16Array(equals)

	wrappers1 := make([]*cutil.Wrapper, 4)

	wrappers1[0], _ = cutil.WrapGoMem(one)
	wrappers1[1], _ = cutil.WrapGoMem(two)
	wrappers1[2], _ = cutil.WrapGoMem(three)
	wrappers1[3], _ = cutil.WrapGoMem(equals)
	spsib := uint(arraylen * 4)
	wrappers2 := make([]*cutil.Wrapper, 4)
	wrappers2[0], err = cutil.WrapGoMem(hone)
	check(err)
	wrappers2[1], err = cutil.WrapGoMem(htwo)
	check(err)
	wrappers2[2], err = cutil.WrapGoMem(hthree)
	check(err)
	wrappers2[3], err = cutil.WrapGoMem(hequals)
	check(err)
	hpsib := uint(arraylen * 2)
	gpu1 := make([]*gocu.CudaPtr, 4)
	gpu2 := make([]*gocu.CudaPtr, 4)
	var mckf cudart.MemcpyKind
	for i := range gpu1 {
		gpu1[i] = new(gocu.CudaPtr)
		gpu2[i] = new(gocu.CudaPtr)
		check(cudart.MallocManagedGlobal(gpu1[i], spsib))
		check(cudart.MallocManagedGlobal(gpu2[i], hpsib))
		check(cudart.MemCpy(gpu1[i], wrappers1[i], spsib, mckf.Default()))
		check(cudart.MemCpy(gpu2[i], wrappers2[i], hpsib, mckf.Default()))
	}
	stream, err := cudart.CreateNonBlockingStream()
	lconhelper, err := gocu.CreateConfigHelper(devs[0])
	check(err)
	lconfig := lconhelper.CreateLaunchConfig(int32(arraylen), 1, 1)
	event, err := cudart.CreateEvent()
	check(err)
	event2, err := (cudart.CreateEvent())
	check(err)
	check(event.Record(stream))
	check(event.Sync())
	check(kern1.Launch(
		lconfig.ThreadPerBlockx,
		lconfig.ThreadPerBlocky,
		lconfig.ThreadPerBlockz,
		lconfig.BlockCountx,
		lconfig.BlockCounty,
		lconfig.BlockCountz, 0, stream, lconfig.Dimx, gpu1[0], gpu1[1], gpu1[2], gpu1[3]))
	//check(event.Sync())
	check(event2.Record(stream))
	check(event2.Sync())
	time, err := event2.ElapsedTime(event)
	check(err)
	fmt.Println("elapsed time 1", time)
	check(event.Record(stream))
	check(event.Sync())
	check(kern2.Launch(
		lconfig.ThreadPerBlockx,
		lconfig.ThreadPerBlocky,
		lconfig.ThreadPerBlockz,
		lconfig.BlockCountx,
		lconfig.BlockCounty,
		lconfig.BlockCountz, 0, stream, lconfig.Dimx, gpu2[0], gpu2[1], gpu2[2], gpu2[3]))
	//check(event.Sync())
	check(event2.Record(stream))
	check(event2.Sync())
	time, err = event2.ElapsedTime(event)
	check(err)
	fmt.Println("elapsed time 2", time)
	check(event.Record(stream))
	check(event.Sync())

	check(cudart.MemCpy(wrappers1[3], gpu1[3], spsib, mckf.Default()))
	check(event2.Record(stream))
	check(event2.Sync())
	time, err = event2.ElapsedTime(event)
	fmt.Println("elapsed copy 1", time)

	check(event.Record(stream))
	check(event.Sync())

	check(cudart.MemCpy(wrappers2[3], gpu2[3], hpsib, mckf.Default()))
	check(event2.Record(stream))
	check(event2.Sync())
	time, err = event2.ElapsedTime(event)
	fmt.Println("elapsed copy 2", time)
	//fmt.Println(equals)
	//fmt.Println(half.ToFloat32(hequals))
	convertedequals := half.ToFloat32(hequals)
	for i := range hequals {
		if convertedequals[i] != float32(5) {
			fmt.Println("Half no work")
		}
		if equals[i] != float32(5) {
			fmt.Println("Single Not work")
		}
	}
	//	cudart.MallocManagedGlobal()
	//	kern1.Launch((gx uint32), gy uint32, gz uint32, bx uint32, by uint32, bz uint32, shared uint32, stream gocu.Streamer, args ...interface{})
	//	kern2.Launch()
}

func program() *nvrtc.Program {
	p, err := nvrtc.CreateProgram(kernel, `testfma.cu`)
	if err != nil {
		panic(err)
	}
	return p
}

func program2() *nvrtc.Program {
	var headers nvrtc.Include
	headers.Name = cudaname
	headers.Source = cudapath
	p, err := nvrtc.CreateProgram(kernelhalf, `testfma2.cu`)
	if err != nil {
		panic(err)
	}
	return p
}
func newArray(length int, val float32) (array []float32) {
	array = make([]float32, length)
	for i := 0; i < length; i++ {
		array[i] = val
	}
	return array
}

const kernel = `
#define StartAxis(i,axis) int i = blockIdx.axis * blockDim.axis + threadIdx.axis;
#define CUDA_GRID_LOOP_X(i, n)                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

#define CUDA_GRID_AXIS_LOOP(i, n, axis)                                 \
	for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n; \
		 i += blockDim.axis * gridDim.axis)
		
		 extern "C" __global__
		void testfma(int n, float *a, float *b, float *c, float *d){
			CUDA_GRID_LOOP_X(i,n){
				d[i] =(a[i]*b[i])+c[i];
			    
			}
			}
	`

const kernelhalf = `
#include <cuda_fp16.h>
#define StartAxis(i,axis) int i = blockIdx.axis * blockDim.axis + threadIdx.axis;
#define CUDA_GRID_LOOP_X(i, n)                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)
#define CUDA_GRID_AXIS_LOOP(i, n, axis)                                 \
    for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n; \
		 i += blockDim.axis * gridDim.axis)
		extern "C" __global__
		void testfma2(int n, __half *a, __half *b, __half *c, __half *d){
			CUDA_GRID_LOOP_X(i,n){
				d[i]=(a[i]*b[i])+c[i];
			}
			}`
