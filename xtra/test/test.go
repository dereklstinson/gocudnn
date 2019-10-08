package main

import (
	//"github.com/dereklstinson/GoCudnn/gocu"
	//"github.com/dereklstinson/GoCudnn/cudart"
	"fmt"
	"github.com/dereklstinson/GoCudnn/cuda"
	"github.com/dereklstinson/GoCudnn/nvrtc"
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
	//fmt.Println(ptx2)
	fmt.Println(kern2)
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
