package main

/*
#include <cudnn.h>
*/
import "C"

import (
	"fmt"
	"runtime"

	"github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/kernels"
)

func main() {

	runtime.LockOSThread()
	trainingkernellocation := "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/"
	memsize := 64
	//cudnn context
	var cuda gocudnn.Cuda

	devices, err := cuda.GetDeviceList()
	devices[0].Set()
	devices[0].Reset()

	if err != nil {
		panic(err)
	}

	devicenum := len(devices)
	fmt.Println("Number of Devices:", devicenum)

	if err != nil {
		panic(err)
	}
	//	ctx, err := cuda.CtxCreate(4, devices[0])

	if err != nil {
		panic(err)
	}
	//	err = ctx.Set()
	if err != nil {
		panic(err)
	}

	bytesize := gocudnn.SizeT(memsize * 4)

	x, err := gocudnn.MallocManaged(bytesize, gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		panic(err)
	}
	some := floatsizedw(memsize)
	gptr, err := gocudnn.MakeGoPointer(some)
	if err != nil {
		panic(err)
	}
	err = gocudnn.CudaMemCopy(x, gptr, bytesize, gocudnn.MemcpyKindFlag{}.Default())
	if err != nil {
		panic(err)
	}
	dx, err := gocudnn.MallocManaged(bytesize, gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		panic(err)
	}
	some2 := floatsizew(memsize)
	gptr1, err := gocudnn.MakeGoPointer(some2)
	if err != nil {
		panic(err)
	}
	err = gocudnn.CudaMemCopy(dx, gptr1, bytesize, gocudnn.MemcpyKindFlag{}.Default())

	kernelname := kernels.MakeMakeFile(trainingkernellocation, "trainingfloat.cu", devices[0])
	mod, err := cuda.NewModule(trainingkernellocation + kernelname)
	if err != nil {
		panic(err)
	}
	nsize := uint32(memsize)
	blocksize := uint32(32)
	gridsize := ((nsize - 1) / blocksize) + 1
	kern, err := cuda.MakeKernel("copyto", mod)
	if err != nil {
		panic(err)
	}

	err = kern.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, &gocudnn.Stream{}, nsize, x, dx)
	if err != nil {
		panic(err)
	}
	err = cuda.CtxSynchronize()
	if err != nil {
		panic(err)
	}
	filler := make([]float32, memsize)

	cudavalues, err := gocudnn.MakeGoPointer(filler)
	if err != nil {
		panic(err)
	}
	err = gocudnn.CudaMemCopy(cudavalues, x, cudavalues.ByteSize(), gocudnn.MemcpyKindFlag{}.Default())
	if err != nil {
		panic(err)
	}
	fmt.Println(filler)
	//fmt.Println(gptr1, gptr2)
	/*
		trainhandle, err := gocudnn.Xtra{}.MakeTrainingHandle(trainingkernellocation, devices[0])
		if err != nil {
			panic(err)
		}
		traind, err := gocudnn.Xtra{}.NewTrainingDescriptor(
			trainhandle,
			gocudnn.TrainingModeFlag{}.Adam(),
			gocudnn.DataTypeFlag{}.Float(),
			gocudnn.RegularizationFlag{}.L1L2(),
		)
	*/
	//trainhandle.SetStream(stream)

	/*
		l1, err := gocudnn.MallocManaged(gocudnn.SizeT(4), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			panic(err)
		}
		l2, err := gocudnn.MallocManaged(gocudnn.SizeT(4), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			panic(err)
		}

	*/
	/*
		gsum, err := makez(memsize)
		if err != nil {
			panic(err)
		}
		xsum, err := makez(memsize)
		if err != nil {
			panic(err)
		}
	*/

	//err = stream.Sync()
	/*
		params := gocudnn.Xtra{}.CreateParamsFloat32(float32(.00001), float32(.0001), float32(20), float32(1e-8), float32(.001), float32(.9), float32(.999))
		err = traind.TrainValues(trainhandle, 128, dx, x, l1, l2, gsum, xsum, params)
		if err != nil {
			panic(err)
		}
		//	err = stream.Sync()
		if err != nil {
			panic(err)
		}
	*/

	/*
		float32slice := make([]float32, memsize)
		err = cuda.CtxSynchronize()
		if err != nil {
			panic(err)
			//	fmt.Println(float32slice)
		}
		err = gsum.FillSlice(float32slice)
		if err != nil {
			panic(err)

		}
	*/
	//	fmt.Println(float32slice)
	/*
		xsumdesc.DestroyDescriptor()
		gsumdesc.DestroyDescriptor()
		l1.Free()
		l2.Free()
		gsum.Free()
		xsum.Free()
		xdesc.DestroyDescriptor()
		dxdesc.DestroyDescriptor()
		x.Free()
		dx.Free()
	*/
	//	runtime.UnlockOSThread()
	runtime.UnlockOSThread()
}

func makedw(size int) (mem *gocudnn.Malloced, gptr *gocudnn.GoPointer, err error) {
	bytesize := gocudnn.SizeT(size * 4)
	mem, err = gocudnn.Malloc(bytesize)
	if err != nil {
		return nil, nil, err
	}
	some := floatsizedw(size)
	gptr, err = gocudnn.MakeGoPointer(some)
	if err != nil {
		return nil, nil, err
	}
	err = gocudnn.CudaMemCopy(mem, gptr, bytesize, gocudnn.MemcpyKindFlag{}.HostToDevice())
	return mem, gptr, err
}
func makew(size int) (mem *gocudnn.Malloced, gptr *gocudnn.GoPointer, err error) {
	bytesize := gocudnn.SizeT(size * 4)
	mem, err = gocudnn.Malloc(bytesize)
	if err != nil {
		return nil, nil, err
	}
	some := floatsizew(size)
	gptr, err = gocudnn.MakeGoPointer(some)
	if err != nil {
		return nil, nil, err
	}
	err = gocudnn.CudaMemCopy(mem, gptr, bytesize, gocudnn.MemcpyKindFlag{}.HostToDevice())
	return mem, gptr, err
}
func makez(size int) (mem *gocudnn.Malloced, gptr *gocudnn.GoPointer, err error) {
	bytesize := gocudnn.SizeT(size * 4)
	mem, err = gocudnn.Malloc(bytesize)
	if err != nil {
		return nil, nil, err
	}
	some := floatsizezero(size)
	gptr, err = gocudnn.MakeGoPointer(some)
	if err != nil {
		return nil, nil, err
	}
	err = gocudnn.CudaMemCopy(mem, gptr, bytesize, gocudnn.MemcpyKindFlag{}.HostToDevice())
	return mem, gptr, err
}

/*
func maketestfilterW() (*gocudnn.FilterD, *gocudnn.Malloced, error) {
	dtf := gocudnn.DataTypeFlag{}.Float()
	tff := gocudnn.TensorFormatFlag{}.NCHW()

	shape := gocudnn.Tensor{}.Shape //shape function
	FilterD, err := gocudnn.Filter{}.NewFilter4dDescriptor(dtf, tff, shape(20, 20, 20, 20))
	size, err := FilterD.TensorD().GetSizeInBytes()
	if err != nil {
		return nil, nil, err
	}
	cudamem, err := gocudnn.MallocManaged(size, gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		return nil, nil, err
	}
	goslice := floatslicew(20, 20, 20, 20)
	gomem, err := gocudnn.MakeGoPointer(goslice)
	if err != nil {
		return nil, nil, err
	}
	err = gocudnn.CudaMemCopy(cudamem, gomem, size, gocudnn.MemcpyKindFlag{}.Default())
	if err != nil {
		return nil, nil, err
	}
	return FilterD, cudamem, nil

}

func maketestfilterDW() (*gocudnn.FilterD, *gocudnn.Malloced, error) {
	dtf := gocudnn.DataTypeFlag{}.Float()
	tff := gocudnn.TensorFormatFlag{}.NCHW()

	shape := gocudnn.Tensor{}.Shape //shape function
	FilterD, err := gocudnn.Filter{}.NewFilter4dDescriptor(dtf, tff, shape(20, 20, 20, 20))
	size, err := FilterD.TensorD().GetSizeInBytes()
	if err != nil {
		return nil, nil, err
	}
	cudamem, err := gocudnn.MallocManaged(size, gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		return nil, nil, err
	}
	goslice := floatslicedw(20, 20, 20, 20)
	gomem, err := gocudnn.MakeGoPointer(goslice)
	if err != nil {
		return nil, nil, err
	}
	err = gocudnn.CudaMemCopy(cudamem, gomem, size, gocudnn.MemcpyKindFlag{}.Default())
	if err != nil {
		return nil, nil, err
	}
	return FilterD, cudamem, nil

}
func maketestxgsum() (*gocudnn.FilterD, *gocudnn.Malloced, error) {

	dtf := gocudnn.DataTypeFlag{}.Float()
	tff := gocudnn.TensorFormatFlag{}.NCHW()

	shape := gocudnn.Tensor{}.Shape //shape function
	FilterD, err := gocudnn.Filter{}.NewFilter4dDescriptor(dtf, tff, shape(20, 20, 20, 20))
	size, err := FilterD.TensorD().GetSizeInBytes()
	if err != nil {
		return nil, nil, err
	}
	cudamem, err := gocudnn.MallocManaged(size, gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		return nil, nil, err
	}
	goslice := floatslicezero(20, 20, 20, 20)
	gomem, err := gocudnn.MakeGoPointer(goslice)
	if err != nil {
		return nil, nil, err
	}
	err = gocudnn.CudaMemCopy(cudamem, gomem, size, gocudnn.MemcpyKindFlag{}.Default())
	if err != nil {
		return nil, nil, err
	}
	return FilterD, cudamem, nil

}
*/
func floatsizezero(size int) []float32 {

	array := make([]float32, size)
	for i := 0; i < size; i++ {
		array[i] = 0.0
	}
	return array

}
func floatsizedw(size int) []float32 {

	array := make([]float32, size)
	for i := 0; i < size; i++ {
		array[i] = float32(1.0) / float32((i%6)-3) //just some patterns
	}
	return array

}

func floatsizew(size int) []float32 {
	array := make([]float32, size)
	for i := 0; i < size; i++ {
		array[i] = float32((i % 6) - 3) //just some patterns
	}
	return array
}

func floatslicezero(dims ...int) []float32 {
	mult := 1
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	array := make([]float32, mult)
	for i := 0; i < mult; i++ {
		array[i] = 0.0
	}
	return array

}
func floatslicedw(dims ...int) []float32 {
	mult := 1
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	array := make([]float32, mult)
	for i := 0; i < mult; i++ {
		array[i] = float32(1.0) / float32((i%6)-3) //just some patterns
	}
	return array
}

func floatslicew(dims ...int) []float32 {
	mult := 1
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	array := make([]float32, mult)
	for i := 0; i < mult; i++ {
		array[i] = float32((i % 6) - 3) //just some patterns
	}
	return array
}
