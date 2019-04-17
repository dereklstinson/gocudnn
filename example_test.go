package gocudnn_test

import (
	"fmt"
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"

	"github.com/dereklstinson/GoCudnn/cudart"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//ExampleActivationD of doing the activation function
func ExampleActivationD() {
	runtime.LockOSThread()
	check := func(e error) {
		if e != nil {
			panic(e)
		}
	}

	h := gocudnn.CreateHandle(true) //Using go garbage collector

	ActOp, err := gocudnn.CreateActivationDescriptor()
	check(err)

	var AMode gocudnn.ActivationMode //Activation Mode Flag
	var NanMode gocudnn.NANProp      //Nan Propagation Flag

	err = ActOp.Set(AMode.Relu(), NanMode.Propigate(), 20)
	check(err)
	am, nm, coef, err := ActOp.Get() //Gets the calues that where set
	if am != AMode.Relu() || nm != NanMode.Propigate() || coef != 20 {
		panic("am!=Amode.Relu()||nm !=NanMode.Propigate()||coef!=20")
	}

	//Dummy Variables
	//Check TensorD to find out how to make xD,yD and x and y
	var x, y *gocu.CudaPtr
	var xD, yD *gocudnn.TensorD

	err = ActOp.Forward(h, 1, xD, x, 0, yD, y)
	check(err)
}

//ExampleTensorD shows tomake a tensor
func ExampleTensorD() {
	//Need to lock os thread.
	runtime.LockOSThread()
	check := func(e error) {
		if e != nil {
			panic(e)
		}
	}
	//Creating a blocking stream
	cs, err := cudart.CreateBlockingStream()
	check(err)
	//Create Device
	dev, err := cudart.CreateDevice(1)
	check(err)

	//Make an Allocator
	CudaMemManager, err := cudart.CreateMemManager(cs, dev) //cs could be nil .  Check out cudart package on more about streams
	check(err)

	//Tensor
	var tflg gocudnn.TensorFormat //Flag for tensor
	var dtflg gocudnn.DataType    //Flag for tensor

	xD, err := gocudnn.CreateTensorDescriptor()

	// Setting Tensor
	err = xD.Set(tflg.NCHW(), dtflg.Float(), []int32{20, 1, 1, 1}, nil)
	check(err)

	//Gets SIB for tensor memory on device
	xSIB, err := xD.GetSizeInBytes()
	check(err)

	//Allocating memory to device and returning pointer to device memory
	x, err := CudaMemManager.Malloc(xSIB)

	//Create some host mem to copy to cuda memory
	hostmem := make([]float32, xSIB/4)
	//You can fill it
	for i := range hostmem {
		hostmem[i] = float32(i)
	}
	//Convert the slice to GoMem
	hostptr, err := gocu.MakeGoMem(hostmem)

	//Copy hostmem to x
	CudaMemManager.Copy(x, hostptr, xSIB) // This allocotor syncs the cuda stream after every copy.
	// You can make your own custom one. This was a default one
	// to help others get going. Some "extra" functions beyond the api
	// require an allocator.

	//if not using an allocator sync the stream before changing the host mem right after a mem copy.  It could cause problems.
	err = cs.Sync()
	check(err)

	//Zero out the golang host mem.
	for i := range hostmem {
		hostmem[i] = float32(0)
	}

	//do some tensor stuff can return vals to host mem by doing another copy
	CudaMemManager.Copy(hostptr, x, xSIB)

	check(err)
	fmt.Println(hostmem)
	//Output: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

}
