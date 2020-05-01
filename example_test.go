package gocudnn_test

import (
	"fmt"
	"runtime"

	"github.com/dereklstinson/gocudnn/gocu"

	"github.com/dereklstinson/gocudnn/cudart"

	gocudnn "github.com/dereklstinson/gocudnn"
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
	dev := cudart.CreateDevice(1)

	//Make an Allocator
	worker := gocu.NewWorker(dev)
	CudaMemManager, err := cudart.CreateMemManager(worker) //cs could be nil .  Check out cudart package on more about streams
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
	err = CudaMemManager.Copy(hostptr, x, xSIB)

	check(err)
	fmt.Println(hostmem)
	//Output: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19]

}

/*
//ExampleStream is an example of a concurrent calculation with one filter
func ExampleStream() {

	//Streams are concurrent and work with different cpu threads, and for each cpu thread there needs to be a new handle.
	type handler struct {
		w *gocu.Worker
		h *gocudnn.Handle
		s gocu.Streamer
	}
	runtime.LockOSThread()
	var err error
	check := func(e error) {
		if e != nil {
			panic(e)
		}
	}
	dev := cudart.CreateDevice(1)
	check(dev.Set())
	worker := gocu.NewWorker(dev)
	CudaMemManager, err := cudart.CreateMemManager(worker)
	check(err)

	//h := gocudnn.CreateHandleEX(worker, true)

	length := 500
	hs := make([]handler, length)
	inputtensors := make([]*gocudnn.TensorD, length)
	inputtensorsmems := make([]cutil.Mem, length)
	outputshost := make([]float32, length)
	outputtensors := make([]*gocudnn.TensorD, length)
	outputtensorsmems := make([]cutil.Mem, length)
	var tflg gocudnn.TensorFormat //Flag for tensor
	var dtflg gocudnn.DataType    //Flag for tensor
	dtflg.Float()
	tflg.NCHW()
	outputmemblock, err := CudaMemManager.Malloc(uint(length * 4))
	check(err)
	dim := int32(25)

	for i := range hs {
		hs[i].w = gocu.NewWorker(dev) //Worker creates a channel to a seperate thread
		hs[i].h = gocudnn.CreateHandle(true)

		err = hs[i].w.Work(func() error { //Work sends a function threading polling a channel. This function blocks.
			var err1 error
			hs[i].s, err1 = cudart.CreateNonBlockingStream()
			if err1 != nil {
				return err1
			}
			return hs[i].h.SetStream(hs[i].s)

		})
		check(err)

		//InputTensors
		inputtensors[i], err = gocudnn.CreateTensorDescriptor()
		check(err)
		//Setting the tensor descriptors
		check(inputtensors[i].Set(tflg, dtflg, []int32{1, dim, dim, dim}, nil))
		sib, err := inputtensors[i].GetSizeInBytes()

		check(err)
		inputtensorsmems[i], err = CudaMemManager.Malloc(sib)
		check(err)
		check(gocudnn.SetTensor(h, inputtensors[i], inputtensorsmems[i], float64(i)))

		//OutputTensors
		outputtensors[i], err = gocudnn.CreateTensorDescriptor()
		check(err)
		//Setting the tensor descriptors
		check(outputtensors[i].Set(tflg.NCHW(), dtflg.Float(), []int32{1, 1, 1, 1}, nil))
		sib, err = outputtensors[i].GetSizeInBytes()
		check(err)
		//outputtensorsmems[i], err = CudaMemManager.Malloc(sib)
		outputtensorsmems[i] = cutil.CreateUnsafeWrapper(cutil.Offset(outputmemblock, uint(4*i)).Ptr(), 4)

	}

	//Since everything is on the same gpu we can set up tensors like this.

	//filter
	filter, err := gocudnn.CreateFilterDescriptor()
	check(err)
	check(filter.Set(dtflg, tflg, []int32{1, dim, dim, dim}))
	filtertensor, err := gocudnn.CreateTensorDescriptor()
	check(err)
	check(filtertensor.Set(tflg, dtflg, []int32{1, dim, dim, dim}, nil))
	sib, err := filter.GetSizeInBytes()
	check(err)
	filtermem, err := CudaMemManager.Malloc(sib)
	check(err)
	check(gocudnn.SetTensor(h, filtertensor, filtermem, 1))

	//CreateConvolutionDescriptor
	var convflg gocudnn.ConvolutionMode //Flag for tensor

	convd, err := gocudnn.CreateConvolutionDescriptor()

	check(convd.Set(convflg.CrossCorrelation(), dtflg, []int32{0, 0}, []int32{1, 1}, []int32{1, 1}))
	var algoprefernce gocudnn.ConvolutionForwardPref
	algoprefernce.NoWorkSpace()
	var wg sync.WaitGroup
	for i, h := range hs {
		wg.Add(1)
		go func(i int, h handler) {
			h.w.Work(func() error {
				algo, err1 := convd.GetForwardAlgorithm(h.h, inputtensors[i], filter, outputtensors[i], algoprefernce, uint(0))
				if err1 != nil {
					return err1
				}
				return convd.Forward(h.h,
					1,
					inputtensors[i], inputtensorsmems[i],
					filter, filtermem,
					algo,
					nil, 0,
					0,
					outputtensors[i], outputtensorsmems[i],
				)

			})
			wg.Done()
		}(i, h)

	}
	wg.Wait()

	//Get Output
	outputwrapper, err := gocu.MakeGoMem(outputshost)
	check(err)
	check(CudaMemManager.Copy(outputwrapper, outputmemblock, uint(4*length)))

	fmt.Println(outputshost)


	//	Output: [0 15625 31250 46875 62500 78125 93750 109375 125000 140625 156250 171875 187500 203125 218750 234375 250000 265625 281250 296875 312500 328125 343750 359375 375000 390625 406250 421875 437500 453125 468750 484375 500000 515625 531250 546875 562500 578125 593750 609375 625000 640625 656250 671875 687500 703125 718750 734375 750000 765625 781250 796875 812500 828125 843750 859375 875000 890625 906250 921875 937500 953125 968750 984375 1e+06 1.015625e+06 1.03125e+06 1.046875e+06 1.0625e+06 1.078125e+06 1.09375e+06 1.109375e+06 1.125e+06 1.140625e+06 1.15625e+06 1.171875e+06 1.1875e+06 1.203125e+06 1.21875e+06 1.234375e+06 1.25e+06 1.265625e+06 1.28125e+06 1.296875e+06 1.3125e+06 1.328125e+06 1.34375e+06 1.359375e+06 1.375e+06 1.390625e+06 1.40625e+06 1.421875e+06 1.4375e+06 1.453125e+06 1.46875e+06 1.484375e+06 1.5e+06 1.515625e+06 1.53125e+06 1.546875e+06 1.5625e+06 1.578125e+06 1.59375e+06 1.609375e+06 1.625e+06 1.640625e+06 1.65625e+06 1.671875e+06 1.6875e+06 1.703125e+06 1.71875e+06 1.734375e+06 1.75e+06 1.765625e+06 1.78125e+06 1.796875e+06 1.8125e+06 1.828125e+06 1.84375e+06 1.859375e+06 1.875e+06 1.890625e+06 1.90625e+06 1.921875e+06 1.9375e+06 1.953125e+06 1.96875e+06 1.984375e+06 2e+06 2.015625e+06 2.03125e+06 2.046875e+06 2.0625e+06 2.078125e+06 2.09375e+06 2.109375e+06 2.125e+06 2.140625e+06 2.15625e+06 2.171875e+06 2.1875e+06 2.203125e+06 2.21875e+06 2.234375e+06 2.25e+06 2.265625e+06 2.28125e+06 2.296875e+06 2.3125e+06 2.328125e+06 2.34375e+06 2.359375e+06 2.375e+06 2.390625e+06 2.40625e+06 2.421875e+06 2.4375e+06 2.453125e+06 2.46875e+06 2.484375e+06 2.5e+06 2.515625e+06 2.53125e+06 2.546875e+06 2.5625e+06 2.578125e+06 2.59375e+06 2.609375e+06 2.625e+06 2.640625e+06 2.65625e+06 2.671875e+06 2.6875e+06 2.703125e+06 2.71875e+06 2.734375e+06 2.75e+06 2.765625e+06 2.78125e+06 2.796875e+06 2.8125e+06 2.828125e+06 2.84375e+06 2.859375e+06 2.875e+06 2.890625e+06 2.90625e+06 2.921875e+06 2.9375e+06 2.953125e+06 2.96875e+06 2.984375e+06 3e+06 3.015625e+06 3.03125e+06 3.046875e+06 3.0625e+06 3.078125e+06 3.09375e+06 3.109375e+06 3.125e+06 3.140625e+06 3.15625e+06 3.171875e+06 3.1875e+06 3.203125e+06 3.21875e+06 3.234375e+06 3.25e+06 3.265625e+06 3.28125e+06 3.296875e+06 3.3125e+06 3.328125e+06 3.34375e+06 3.359375e+06 3.375e+06 3.390625e+06 3.40625e+06 3.421875e+06 3.4375e+06 3.453125e+06 3.46875e+06 3.484375e+06 3.5e+06 3.515625e+06 3.53125e+06 3.546875e+06 3.5625e+06 3.578125e+06 3.59375e+06 3.609375e+06 3.625e+06 3.640625e+06 3.65625e+06 3.671875e+06 3.6875e+06 3.703125e+06 3.71875e+06 3.734375e+06 3.75e+06 3.765625e+06 3.78125e+06 3.796875e+06 3.8125e+06 3.828125e+06 3.84375e+06 3.859375e+06 3.875e+06 3.890625e+06 3.90625e+06 3.921875e+06 3.9375e+06 3.953125e+06 3.96875e+06 3.984375e+06 4e+06 4.015625e+06 4.03125e+06 4.046875e+06 4.0625e+06 4.078125e+06 4.09375e+06 4.109375e+06 4.125e+06 4.140625e+06 4.15625e+06 4.171875e+06 4.1875e+06 4.203125e+06 4.21875e+06 4.234375e+06 4.25e+06 4.265625e+06 4.28125e+06 4.296875e+06 4.3125e+06 4.328125e+06 4.34375e+06 4.359375e+06 4.375e+06 4.390625e+06 4.40625e+06 4.421875e+06 4.4375e+06 4.453125e+06 4.46875e+06 4.484375e+06 4.5e+06 4.515625e+06 4.53125e+06 4.546875e+06 4.5625e+06 4.578125e+06 4.59375e+06 4.609375e+06 4.625e+06 4.640625e+06 4.65625e+06 4.671875e+06 4.6875e+06 4.703125e+06 4.71875e+06 4.734375e+06 4.75e+06 4.765625e+06 4.78125e+06 4.796875e+06 4.8125e+06 4.828125e+06 4.84375e+06 4.859375e+06 4.875e+06 4.890625e+06 4.90625e+06 4.921875e+06 4.9375e+06 4.953125e+06 4.96875e+06 4.984375e+06 5e+06 5.015625e+06 5.03125e+06 5.046875e+06 5.0625e+06 5.078125e+06 5.09375e+06 5.109375e+06 5.125e+06 5.140625e+06 5.15625e+06 5.171875e+06 5.1875e+06 5.203125e+06 5.21875e+06 5.234375e+06 5.25e+06 5.265625e+06 5.28125e+06 5.296875e+06 5.3125e+06 5.328125e+06 5.34375e+06 5.359375e+06 5.375e+06 5.390625e+06 5.40625e+06 5.421875e+06 5.4375e+06 5.453125e+06 5.46875e+06 5.484375e+06 5.5e+06 5.515625e+06 5.53125e+06 5.546875e+06 5.5625e+06 5.578125e+06 5.59375e+06 5.609375e+06 5.625e+06 5.640625e+06 5.65625e+06 5.671875e+06 5.6875e+06 5.703125e+06 5.71875e+06 5.734375e+06 5.75e+06 5.765625e+06 5.78125e+06 5.796875e+06 5.8125e+06 5.828125e+06 5.84375e+06 5.859375e+06 5.875e+06 5.890625e+06 5.90625e+06 5.921875e+06 5.9375e+06 5.953125e+06 5.96875e+06 5.984375e+06 6e+06 6.015625e+06 6.03125e+06 6.046875e+06 6.0625e+06 6.078125e+06 6.09375e+06 6.109375e+06 6.125e+06 6.140625e+06 6.15625e+06 6.171875e+06 6.1875e+06 6.203125e+06 6.21875e+06 6.234375e+06 6.25e+06 6.265625e+06 6.28125e+06 6.296875e+06 6.3125e+06 6.328125e+06 6.34375e+06 6.359375e+06 6.375e+06 6.390625e+06 6.40625e+06 6.421875e+06 6.4375e+06 6.453125e+06 6.46875e+06 6.484375e+06 6.5e+06 6.515625e+06 6.53125e+06 6.546875e+06 6.5625e+06 6.578125e+06 6.59375e+06 6.609375e+06 6.625e+06 6.640625e+06 6.65625e+06 6.671875e+06 6.6875e+06 6.703125e+06 6.71875e+06 6.734375e+06 6.75e+06 6.765625e+06 6.78125e+06 6.796875e+06 6.8125e+06 6.828125e+06 6.84375e+06 6.859375e+06 6.875e+06 6.890625e+06 6.90625e+06 6.921875e+06 6.9375e+06 6.953125e+06 6.96875e+06 6.984375e+06 7e+06 7.015625e+06 7.03125e+06 7.046875e+06 7.0625e+06 7.078125e+06 7.09375e+06 7.109375e+06 7.125e+06 7.140625e+06 7.15625e+06 7.171875e+06 7.1875e+06 7.203125e+06 7.21875e+06 7.234375e+06 7.25e+06 7.265625e+06 7.28125e+06 7.296875e+06 7.3125e+06 7.328125e+06 7.34375e+06 7.359375e+06 7.375e+06 7.390625e+06 7.40625e+06 7.421875e+06 7.4375e+06 7.453125e+06 7.46875e+06 7.484375e+06 7.5e+06 7.515625e+06 7.53125e+06 7.546875e+06 7.5625e+06 7.578125e+06 7.59375e+06 7.609375e+06 7.625e+06 7.640625e+06 7.65625e+06 7.671875e+06 7.6875e+06 7.703125e+06 7.71875e+06 7.734375e+06 7.75e+06 7.765625e+06 7.78125e+06 7.796875e+06]

}
*/
