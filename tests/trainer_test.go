package tests

import (
	"fmt"
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

func TestTrainer(t *testing.T) {
	trainingkernellocation := "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/"
	gocudnn.Cuda{}.LockHostThread()
	//cudnn context
	var cuda gocudnn.Cuda

	devices, err := cuda.GetDeviceList()
	if err != nil {
		t.Error(err)
	}

	devicenum := len(devices)
	fmt.Println("Number of Devices:", devicenum)
	/*	err = devices[0].Set()
		if err != nil {
			t.Error(err)
		}*/
	ctx, err := cuda.CtxCreate(-1, devices[0])
	if err != nil {
		t.Error(err)
	}
	err = ctx.Set()
	if err != nil {
		t.Error(err)
	}
	//handle := gocudnn.NewHandle()

	//stream, err := gocudnn.Cuda{}.CreateBlockingStream()
	if err != nil {
		t.Error(err)
	}
	/*
		err = handle.SetStream(stream)
		if err != nil {
			t.Error(err)
		}
	*/
	_, x, err := maketestfilterW()
	if err != nil {
		t.Error(err)
	}
	_, dx, err := maketestfilterDW()
	if err != nil {
		t.Error(err)
	}

	trainhandle, err := gocudnn.Xtra{}.MakeXHandle(trainingkernellocation, devices[0])
	if err != nil {
		t.Error(err)
	}
	traind, err := gocudnn.Xtra{}.NewTrainingDescriptor(
		trainhandle,
		gocudnn.TrainingModeFlag{}.Adam(),
		gocudnn.DataTypeFlag{}.Float(),
		gocudnn.RegularizationFlag{}.L1L2(),
	)
	//trainhandle.SetStream(stream)
	if err != nil {
		t.Error(err)
	}
	/*
		l1, err := gocudnn.MallocManaged(gocudnn.SizeT(4), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			t.Error(err)
		}
	*/
	/*l2, err := gocudnn.MallocManaged(gocudnn.SizeT(4), gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		t.Error(err)
	}
	*/
	_, gsum, err := maketestxgsum()
	if err != nil {
		t.Error(err)
	}
	_, xsum, err := maketestxgsum()
	if err != nil {
		t.Error(err)
	}
	//err = stream.Sync()

	params := gocudnn.Xtra{}.CreateParamsFloat32(float32(1e-8), float32(.001), float32(.9), float32(.999))
	err = traind.TrainValues(trainhandle, 128, dx, x, gsum, xsum, params)
	if err != nil {
		t.Error(err)
	}
	//	err = stream.Sync()qw
	if err != nil {
		t.Error(err)
	}
	float32slice := make([]float32, 20*20*20*20)
	err = cuda.CtxSynchronize()
	if err != nil {
		t.Error(err)
		//	fmt.Println(float32slice)
	}
	err = gsum.FillSlice(float32slice)
	if err != nil {
		t.Error(err)
		//	fmt.Println(float32slice)
	}
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
}
