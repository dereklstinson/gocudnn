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
	//cuda.
	devices, err := cuda.GetDeviceList()
	if err != nil {
		t.Error(err)
	}

	devicenum := len(devices)
	fmt.Println("Number of Devices:", devicenum)
	err = devices[0].Set()
	if err != nil {
		t.Error(err)
	}
	handle := gocudnn.NewHandle()
	stream, err := gocudnn.Cuda{}.CreateBlockingStream()
	if err != nil {
		t.Error(err)
	}
	err = handle.SetStream(stream)
	if err != nil {
		t.Error(err)
	}

	xdesc, x, err := maketestfilterW()
	if err != nil {
		t.Error(err)
	}
	dxdesc, dx, err := maketestfilterDW()
	if err != nil {
		t.Error(err)
	}
	fmt.Println(xdesc, x, dxdesc, dx)
	trainhandle, err := gocudnn.Xtra{}.MakeTrainingHandle(trainingkernellocation, devices[0])
	if err != nil {
		t.Error(err)
	}
	traind, err := gocudnn.Xtra{}.NewTrainingDescriptor(
		trainhandle,
		gocudnn.TrainingModeFlag{}.Adam(),
		gocudnn.DataTypeFlag{}.Float(),
		gocudnn.RegularizationFlag{}.L1L2(),
	)
	if err != nil {
		t.Error(err)
	}
	l1, err := gocudnn.MallocManaged(gocudnn.SizeT(4), gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		t.Error(err)
	}
	l2, err := gocudnn.MallocManaged(gocudnn.SizeT(4), gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		t.Error(err)
	}
	fmt.Println(l1, l2, traind)
	//	traind.TrainValues(trainhandle, 32, dx, x, l1, l2)

}
