package tests

import (
	"testing"
)

func TestTranspose(t *testing.T) {
	/*
		gocudnn.Cuda{}.LockHostThread()
		trainingkernellocation := "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/"
		var cu gocudnn.Cuda
		dev, err := cu.GetDeviceList()
		err = dev[0].Set()
		handle := gocudnn.NewHandle()
		xhandle, err := gocudnn.Xtra{}.MakeXHandle(trainingkernellocation, dev[0])
		if err != nil {
			t.Error(err)
		}
		stream, err := cu.CreateBlockingStream()
		if err != nil {
			t.Error(err)
		}
		handle.SetStream(stream)
		xhandle.SetStream(stream)
		originaldims := []int32{1, 128, 36, 36}
		descX, Xmem, err := testTensorFloat4dNCHW(originaldims)
		if err != nil {
			t.Error(err)
		}

		goarray := goarraytest(originaldims)
		//fmt.Println(goarray)
		Xhmem, err := gocudnn.MakeGoPointer(goarray)
		if err != nil {
			t.Error(err)
		}

		err = gocudnn.CudaMemCopy(Xmem, Xhmem, Xmem.ByteSize(), gocudnn.MemcpyKindFlag{}.HostToDevice())
		if err != nil {
			t.Error(err)
		}

		DescTrans, err := gocudnn.Xtra{}.CreateTransposeDesc(xhandle)

		transdesc, perms, err := DescTrans.GetChannelTransposeOutputDescAndPerm4d(descX)
		fmt.Println(perms)
		if err != nil {
			t.Error(err)
		}
		tsib, err := transdesc.GetSizeInBytes()
		if err != nil {
			t.Error(err)
		}
		transmem, err := gocudnn.UnifiedMangedGlobal(tsib)
		if err != nil {
			t.Error(err)
		}
		err = DescTrans.Transpose(xhandle, perms, descX, Xmem, transdesc, transmem)
		if err != nil {
			t.Error(err)
		}
		flt := gocudnn.DataTypeFlag{}.Float()
		length := gocudnn.FindLength(tsib, flt)
		slice := make([]float32, length)
		slice2 := make([]float32, length)
		transmem.FillSlice(slice)
		//	fmt.Println(slice)
		transback, permsback, err := DescTrans.GetChannelTransposeOutputDescAndPerm4d(transdesc)
		transmemback, err := gocudnn.UnifiedMangedGlobal(tsib)
		err = DescTrans.Transpose(xhandle, permsback, transdesc, transmem, transback, transmemback)
		transmemback.FillSlice(slice2)
		flag := false
		for i := range slice {
			if slice2[i] != goarray[i] {
				flag = true
			}
		}
		if flag == true {
			t.Error("Slice No Match Goarray")
		}
	*/
}

func goarraytest(dims []int32) []float32 {
	array := make([]float32, parammaker(dims))
	for i := 0; i < len(array); i++ {
		array[i] = float32(i)
	}

	return array
}
