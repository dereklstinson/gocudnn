package gocudnn_test

import (
	"sync"
	"testing"

	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocudnn/cudart"
	"github.com/dereklstinson/gocudnn/gocu"
	"github.com/dereklstinson/cutil"
)

func TestParrallel(t *testing.T) {
	check := func(e error) {
		if e != nil {
			t.Error(e)
		}
	}
	ndevs, err := cudart.GetDeviceCount()
	check(err)
	handles := make([]*gocudnn.Handle, ndevs)
	workers := make([]*gocu.Worker, ndevs)
	MemManager := make([]*cudart.MemManager, ndevs)
	var wg sync.WaitGroup
	for i := range handles {
		wg.Add(1)
		go func(i int, err error) {
			workers[i] = gocu.NewWorker(cudart.CreateDevice(int32(i)))
			handles[i] = gocudnn.CreateHandleEX(workers[i], true)
			MemManager[i], err = cudart.CreateMemManager(workers[i])
			check(err)
			wg.Done()
		}(i, err)
	}
	wg.Wait()
	var cmode gocudnn.ConvolutionMode //Creating Flags
	var frmt gocudnn.TensorFormat     //Creating Flags
	var dtype gocudnn.DataType        //Creating Flags

	cmode.CrossCorrelation() //Setting Flags
	frmt.NCHW()              //Setting Flags
	dtype.Float()            //Setting Flags

	inputdims := []int32{12, 3, 32, 32}
	filterdims := []int32{256, 3, 5, 5} //Assumming this is an input image
	biasdims := []int32{1, 256, 1, 1}
	inputD, err := gocudnn.CreateTensorDescriptor()
	check(err)
	filterD, err := gocudnn.CreateFilterDescriptor()
	check(err)
	biasD, err := gocudnn.CreateTensorDescriptor()
	check(err)
	outputD, err := gocudnn.CreateTensorDescriptor()
	check(inputD.Set(frmt, dtype, inputdims, nil))
	check(filterD.Set(dtype, frmt, filterdims))
	check(biasD.Set(frmt, dtype, biasdims, nil))
	insib, err := inputD.GetSizeInBytes()
	check(err)
	fsib, err := filterD.GetSizeInBytes()
	check(err)
	bsib, err := biasD.GetSizeInBytes()
	check(err)
	convD, err := gocudnn.CreateConvolutionDescriptor()
	check(err)
	check(convD.Set(cmode, dtype, []int32{2, 2}, []int32{1, 1}, []int32{1, 1}))
	outdims, err := convD.GetOutputDims(inputD, filterD)
	check(err)
	check(outputD.Set(frmt, dtype, outdims, nil))
	outsib, err := outputD.GetSizeInBytes()
	check(err)
	inputMEM := make([]cutil.Mem, len(MemManager))
	filterMEM := make([]cutil.Mem, len(MemManager))
	biasMEM := make([]cutil.Mem, len(MemManager))
	outputMEM := make([]cutil.Mem, len(MemManager))
	for i, m := range MemManager {
		wg.Add(1)
		go func(i int, m *cudart.MemManager, err error) {
			inputMEM[i], err = m.Malloc(insib)
			check(err)
			filterMEM[i], err = m.Malloc(fsib)
			check(err)
			biasMEM[i], err = m.Malloc(bsib)
			check(err)
			outputMEM[i], err = m.Malloc(outsib)
			check(err)
			wg.Done()
		}(i, m, err)
	}
	wg.Wait()
	var algopref gocudnn.ConvolutionForwardPref
	algopref.PreferFastest()

	for i, h := range handles {
		wg.Add(1)
		go func(i int, h *gocudnn.Handle, err error) {

			algo, err := convD.GetForwardAlgorithm(h, inputD, filterD, outputD, algopref, 0)
			check(err)
			wspacesib, err := convD.GetForwardWorkspaceSize(h, inputD, filterD, outputD, algo)
			check(err)

			wspacemem, err := MemManager[i].Malloc(wspacesib)
			check(err)
			check(convD.Forward(h, 1, inputD, inputMEM[i], filterD, filterMEM[i], algo, wspacemem, wspacesib, 0, outputD, outputMEM[i]))
			wg.Done()
		}(i, h, err)
	}
	wg.Wait()

}
