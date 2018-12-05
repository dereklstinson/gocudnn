package gocudnn_test

import gocudnn "github.com/dereklstinson/GoCudnn"

func maketestingmatrial4dnchw(val float32, dims []int32) (*gocudnn.TensorD, *gocudnn.Malloced, *gocudnn.GoPointer, []float32, error) {
	slice := makeallvalueslice(val, dims)
	gptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	tensord, err := gocudnn.Tensor{}.NewTensor4dDescriptor(gocudnn.DataTypeFlag{}.Float(), gocudnn.TensorFormatFlag{}.NCHW(), dims)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	cptr, err := gocudnn.UnifiedMangedGlobal(gptr.ByteSize())
	if err != nil {
		return nil, nil, nil, nil, err
	}
	err = gocudnn.UnifiedMemCopy(cptr, gptr)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return tensord, cptr, gptr, slice, nil

}
func makeallvalueslice(val float32, dims []int32) (slice []float32) {
	vol := 1
	for i := range dims {
		vol *= int(dims[i])
	}
	slice = make([]float32, vol)
	for i := 0; i < vol; i++ {
		slice[i] = val
	}
	return slice
}
func makeemptyslice(dims []int32) (slice []float32) {
	vol := 1
	for i := range dims {
		vol *= int(dims[i])
	}
	slice = make([]float32, vol)
	return slice
}
func makesliceofones(dims []int32) (slice []float32) {
	vol := 1
	for i := range dims {
		vol *= int(dims[i])
	}
	slice = make([]float32, vol)
	for i := 0; i < vol; i++ {
		slice[i] = 1
	}
	return slice
}
func helperwithdifferentelements(dims []int32) (*gocudnn.TensorD, *gocudnn.Malloced, *gocudnn.GoPointer, []float32, error) {
	slice := regular4darray(dims)
	gptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	tensord, err := gocudnn.Tensor{}.NewTensor4dDescriptor(gocudnn.DataTypeFlag{}.Float(), gocudnn.TensorFormatFlag{}.NCHW(), dims)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	cptr, err := gocudnn.UnifiedMangedGlobal(gptr.ByteSize())
	if err != nil {
		return nil, nil, nil, nil, err
	}
	err = gocudnn.UnifiedMemCopy(cptr, gptr)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return tensord, cptr, gptr, slice, nil

}
func regular4darray(dims []int32) (slice []float32) {
	vol := 1
	size := len(dims)
	s := make([]int32, size)

	for i := range dims {

		vol *= int(dims[i])
	}
	mult := int32(1)
	for i := size - 1; i >= 0; i-- {
		s[i] = mult
		mult *= (dims[i])
	}
	z := int32(0)

	slice = make([]float32, vol)
	for n := z; n < dims[0]; n++ {
		for c := z; c < dims[1]; c++ {
			for h := z; h < dims[2]; h++ {
				for w := z; w < dims[3]; w++ {
					slice[(n*s[0])+(c*s[1])+(h*s[2])+(w*s[3])] = float32((n+1)*10) + float32((c+1)*1) + float32(h+1)*.1 + float32(w+1)*.01
				}
			}
		}
	}
	return slice
}
