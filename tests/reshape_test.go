package tests

import (
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//THIS IS FAILING

func TestReshape(t *testing.T) {

}
func goptrtest(dims []int32) (*gocudnn.GoPointer, error) {
	params := parammaker(dims)
	array := make([]float32, params)
	for i := int32(0); i < dims[0]; i++ {
		for j := int32(0); j < dims[1]; j++ {
			for k := int32(0); j < dims[2]; k++ {
				for l := int32(0); l < dims[3]; l++ {
					array[i] = float32(i)
				}
			}
		}

	}
	return gocudnn.MakeGoPointer(array)
}
func parammaker(dims []int32) int32 {
	mult := int32(1)
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	return mult
}
