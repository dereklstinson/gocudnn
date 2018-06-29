package gocudnn

//#include <cuda_runtime_api.h>
import "C"

//Cuda is a nil struct that is used to pass Cuda functions
type Cuda struct {
}

//GetDevice will get the cuda device
func (cu Cuda) GetDevice() {

}
