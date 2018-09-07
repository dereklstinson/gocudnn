package gocudnn

/*
#include <cuda.h>
*/
import "C"

//Context holds a CUcontext.  This is soon going to be added!
type Context struct {
	ctx C.CUcontext
}
