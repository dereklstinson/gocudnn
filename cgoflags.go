package gocudnn

//#cgo LDFLAGS:-lcuda
//#cgo LDFLAGS: -lcudart
//#cgo LDFLAGS: -lcudnn
//
//#cgo LDFLAGS:-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib
//#cgo CFLAGS: -I/usr/local/cuda/include/
import "C"
