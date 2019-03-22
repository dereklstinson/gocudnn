package gocudnn

/*

#include <cudnn.h>

*/
import "C"

//GetLibraryVersion will return the library version you have installed
func GetLibraryVersion() (major, minor, patch int32, err error) {
	var ma, mi, pa C.int
	err = Status(C.cudnnGetProperty(C.MAJOR_VERSION, &ma)).error("Major")
	if err != nil {
		return
	}
	err = Status(C.cudnnGetProperty(C.MINOR_VERSION, &mi)).error("Minor")
	if err != nil {
		return
	}
	err = Status(C.cudnnGetProperty(C.PATCH_LEVEL, &pa)).error("Patch")
	if err != nil {
		return
	}
	major = (int32)(ma)
	minor = (int32)(mi)
	patch = (int32)(pa)
	return major, minor, patch, err
}

//GetBindingVersion will return the library version this binding was made for.
func GetBindingVersion() (major, minor, patch int32) {
	return 7, 5, 0
}
