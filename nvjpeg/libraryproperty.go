package nvjpeg

/*
#include<nvjpeg.h>
*/
import "C"

// GetLibraryProperties returns library's property values. The major ,minor,and patch version
func GetLibraryProperties() (major, minor, patch int, err error) {
	var cmaj C.int
	var cmin C.int
	var cpat C.int
	err = status(C.nvjpegGetProperty(C.MAJOR_VERSION, &cmaj)).error()
	if err != nil {
		return
	}
	err = status(C.nvjpegGetProperty(C.MINOR_VERSION, &cmin)).error()
	if err != nil {
		return
	}
	err = status(C.nvjpegGetProperty(C.PATCH_LEVEL, &cpat)).error()
	if err != nil {
		return
	}
	major = int(cmaj)
	minor = int(cmin)
	patch = int(cpat)
	return major, minor, patch, nil
}
