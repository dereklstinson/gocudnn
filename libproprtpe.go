package cudnn

/*

#include <cudnn.h>

*/
import "C"

//LibraryPropertyType is basically values that are for the current library that you have
type LibraryPropertyType C.libraryPropertyType

//Will have to double check this
const (
	MajorVersion LibraryPropertyType = C.MAJOR_VERSION
	MinorVersion LibraryPropertyType = C.MINOR_VERSION
	PatchLevel   LibraryPropertyType = C.PATCH_LEVEL
)
const gocudnnversionmadefor = 71
const gocudaversionmadefor = 92

//GetProperty returns the library versions passed
func GetProperty(property LibraryPropertyType) (int, error) {
	var value C.int
	x := C.cudnnGetProperty(C.libraryPropertyType(property), &value)
	return int(value), Status(x).error("LibraryPropertyType")
}

//GOCUversioning returns the versions of cuda and cudnn when the bindings were made
func GOCUversioning() (int, int) {
	return gocudaversionmadefor, gocudnnversionmadefor
}
