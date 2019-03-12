package nvjpeg

/*
#include<nvjpeg.h>
*/
import "C"

//LibraryPropertyType are flags for finding the library major, minor,patch
type LibraryPropertyType C.libraryPropertyType

func (l LibraryPropertyType) c() C.libraryPropertyType {
	return C.libraryPropertyType(l)
}

//LibraryPropertyTypeFlag passes LibraryPropertyType flags through methods
type LibraryPropertyTypeFlag struct {
}

//Major passes the Major flag
func (l LibraryPropertyTypeFlag) Major() LibraryPropertyType {
	return LibraryPropertyType(C.MAJOR_VERSION)
}

//Minor passes the minor flag
func (l LibraryPropertyTypeFlag) Minor() LibraryPropertyType {
	return LibraryPropertyType(C.MINOR_VERSION)
}

//Patch passes the patch flag
func (l LibraryPropertyTypeFlag) Patch() LibraryPropertyType {
	return LibraryPropertyType(C.PATCH_LEVEL)
}

// GetProperty returns library's property values, such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
func GetProperty(ltype LibraryPropertyType) (int, error) {
	var x C.int
	err := status(C.nvjpegGetProperty(ltype.c(), &x)).error()
	return int(x), err
}
