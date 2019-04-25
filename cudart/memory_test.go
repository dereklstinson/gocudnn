package cudart

import (
	"unsafe"
)
import "testing"

func TestMallocManagedGlobalUS(t *testing.T) {
	var x unsafe.Pointer

	err := MallocManagedGlobalUS(x, 512)
	if err != nil {
		t.Error(err)
	}
}
