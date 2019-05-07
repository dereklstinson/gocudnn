package npp_test

import (
	"testing"

	"github.com/dereklstinson/GoCudnn/npp"
)

func TestMemory(t *testing.T) {
	x, padding := npp.Malloc8uC1(1536, 1536)
	if padding != 1536 {
		t.Error(x, padding)
	}

}
