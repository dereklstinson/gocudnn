package cudart

import "testing"

func TestMallocArray(t *testing.T) {
	var flag ArrayFlag
	arr := new(Array)

	extent := MakeCudaExtent(200, 200, 200)
	var cfk ChannelFormatKind

	var cfd = CreateChannelFormatDesc(8, 8, 8, 8, cfk.Signed())
	err := Malloc3dArray(arr, &cfd, extent, flag.Default())
	if err != nil {
		t.Error(err)
	}

}
