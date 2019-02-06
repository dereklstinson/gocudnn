package gocudnn_test

import (
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

func TestCuda_GetDeviceList(t *testing.T) {
	var cu gocudnn.Cuda
	cu.LockHostThread()
	tests := []struct {
		name    string
		cu      gocudnn.Cuda
		want    []*gocudnn.Device
		wantErr bool
	}{
		{
			name:    "list of devices",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			got, err := cu.GetDeviceList()
			if (err != nil) != tt.wantErr {
				t.Errorf("Cuda.GetDeviceList() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if len(got) < 1 {
				t.Errorf("Should have at least cuda program")
			}
			for i := range got {
				err = got[i].Set()
				if err != nil {
					t.Error(err)
				}
				used, have, err := cu.MemGetInfo()
				if err != nil {
					t.Error(err)
				}
				t.Errorf("Device %d has is using %d of Memory, and has %d of memory", i, used, have)
			}
		})
	}
}
