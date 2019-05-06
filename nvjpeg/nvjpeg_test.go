package nvjpeg_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"github.com/dereklstinson/GoCudnn/nvjpeg"
)

func TestImageInfo(t *testing.T) {
	check := func(err error, t *testing.T) {
		if err != nil {
			t.Error(err)
		}
	}
	var backendflag nvjpeg.Backend
	handle, err := nvjpeg.CreateEx(backendflag.Default())
	check(err, t)
	imgreader, err := os.Open("1.JPG")
	check(err, t)
	imgbytes, err := ioutil.ReadAll(imgreader)
	check(err, t)
	subsampletype, ws, hs, err := nvjpeg.GetImageInfo(handle, imgbytes)
	check(err, t)
	fmt.Println("Subsample Type: ", subsampletype.String())
	fmt.Println("Widths :", ws)
	fmt.Println("Heights :", hs)
	//decoder, err := nvjpeg.JpegStateCreate(handle)
	//	check(err, t)

}
