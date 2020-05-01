package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io/ioutil"
	"os"

	"github.com/dereklstinson/gocudnn/gocu"

	"github.com/dereklstinson/gocudnn/cudart"

	"github.com/dereklstinson/gocudnn/nvjpeg"
)

func main() {
	stream, err := cudart.CreateBlockingStream()

	check(err)
	numofdevices, err := cudart.GetDeviceCount()
	check(err)
	var devicenumber int32
	if numofdevices > 1 {
		devicenumber = 0
	}
	dev, err := cudart.CreateDevice(devicenumber)
	check(err)
	//fmt.Println(dev.MemGetInfo())
	allocator, err := cudart.CreateMemManager(stream, dev)

	var backendflag nvjpeg.Backend
	var frmtflag nvjpeg.OutputFormat
	handle, err := nvjpeg.CreateEx(backendflag.Default())
	check(err)
	imgreader, err := os.Open("1.JPG")
	check(err)
	imgbytes, err := ioutil.ReadAll(imgreader)
	check(err)
	subsampletype, ws, hs, err := nvjpeg.GetImageInfo(handle, imgbytes)
	check(err)
	fmt.Println("Subsample Type: ", subsampletype.String())
	fmt.Println("Widths :", ws)
	fmt.Println("Heights :", hs)
	js, err := nvjpeg.CreateJpegState(handle)
	check(err)
	nvimage, err := nvjpeg.CreateImageDest(frmtflag.RGB(), ws, hs, allocator)
	check(err)
	check(js.Decode(handle, imgbytes, frmtflag.RGB(), nvimage, stream))
	channels, _ := nvimage.Get()
	channelbytes := make([][]byte, len(channels))
	stream.Sync()
	h, w := (int)(hs[0]), (int)(ws[0])
	for i := range channels {
		channelbytes[i] = make([]byte, h*w)
		cbptr, err := gocu.MakeGoMem(channelbytes[i])
		check(err)
		err = allocator.Copy(cbptr, channels[i], (uint)(h*w))
		check(err)

		stream.Sync()
	}
	/*
		for i := range channelbytes {
			fmt.Println("Channel", i)
			fmt.Println("Data: ", channelbytes[i])
		}
	*/
	newimage, err := os.Create("1go.JPG")
	check(err)

	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			var val color.RGBA
			val.A = 255
			val.R = channelbytes[0][i*w+j]
			val.G = channelbytes[1][i*w+j]
			val.B = channelbytes[2][i*w+j]
			rgba.Set(j, i, val)
		}
	}

	check(jpeg.Encode(newimage, rgba, nil))

}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
