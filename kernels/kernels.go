package kernels

import "C"
import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
	//	"github.com/dereklstinson/GoCudnn"
)

//Device just has to return those two things in order to compute the kernels
type Device interface {
	Major() int
	Minor() int
}

const nvcccmd = "nvcc "

//const nvccarg = " --gpu-architecture=compute_"
//const nvccarg1 = " --gpu-code=compute_"
//const nvccarg2 = " --ptx "
const nvccarg = "--gpu-architecture=compute_"
const nvccarg1 = "--gpu-code=compute_"
const nvccarg2 = "--ptx "

const defaultmakedirectory = "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/"

type makefile struct {
	lines []string
}

//MakeMakeFile Makes the make file
func MakeMakeFile(directory string, dotCUname string, device Device) string {
	if directory == "__default__" {
		directory = defaultmakedirectory
	}

	device.Major()

	majstr := strconv.Itoa(device.Major())

	minstr := strconv.Itoa(device.Minor())
	computecapability := majstr + minstr + " "

	newname := dotCUname

	if strings.Contains(dotCUname, ".cu") {
		newname = strings.TrimSuffix(dotCUname, ".cu")

	} else {
		dotCUname = dotCUname + ".cu"
	}
	newname = newname + ".ptx"
	var some makefile
	//some.lines=make([]string,13)
	some.lines = make([]string, 2)
	some.lines[0] = "run:\n"
	some.lines[1] = "\t" + nvcccmd + nvccarg + computecapability + nvccarg1 + computecapability + nvccarg2 + dotCUname + "\n"

	data := []byte(some.lines[0] + some.lines[1])
	err := os.MkdirAll(directory, 0644)
	if err != nil {
		fmt.Println(err)
		fmt.Println(directory)
		panic(err)
	}
	err = ioutil.WriteFile(directory+"Makefile", data, 0644)
	if err != nil {
		fmt.Println(err)
		panic(err)
	}

	//	newcommand := exec.Command(nvcccmd, nvccarg+computecapability, nvccarg1+computecapability, nvccarg2+dotCUname)
	newcommand := exec.Command("make")
	newcommand.Dir = directory
	time.Sleep(time.Millisecond)
	response, err := newcommand.Output()
	//err = newcommand.Run()

	if err != nil {
		fmt.Println("*****Something Is wrong with the" + dotCUname + " file*******")
		fmt.Println(string(response))
		panic(err)
	}
	return newname
}

//LoadPTXFile Loads the ptx file
func LoadPTXFile(directory, filename string) string {

	ptxdata, err := ioutil.ReadFile(directory + filename)
	if err != nil {
		panic(err)
	}
	return string(ptxdata)
}
