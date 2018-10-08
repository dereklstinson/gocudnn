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

type Device interface {
	Major() int
	Minor() int
}

const nvcccompute30 = "	nvcc --gpu-architecture=compute_30 --gpu-code=compute_30 --ptx Activation30.cu"
const nvcccompute35 = "	nvcc --gpu-architecture=compute_35 --gpu-code=compute_35 --ptx Activation35.cu"
const nvcccompute50 = "	nvcc --gpu-architecture=compute_50 --gpu-code=compute_50 --ptx Activation50.cu"
const nvcccompute52 = "	nvcc --gpu-architecture=compute_52 --gpu-code=compute_52 --ptx Activation52.cu"
const nvcccompute60 = "	nvcc --gpu-architecture=compute_60 --gpu-code=compute_60 --ptx Activation60.cu"
const nvcccompute61 = "	nvcc --gpu-architecture=compute_61 --gpu-code=compute_61 --ptx Activation61.cu"
const nvcccompute62 = "	nvcc --gpu-architecture=compute_62 --gpu-code=compute_62 --ptx Activation61.cu"
const nvcccompute70 = "	nvcc --gpu-architecture=compute_70 --gpu-code=compute_70 --ptx Activation70.cu"

func putconstintoarray() ([]string, []string) {
	computs := make([]string, 8)
	computs[0] = nvcccompute30
	computs[1] = nvcccompute35
	computs[2] = nvcccompute50
	computs[3] = nvcccompute52
	computs[4] = nvcccompute60
	computs[5] = nvcccompute61
	computs[6] = nvcccompute62
	computs[7] = nvcccompute70
	names := make([]string, 8)
	names[0] = "Activation30"
	names[1] = "Activation35"
	names[2] = "Activation50"
	names[3] = "Activation52"
	names[4] = "Activation60"
	names[5] = "Activation61"
	names[6] = "Activation62"
	names[7] = "Activation70"
	return computs, names

}
func MakeSeveralMakes(directory, dotCUname string) {

	computes, names := putconstintoarray()
	for i := 0; i < len(computes); i++ {
		var some makefile
		some.lines = make([]string, 2)
		some.lines[0] = "run:\n"
		some.lines[1] = "\t" + computes[i] + names[i] + "\n"
		data := []byte(some.lines[0] + some.lines[1])
		err := os.MkdirAll(directory, 0644)
		if err != nil {
			fmt.Println(err)
			panic(err)
		}
		err = ioutil.WriteFile(directory+"Makefile", data, 0644)
		if err != nil {
			fmt.Println(err)
			panic(err)
		}
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
	}

}

const nvccarg = "nvcc --gpu-architecture=compute_"
const nvccarg1 = " --gpu-code=compute_"
const nvccarg2 = " --ptx "

type makefile struct {
	lines []string
}

func MakeMakeFile(directory string, dotCUname string, device Device) string {

	device.Major()

	majstr := strconv.Itoa(device.Major())

	minstr := strconv.Itoa(device.Minor())
	computecapability := majstr + minstr

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
	some.lines[1] = "\t" + nvccarg + computecapability + nvccarg1 + computecapability + nvccarg2 + dotCUname + "\n"

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

func LoadPTXFile(directory, filename string) string {

	ptxdata, err := ioutil.ReadFile(directory + filename)
	if err != nil {
		panic(err)
	}
	return string(ptxdata)
}
