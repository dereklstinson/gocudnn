package gocudnn

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/half"

	"github.com/dereklstinson/cutil"
)

type memstringer struct {
	td   *TensorD
	t    cutil.Pointer
	kind cudart.MemcpyKind
}

func (m *memstringer) String() string {
	frmt, dtype, dims, stride, err := m.td.Get()
	if err != nil {
		return fmt.Sprintf("Tensor Data: {\n%v\n}\n", "Err in getting hidden Tensor Descriptor")
	}
	sib, err := m.td.GetSizeInBytes()
	if err != nil {
		return fmt.Sprintf("Tensor Data: {\n%v\n}\n", "Err in getting sib for hidden Tensor Descriptor")
	}
	length := findvolume(dims)
	fflg := frmt
	dflg := dtype
	switch dtype {
	case dflg.Float():
		data := make([]float32, length)
		hptr, err := gocu.MakeGoMem(data)

		if err != nil {
			return fmt.Sprintf("Tensor Data: {\n%v\n}\n", "Err in wrapping data::"+err.Error())
		}
		err = cudart.MemCpy(hptr, m.t, sib, m.kind)
		if err != nil {
			return fmt.Sprintf("Tensor Data: {\n%v\n}\n", "Err in copy to host ::"+err.Error())
		}
		if frmt == fflg.NCHW() {
			return fmt.Sprintf("Tensor Data: {\n%v\n}\n", nchwtensorstringformated(dims, stride, data))
		}
		return fmt.Sprintf("Tensor Data: {\n%v\n}\n", nhwctensorstringformated(dims, stride, data))

	case dflg.Half():
		data := make([]half.Float16, length)
		hptr, err := gocu.MakeGoMem(data)

		if err != nil {
			return fmt.Sprintf("Tensor Data: {\n%v\n}\n", "Err in wrapping data::"+err.Error())
		}
		err = cudart.MemCpy(hptr, m.t, sib, m.kind)
		if err != nil {
			return fmt.Sprintf("Tensor Data: {\n%v\n}\n", "Err in copy to host ::"+err.Error())
		}
		if frmt == fflg.NCHW() {
			return fmt.Sprintf("Tensor Data: {\n%v\n}\n", nchwtensorstringformated(dims, stride, half.ToFloat32(data)))
		}
		return fmt.Sprintf("Tensor Data: {\n%v\n}\n", nhwctensorstringformated(dims, stride, half.ToFloat32(data)))

	default:
		return fmt.Sprintf("Tensor Data: {\n%v\n}\n", "Unsupported DataType")

	}
}

//GetStringer returns a stringer that will pring cuda allocated memory formated in NHWC or NCHW.
//Only works for 4d tensors with float or half datatype. It will only print the data.
func GetStringer(tD *TensorD, t cutil.Pointer) (fmt.Stringer, error) {

	frmt, dtype, _, _, err := tD.Get()
	if err != nil {
		return nil, err
	}
	fflg := frmt
	if !(frmt == fflg.NCHW() || frmt == fflg.NHWC()) {
		return nil, errors.New(" GetStringer(tD *TensorD, t cutil.Pointer): Unsuported Format")
	}
	dflg := dtype
	if !(dtype == dflg.Float() || dtype == dflg.Half()) {
		return nil, errors.New(" GetStringer(tD *TensorD, t cutil.Pointer): Unsupported Type")
	}
	var kind cudart.MemcpyKind
	return &memstringer{
		td:   tD,
		t:    t,
		kind: kind.Default(),
	}, nil

}
func nhwctensorstringformated(dims, strides []int32, data []float32) string {

	var s string
	s = "\n"
	for i := int32(0); i < dims[0]; i++ {
		s = s + fmt.Sprintf("Batch[%v]{\n", i)
		for j := int32(0); j < dims[1]; j++ {

			for k := int32(0); k < dims[2]; k++ {
				s = s + fmt.Sprintf("(%v,%v)[ ", j, k)
				for l := int32(0); l < dims[3]; l++ {
					val := data[i*strides[0]+j*strides[1]+k*strides[2]+l*strides[3]]
					if val >= 0 {
						s = s + fmt.Sprintf(" %.5f ", val)
					} else {
						s = s + fmt.Sprintf("%.5f ", val)
					}

				}
				s = s + "], "

			}
			s = s + "\n"
		}
		s = s + "}\n"
	}
	return s
}
func nchwtensorstringformated(dims, strides []int32, data []float32) string {
	//flg := t.Format()

	var s string
	s = "\n"
	for i := int32(0); i < dims[0]; i++ {
		s = s + fmt.Sprintf("Batch[%v]{\n", i)
		for j := int32(0); j < dims[1]; j++ {
			s = s + fmt.Sprintf("\tChannel[%v]{\n", j)
			for k := int32(0); k < dims[2]; k++ {
				s = s + "\t\t"
				for l := int32(0); l < dims[3]; l++ {
					val := data[i*strides[0]+j*strides[1]+k*strides[2]+l*strides[3]]
					if val >= 0 {
						s = s + fmt.Sprintf(" %.4f ", val)
					} else {
						s = s + fmt.Sprintf("%.4f ", val)
					}

				}
				s = s + "\n"

			}
			s = s + "\t}\n"
		}
		s = s + "}\n"
	}
	return s
}
