package cudart

import (
	"errors"

	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/cutil"
)

//MemManager allocates memory to cuda under the unified memory management,
//and handles memory copies between memory under the unified memory mangement, and copies to and from Go memory.
type MemManager struct {
	//s      *Stream
	//d      Device
	flg    MemcpyKind
	onhost bool
}

//CreateMemManager creates an allocator that is bounded to cudas unified memory management.
func CreateMemManager(d Device) (*MemManager, error) {

	major, err := d.Major()
	if err != nil {
		return nil, err
	}
	if major < 6 {
		return nil, errors.New("Only Supported on Devices that are Compute Major 6 and up")
	}
	var flg MemcpyKind
	return &MemManager{
		//	s:   s,
		//	d:   d,
		flg: flg.Default(),
	}, nil
}

//SetHost sets a host allocation flag. SetHost can be changed at anytime.
//	-onhost=true all mallocs with allocator will allocate to host
//  -onhost=false all mallocs with allocator will allocate to device assigned to allocater. (default)
func (m *MemManager) SetHost(onhost bool) {
	m.onhost = onhost

}

//Malloc allocates memory to either the host or the device. sib = size in bytes
func (m *MemManager) Malloc(sib uint) (cuda cutil.Mem, err error) {
	cuda = new(gocu.CudaPtr)
	if m.onhost {
		err = MallocManagedHost(cuda, sib)
		return cuda, err
	}
	//err = m.d.Set()
	//if err != nil {
	//	return cuda, err
	//}
	err = MallocManagedGlobal(cuda, sib)
	if err != nil {
		return nil, err
	}
	return cuda, err
}

//Copy copies memory with amount of bytes passed in sib from src to dest
func (m *MemManager) Copy(dest, src cutil.Pointer, sib uint) error {
	return MemCpy(dest, src, sib, m.flg)

}

//AsyncCopy does an AsyncCopy with the mem manager.
func (m *MemManager) AsyncCopy(dest, src cutil.Pointer, sib uint, s gocu.Streamer) error {
	return MemcpyAsync(dest, src, sib, m.flg, s)
}
