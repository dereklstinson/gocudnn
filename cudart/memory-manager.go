package cudart

import (
	"errors"

	"github.com/dereklstinson/GoCudnn/gocu"
)

//MemManager allocates memory to cuda under the unified memory management,
//and handles memory copies between memory under the unified memory mangement, and copies to and from Go memory.
//Device support is for deivces that use compute major 6 and up.
//Device || Streamer cannot cannot be nil
//Allocator is technically a wrapper for malloc functions, but I included it for users of the package.
//Since nvjpeg contains a function that requires a gocu.Allocator
type MemManager struct {
	s      *Stream
	d      Device
	flg    MemcpyKind
	onhost bool
}

//CreateAllocator creates an allocator that is bounded to cudas unified memory management.
func CreateAllocator(s *Stream, d Device) (*MemManager, error) {

	major, err := d.Major()
	if err != nil {
		return nil, err
	}
	if major < 6 {
		return nil, errors.New("Only Supported on Devices that are Compute Major 6 and up")
	}
	var flg MemcpyKind
	return &MemManager{
		s:   s,
		d:   d,
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
//If onhost is false then device assigned to Allocator will be set. If onhost is false then it won't be set.
func (m *MemManager) Malloc(sib uint) (cuda gocu.Mem, err error) {
	cuda = new(gocu.CudaPtr)
	if m.onhost {
		err = MallocManagedHost(cuda, sib)
		return cuda, err
	}
	err = m.d.Set()
	if err != nil {
		return cuda, err
	}
	err = MallocManagedGlobal(cuda, sib)
	if err != nil {
		return nil, err
	}
	return cuda, m.s.Sync()
}

//Copy copies memory with amount of bytes passed in sib from src to dest
func (m *MemManager) Copy(dest, src gocu.Mem, sib uint) error {
	return MemCpy(dest, src, sib, m.flg)

}
