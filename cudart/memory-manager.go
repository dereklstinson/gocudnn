package cudart

import (
	"errors"

	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/cutil"
)

//MemManager allocates memory to a cuda context/device under the unified memory management,
//and handles memory copies between memory under the unified memory mangement, and copies to and from Go memory.
type MemManager struct {
	w      *gocu.Worker
	flg    MemcpyKind
	onhost bool
}

//CreateMemManager creates an allocator that is bounded to cudas unified memory management.
func CreateMemManager(w *gocu.Worker) (*MemManager, error) {
	if w == nil {
		return nil, errors.New("CreateMemManager(): w is nil")
	}
	var flg MemcpyKind
	return &MemManager{
		w:   w,
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

	err = m.w.Work(func() error {
		if m.onhost {
			return MallocManagedHost(cuda, sib)

		}
		return MallocManagedGlobal(cuda, sib)

	})
	if err != nil {
		return nil, err
	}
	return cuda, err

}

//Copy copies memory with amount of bytes passed in sib from src to dest
func (m *MemManager) Copy(dest, src cutil.Pointer, sib uint) error {
	return m.w.Work(func() error {
		return Memcpy(dest, src, sib, m.flg)
	})

}

//AsyncCopy does an AsyncCopy with the mem manager.
func (m *MemManager) AsyncCopy(dest, src cutil.Pointer, sib uint, s gocu.Streamer) error {
	return m.w.Work(func() error {
		return MemcpyAsync(dest, src, sib, m.flg, s)
	})

}
