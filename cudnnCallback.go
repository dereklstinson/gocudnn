package gocudnn

/*
#include "cudnnCallback.h"
#include <cudnn.h>
extern void go_call_back(cudnnSeverity_t sev, void *udata, cudnnDebug_t *dbg, char *msg);
//void go_call_back_cgo(cudnnSeverity_t sev, void *udata, cudnnDebug_t *dbg, char *msg);
*/
import "C"

import (
	"fmt"
	"io"
	"sync"
	"unsafe"
)

//typedef struct {
//    unsigned cudnn_version;
//    cudnnStatus_t cudnnStatus;
//    unsigned time_sec;      /* epoch time in seconds */
//    unsigned time_usec;     /* microseconds part of epoch time */
//    unsigned time_delta;    /* time since start in seconds */
//    cudnnHandle_t handle;   /* cudnn handle */
//    cudaStream_t stream;    /* cuda stream ID */
//    unsigned long long pid; /* process ID */
//    unsigned long long tid; /* thread ID */
//    int cudaDeviceId;       /* CUDA device ID */
//    int reserved[15];       /* reserved for future use */
//} cudnnDebug_t;
//export go_call_back
func go_call_back(sev C.cudnnSeverity_t, udata unsafe.Pointer, dbg *C.cudnnDebug_t, msg *C.char) {
	var err error
	var userdata string
	if udata == nil {
		userdata = "None"
	} else {
		userdata = C.GoString((*C.char)(udata))
	}
	fmt.Println("going into callback")
	gomsg := C.GoString(msg)
	gdbg := (*Debug)(dbg)
	s := fmt.Sprintf("UserData: %v\n: %v\n,Message: %v\n",
		userdata, gdbg, gomsg)
	//	gs := C.GoString(msg)
	mu.Lock()

	_, err = callbackwriter.Write([]byte(s))
	mu.Unlock()
	if err != nil {
		panic(err)
	}
}

//Debug is Debug type
type Debug C.cudnnDebug_t

func (d *Debug) String() string {
	v := d.cudnn_version
	ts := d.time_sec
	tus := d.time_usec
	td := d.time_delta
	handle := d.handle
	stream := d.stream
	pid := d.pid
	tid := d.tid
	did := d.cudaDeviceId
	return fmt.Sprintf("Debug{\n\tVersion: %v\n"+
		"\tTime(s): %v\n"+
		"\tTime(us): %v\n"+
		"\tTime(delta): %v\n"+
		"\tHandle: %v\n"+
		"\tStream: %v\n"+
		"\tPID: %v\n"+
		"\tTID: %v\n"+
		"\tDeviceID: %v\n}", v, ts, tus, td, handle, stream, pid, tid, did)
}

var mu sync.Mutex
var callbackwriter io.Writer

//SetCallBack sets the debug callback function.  Callback data will be writer to the writer.
//udata is custom user data that will write to the call back.  udata can be nil
//Callback is not functional.
func SetCallBack(udata fmt.Stringer, w io.Writer) error {
	mu.Lock()

	callbackwriter = w
	mu.Unlock()
	if udata == nil {
		return Status(C.cudnnSetCallback(C.CUDNN_SEV_ERROR_EN, nil, createcallback())).error("SetCallBack")

	}
	udatabytes := C.CString(udata.String())
	return Status(C.cudnnSetCallback(C.CUDNN_SEV_ERROR_EN, (unsafe.Pointer)(udatabytes), createcallback())).error("SetCallBack")

}
func createcallback() C.cudnnCallback_t {
	var cb C.cudnnCallback_t
	cb = (C.cudnnCallback_t)(unsafe.Pointer(C.CudaCallBack))
	return cb
}

//#define CUDNN_SEV_ERROR_EN (1U << CUDNN_SEV_ERROR)
//#define CUDNN_SEV_WARNING_EN (1U << CUDNN_SEV_WARNING)
//#define CUDNN_SEV_INFO_EN (1U << CUDNN_SEV_INFO)
