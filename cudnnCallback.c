#include <cudnn.h>
#include "_cgo_export.h"


extern void CudaCallBack(cudnnSeverity_t sev, void *udata, cudnnDebug_t *dbg, char *msg){
	//char *msg1=msg;
	go_call_back(sev,udata,dbg,msg);
}