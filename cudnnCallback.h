#ifndef CUDNNCALLBACK_H
#define CUDNNCALLBACK_H
#include <cudnn.h>
void CudaCallBack(cudnnSeverity_t sev, void *udata, cudnnDebug_t *dbg,  char *msg);
//extern void go_call_back(cudnnSeverity_t sev, void *udata, cudnnDebug_t *dbg, char *msg);
#endif