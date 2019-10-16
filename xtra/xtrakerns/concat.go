package xtrakerns

//ConcatForwardNCHW is a concat for NCHW that hasn't been tested.
func ConcatForwardNCHW() Kernel {
	return Kernel{
		Name: `ConcatForwardNCHW`,
		Code: `extern "C" __global__ void ConcatForwardNCHW( const int XThreads,
			const int Batches,
			const int Channels1,
			const int src1vol,
			const float *Src1,
			const int Channels2,
			const int src2vol,
			const float *Src2,
			float *dest)
{
for (int i = 0;i<Batches;i++)
{
const int Stride= Batches*(src1vol+src2vol);
const int src1batchstride=src1vol*i;
const int src2batchstride=src2vol*i;
for (int j=0;j<Channels1;j++)
{
CUDA_GRID_LOOP_X(xIdx, XThreads)
{
dest[Stride+(j*XThreads)+xIdx]  = Src1[src1batchstride+(j*XThreads)+xIdx];
}
}
for (int j=0;j<Channels2;j++){
CUDA_GRID_LOOP_X(xIdx, XThreads)
{
dest[Stride+(j*XThreads)+src1vol+xIdx]  = Src2[src2batchstride+(j*XThreads)+xIdx];
}
}
}
}`,
	}
}

//ConcatBackwardNCHW is a concat for NCHW that hasn't been tested.
func ConcatBackwardNCHW() Kernel {
	return Kernel{
		Name: `ConcatBackwardNCHW`,
		Code: `extern "C" __global__ void ConcatBackwardNCHW( const int XThreads,
			const int Batches,
			const int Channels1,
			const int src1vol,
			 float *Src1,
			const int Channels2,
			const int src2vol,
			 float *Src2,
			const float *dest)
{
for (int i = 0;i<Batches;i++)
{
const int Stride= Batches*(src1vol+src2vol);
const int src1batchstride=src1vol*i;
const int src2batchstride=src2vol*i;
for (int j=0;j<Channels1;j++)
{
CUDA_GRID_LOOP_X(xIdx, XThreads)
{
Src1[src1batchstride+(j*XThreads)+xIdx]=  dest[Stride+(j*XThreads)+xIdx];  
}
}
for (int j=0;j<Channels2;j++){
CUDA_GRID_LOOP_X(xIdx, XThreads)
{
Src2[src2batchstride+(j*XThreads)+xIdx]  = dest[Stride+(j*XThreads)+src1vol+xIdx];  
}
}
}
}`,
	}
}

//ConcatForwardNCHWhalf is concat func in halfs
func ConcatForwardNCHWhalf() Kernel {
	return Kernel{
		Name: `ConcatForwardNCHWhalf`,
		Code: `extern "C" __global__ void ConcatForwardNCHWhalf( const int XThreads,
			const int Batches,
			const int Channels1,
			const int src1vol,
			const __half *Src1,
			const int Channels2,
			const int src2vol,
			const __half *Src2,
			__half *dest)
{
for (int i = 0;i<Batches;i++)
{
const int Stride= Batches*(src1vol+src2vol);
const int src1batchstride=src1vol*i;
const int src2batchstride=src2vol*i;
for (int j=0;j<Channels1;j++)
{
CUDA_GRID_LOOP_X(xIdx, XThreads)
{
dest[Stride+(j*XThreads)+xIdx]  = Src1[src1batchstride+(j*XThreads)+xIdx];
}
}
for (int j=0;j<Channels2;j++){
CUDA_GRID_LOOP_X(xIdx, XThreads)
{
dest[Stride+(j*XThreads)+src1vol+xIdx]  = Src2[src2batchstride+(j*XThreads)+xIdx];
}
}
}
}`,
	}
}

//ConcatBackwardNCHWhalf for concat half
func ConcatBackwardNCHWhalf() Kernel {
	return Kernel{
		Name: `ConcatBackwardNCHWhalf`,
		Code: `extern "C" __global__ void ConcatBackwardNCHWhalf( const int XThreads,
			const int Batches,
			const int Channels1,
			const int src1vol,
		__half *Src1,
	   const int Channels2,
	   const int src2vol,
		__half *Src2,
	   const __half *dest)
{
for (int i = 0;i<Batches;i++)
{
const int Stride= Batches*(src1vol+src2vol);
const int src1batchstride=src1vol*i;
const int src2batchstride=src2vol*i;
for (int j=0;j<Channels1;j++)
{
CUDA_GRID_LOOP_X(xIdx, XThreads)
{
Src1[src1batchstride+(j*XThreads)+xIdx]=  dest[Stride+(j*XThreads)+xIdx];  
}
}
for (int j=0;j<Channels2;j++){
CUDA_GRID_LOOP_X(xIdx, XThreads)
{
Src2[src2batchstride+(j*XThreads)+xIdx]  = dest[Stride+(j*XThreads)+src1vol+xIdx];  
}
}
}
}`,
	}
}
