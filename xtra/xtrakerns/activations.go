package xtrakerns

//ThreshForward is kind of memory expensive, mostly because it is experimental.
//To test start the positive at random uniform numbers between .9 and 1.1
//and do the negcoefs between .01 and .2 or something along those lines.
//maybe the threshold should be between -.3 and .3 uniform number
func ThreshForward() Kernel {
	return Kernel{
		Name: `ThreshForward`,
		Code: `
		extern "C" __global__ void ThreshForward(const int XThreads,
												 const int batchsize,
												 const float *x,
												 float *y,
												 const float *negcoefs,
												 const float *threshhold,
												 const float *poscoefs)
		{
			for (int i=0;i<batchsize;i++)
			{
				int stride=XThreads*i;
					CUDA_GRID_LOOP_X(xIdx,XThreads)
					{
						if (x[stride+xIdx]>threshhold[xIdx])
						{
							y[stride+xIdx]=  x[stride+xIdx]*poscoefs[xIdx];
						}
						else
						{
							y[stride+xIdx]=  negcoefs[xIdx]*x[stride+xIdx];
						}
					}
			}
		}`,
	}
}

//ThreshForwardFP16 is kind of memory expensive, mostly because it is experimental.
//To test start the positive at random uniform numbers between .9 and 1.1
//and do the negcoefs between .01 and .2 or something along those lines.
//maybe the threshold should be between -.3 and .3 uniform number
func ThreshForwardFP16() Kernel {
	return Kernel{
		Name: `ThreshForwardFP16`,
		Code: `extern "C" __global__ void ThreshForwardFP16(const int XThreads,
			const int batchsize,
			const __half *x,
			__half *y,
			const __half *negcoefs,
			const __half *threshhold,
			const __half *poscoefs)
{
for (int i=0;i<batchsize;i++)
{
int stride=XThreads*i;
CUDA_GRID_LOOP_X(xIdx,XThreads)
{
if (__hgt(x[stride+xIdx],threshhold[xIdx]))
{
y[stride+xIdx]=  __hmul(x[stride+xIdx],poscoefs[xIdx]);
}
else
{
y[stride+xIdx]=   __hmul(negcoefs[xIdx],x[stride+xIdx]);
}
}
}
}`,
	}
}

//ThreshBackward --backwards function for forward
func ThreshBackward() Kernel {
	return Kernel{
		Name: `ThreshBackward`,
		Code: `extern "C" __global__ void ThreshBackward(const int XThreads,
			const int batchsize,
			const float *x,
			float *dx,
			const float *dy,
			const float *negcoefs,
			float *dnegcoefs,
			const float *threshhold,
			const float *poscoefs,
			float *dposcoefs)
{

for (int i=0;i<batchsize;i++)
{
int stride=XThreads*i;
CUDA_GRID_LOOP_X(xIdx,XThreads)
{
if (x[stride+xIdx]>threshhold[xIdx])
{
dx[stride+xIdx]=  poscoefs[xIdx]*dy[stride+xIdx];
dposcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];

}
else
{
dx[stride+xIdx]=  negcoefs[xIdx]*dy[stride+xIdx];
dnegcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];
}
}
}
}`,
	}
}

//ThreshBackwardFP16 --backwards function for forward
func ThreshBackwardFP16() Kernel {
	return Kernel{
		Name: `ThreshBackwardFP16`,
		Code: `extern "C" __global__ void ThreshBackwardFP16(const int XThreads,
			const int batchsize,
			const __half *x,
			__half *dx,
			const __half *dy,
			const __half *negcoefs,
			__half *dnegcoefs,
			const __half *threshhold,
			const __half *poscoefs,
			__half *dposcoefs)
{
for (int i=0;i<batchsize;i++)
{
int stride=XThreads*i;
CUDA_GRID_LOOP_X(xIdx,XThreads)
{
if (__hgt(x[stride+xIdx],threshhold[xIdx]))
{
//  dx[stride+xIdx]=  poscoefs[xIdx]*dy[stride+xIdx];
dx[stride+xIdx]=__hmul(dy[stride+xIdx],poscoefs[xIdx]);
// dposcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];
dposcoefs[xIdx]=__hfma(dy[xIdx],x[stride+xIdx],dposcoefs[xIdx]);
}
else
{
// dx[stride+xIdx]=  negcoefs[xIdx]*dy[stride+xIdx];
dx[stride+xIdx]= __hmul(dy[stride+xIdx],negcoefs[xIdx]);
// dnegcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];
dnegcoefs[xIdx]=__hfma(dy[xIdx],x[stride+xIdx],dnegcoefs[xIdx]);
}
}
}
}`,
	}
}

//PreluForward does the forward Prelu
func PreluForward() Kernel {
	return Kernel{
		Name: `PreluForward`,
		Code: `
		extern "C" __global__ void PreluForward(const int XThreads,
												const int batchsize,
												const float *x,
												float *y,
												const float *coefs)
		{
		  
			for (int i=0;i<batchsize;i++)
			{
				int stride=XThreads*i;
					CUDA_GRID_LOOP_X(xIdx,XThreads)
					{
						if (x[stride+xIdx]>0)
						{
							y[stride+xIdx]=  x[stride+xIdx];
						}
						else
						{
							y[stride+xIdx]=  coefs[xIdx]*x[stride+xIdx];
						}
					}
			}
		   
		}`,
	}
}

//PreluForwardFP16 does the forward Prelu
func PreluForwardFP16() Kernel {
	return Kernel{
		Name: `PreluForwardFP16`,
		Code: `extern "C" __global__ void PreluForwardFP16(const int XThreads,
			const int batchsize,
			const __half *x,
			__half *y,
			const __half *coefs)
{

for (int i=0;i<batchsize;i++)
{
int stride=XThreads*i;
CUDA_GRID_LOOP_X(xIdx,XThreads)
{
if (__hgt(x[stride+xIdx],0))
{
y[stride+xIdx]=  x[stride+xIdx];
}
else
{
y[stride+xIdx]=  __hmul(coefs[xIdx],x[stride+xIdx]);
}
}
}

}   `,
	}
}

//PreluBackward --backwards function for forward
func PreluBackward() Kernel {
	return Kernel{
		Name: `PreluBackward`,
		Code: `extern "C" __global__ void PreluBackward(const int XThreads,
			const int batchsize,
			float *dx,
			const float *x,
			const float *dy,
			const float *coefs,
			float *dcoefs)
{
for (int i=0;i<batchsize;i++)
{
int stride=XThreads*i;
CUDA_GRID_LOOP_X(xIdx,XThreads)
{
if (x[stride+xIdx]>0)
{
dx[stride+xIdx]=  dy[stride+xIdx];
}
else
{
dx[stride+xIdx]=  coefs[xIdx]*dy[stride+xIdx];
dcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];
}
}
}
}`,
	}
}

//PreluBackwardFP16 --backwards function for forward
func PreluBackwardFP16() Kernel {
	return Kernel{
		Name: `PreluBackwardFP16`,
		Code: `extern "C" __global__ void PreluBackwardFP16(const int XThreads,
			const int batchsize,
			__half *dx,
			const __half *x,
			const __half *dy,
			const __half *coefs,
			__half *dcoefs)
{
const __half zero0 = __float2half(0);
for (int i=0;i<batchsize;i++)
{
int stride=XThreads*i;
CUDA_GRID_LOOP_X(xIdx,XThreads)
{
if (__hgt(x[stride+xIdx],zero0))
{
dx[stride+xIdx]=  dy[stride+xIdx];
}
else
{
//  dx[stride+xIdx]=  coefs[xIdx]*dy[stride+xIdx];
dx[stride+xIdx]=  __hmul(coefs[xIdx],dy[stride+xIdx]);
// dcoefs[xIdx]+=dy[xIdx]*x[stride+xIdx];
dcoefs[xIdx]=__hfma(dy[xIdx],x[stride+xIdx],dcoefs[xIdx]);
}
}
}
}`,
	}
}

//LeakyForwardAlphaBeta - is the leaky activation
func LeakyForwardAlphaBeta() Kernel {
	return Kernel{
		Name: `LeakyForwardAlphaBeta`,
		Code: `extern "C" __global__ void LeakyForwardAlphaBeta(const int length,
			const float *x,
			float *y,
			const float coef,
			const float alpha,
			 const float beta)
{

CUDA_GRID_LOOP_X(i, length)
{
const float previous = y[i];
if (x[i] > 0.0)
{
const float current = x[i];
y[i] = (beta*previous) + (alpha *current) ;
}
else
{
const float current = x[i]*coef;
y[i] = (beta*previous) + (alpha *current) ;
}
__syncthreads();
}
}`,
	}
}

//LeakyForwardAlphaBetaFP16 is the leaky activation
func LeakyForwardAlphaBetaFP16() Kernel {
	return Kernel{
		Name: `LeakyForwardAlphaBetaFP16`,
		Code: `extern "C" __global__ void LeakyForwardAlphaBetaFP16(const int length,
			const __half *x,
			__half *y,
			const __half coef,
			const __half alpha,
			 const __half beta)
{
const __half zero0 = __float2half(0);
CUDA_GRID_LOOP_X(i, length)
{

if (__hgt(x[i],zero0))
{
// y[i] = (beta*y[i]) + (alpha *x[i]) ;
y[i]=__hadd(__hmul(beta,y[i]),__hmul(alpha,x[i]));
}
else
{
//y[i] = (beta*previous) + (alpha *x[i]*coef);
y[i]=__hadd(__hmul(beta,y[i]),__hmul(alpha,__hmul(x[i],coef)));
}
__syncthreads();
}
}`,
	}
}

//LeakyBackwardAlphaBeta --backwards function for forward
func LeakyBackwardAlphaBeta() Kernel {
	return Kernel{
		Name: `LeakyBackwardAlphaBeta`,
		Code: `extern "C" __global__ void LeakyBackwardAlphaBeta(const int length,
			const float *x,
			float *dx,
			const float *dy,
			const float coef,
			const float alpha,
			const float beta)
{

CUDA_GRID_LOOP_X(i, length)
{
const float previous = dx[i];
if (x[i] > 0.0)
{
const float current= dy[i];
dx[i] =(beta *previous) + (current * alpha);
}
else
{
const float current= dy[i]*coef;
dx[i] = (beta *previous) + (current * alpha);
}
__syncthreads();
}
}`,
	}
}

//LeakyBackwardAlphaBetaFP16 --backwards function for forward
func LeakyBackwardAlphaBetaFP16() Kernel {
	return Kernel{
		Name: `LeakyBackwardAlphaBetaFP16`,
		Code: `extern "C" __global__ void LeakyBackwardAlphaBetaFP16(const int length,
			const __half *x,
			__half *dx,
			const __half *dy,
			const __half coef,
			const __half alpha,
			const __half beta)
{
const __half zero0 = __float2half(0);
CUDA_GRID_LOOP_X(i, length)
{

if (__hgt(x[i],zero0))
{
// dx[i] =(beta *dx[i]) + (dy[i] * alpha);
dx[i]=__hadd(__hmul(beta,dy[i]),__hmul(alpha,dx[i]));
}
else
{
// dx[i] = (beta *dx[i]) + (dy[i]*coef * alpha);
dx[i]=__hadd(__hmul(beta,dx[i]),__hmul(alpha,__hmul(dy[i],coef)));
}
__syncthreads();
}
}`,
	}
}

//LeakyForwardAlpha is a leaky function
func LeakyForwardAlpha() Kernel {
	return Kernel{
		Name: `LeakyForwardAlpha`,
		Code: `
		extern "C" __global__ void LeakyForwardAlpha(const int length,
													 const float *x,
													 float *y,
													 const float coef,
													 const float alpha)
		{
		
			CUDA_GRID_LOOP_X(i, length)
			{
				
				if (x[i] > 0.0)
				{
					y[i] = alpha *x[i];
				}
				else
				{
					const float current=x[i]*coef;
					y[i] =current * alpha;
				}
				 __syncthreads();
			}
		}`,
	}
}

//LeakyForwardAlphaFP16 is a leaky function
func LeakyForwardAlphaFP16() Kernel {
	return Kernel{
		Name: `LeakyForwardAlphaFP16`,
		Code: `extern "C" __global__ void LeakyForwardAlphaFP16(const int length,
			const __half *x,
			__half *y,
			const __half coef,
			const __half alpha)
{
const __half zero0 = __float2half(0);
CUDA_GRID_LOOP_X(i, length)
{

if (__hgt(x[i],zero0))
{
y[i] = __hmul(alpha ,x[i]);
}
else
{

y[i] =__hmul(__hmul(x[i],coef) , alpha);
}
__syncthreads();
}
}`,
	}
}

//LeakyBackwardAlpha --backwards function for forward
func LeakyBackwardAlpha() Kernel {
	return Kernel{
		Name: `LeakyBackwardAlpha`,
		Code: `extern "C" __global__ void LeakyBackwardAlpha(const int length,
			const float *x,
			float *dx,
			const float *dy,
			const float coef,
			const float alpha)
{

CUDA_GRID_LOOP_X(i, length)
{

if (x[i] > 0.0)
{
dx[i] = dy[i]*alpha;
}
else   
{
const float current=dy[i]*coef;
dx[i] = current *alpha;
}
__syncthreads();
}
}`,
	}
}

//LeakyBackwardAlphaFP16 --backwards function for forward
func LeakyBackwardAlphaFP16() Kernel {
	return Kernel{
		Name: `LeakyBackwardAlphaFP16`,
		Code: `extern "C" __global__ void LeakyBackwardAlphaFP16(const int length,
			const __half *x,
			__half *dx,
			const __half *dy,
			const __half coef,
			const __half alpha)
{
const __half zero0 = __float2half(0);

CUDA_GRID_LOOP_X(i, length)
{

if  (__hgt(x[i],zero0))
{
// dx[i] = dy[i]*alpha;
dx[i] = __hmul(alpha ,dy[i]);
}
else
{
// dx[i] = dy[i]*coef *alpha;
dx[i] =__hmul(__hmul(dy[i],coef) , alpha);
}
__syncthreads();
}
}`,
	}
}

//LeakyForward is a leaky function
func LeakyForward() Kernel {
	return Kernel{
		Name: `LeakyForward`,
		Code: `extern "C" __global__ void LeakyForward(const int length,
			const float *x,
			float *y,
			const float coef)
{
CUDA_GRID_LOOP_X(i, length)
{
if (x[i] > 0.0)
{
y[i] = x[i];
}
else
{
y[i] = x[i] * coef;
}
}
}`,
	}
}

//LeakyForwardFP16 is a leaky function
func LeakyForwardFP16() Kernel {
	return Kernel{
		Name: `LeakyForwardFP16`,
		Code: `extern "C" __global__ void LeakyForwardFP16(const int length,
			const __half *x,
			__half *y,
			const __half coef)
{
const __half zero0 = __float2half(0);
CUDA_GRID_LOOP_X(i, length)
{
if  (__hgt(x[i],zero0))
{
y[i] = x[i];
}
else
{
//   y[i] = x[i] * coef;
y[i]= __hmul( x[i] , coef);
}
}
}`,
	}
}

//LeakyBackward --backwards function for forward
func LeakyBackward() Kernel {
	return Kernel{
		Name: `LeakyBackward`,
		Code: `extern "C" __global__ void LeakyBackward(const int length,
			const float *x,
			float *dx,
			const float *dy,
			const float coef)
{

CUDA_GRID_LOOP_X(i, length)
{

if (x[i] > 0.0)
{

dx[i] = dy[i];
}
else
{

dx[i] = dy[i] * coef;
}
}
}`,
	}
}

//LeakyBackwardFP16 --backwards function for forward
func LeakyBackwardFP16() Kernel {
	return Kernel{
		Name: `LeakyBackwardFP16`,
		Code: `extern "C" __global__ void LeakyBackwardFP16(const int length,
			const __half *x,
			__half *dx,
			const __half *dy,
			const __half coef)
{
const __half zero0 = __float2half(0);
CUDA_GRID_LOOP_X(i, length)
{

if  (__hgt(x[i],zero0))
{
dx[i] = dy[i];
}
else
{
//       dx[i] = dy[i] * coef;
dx[i]= __hmul( dy[i] , coef);
}
}
}`,
	}
}
