package xtrakerns

//AdaGrad is the AdaGrad training weight updater
func AdaGrad() Kernel {
	return Kernel{
		Name: `AdaGrad`,
		Code: `extern "C" __global__ void AdaGrad(const int length,
			float *weights,   //weights input and output
			float *dw,        //input and will have to set to zero
			float *gsum,      //storage
			const float rate, //input
			const float eps,
			const float dwalpha)
{ //input
CUDA_GRID_LOOP_X(cell, length)
{
gsum[cell] =  gsum[cell] + (dw[cell] * dw[cell]);
weights[cell] += -(rate * dw[cell]) / (sqrtf(gsum[cell]) + eps);
dw[cell] = dw[cell]*dwalpha; //smoothing factor.
}
}`,
	}
}

//AdaGradFP16 is a weight updater
func AdaGradFP16() Kernel {
	return Kernel{
		Name: `AdaGradFP16`,
		Code: `extern "C" __global__ void AdaGradFP16(const int n,
			__half *w,   //w input and output
			__half *dw,        //input and will have to set to zero
			__half *gsum,      //storage
			const __half rate, //input
			const __half eps,
			const __half dwalpha)
{ //input
StartAxis(stx,x)
int n2=n/2;
__half2 *w2=(__half2*)w,*dw2=(__half2*)dw,*gsum2=(__half2*)gsum;

const __half2 rate2=__halves2half2(rate,rate);
const __half2 eps2=__halves2half2(eps,eps);
const __half2 dwalpha2=__halves2half2(dwalpha,dwalpha);
CUDA_GRID_LOOP_X(i, n2)
{
__half2 holder = gsum2[i];
gsum2[i] = __hfma2(dw2[i],dw2[i],holder);
w2[i] = __hadd2(-__h2div((__hmul2(rate2,dw2[i])) , (__hadd2(h2sqrt(gsum2[i]), eps2))),w2[i]);
dw2[i] =__hmul2(dw2[i],dwalpha2);
}
if (stx==0 && (n%2)){
__half holder = gsum[n-1];
gsum[n-1] = __hfma(dw[n-1],dw[n-1],holder);
w[n-1] = -__hdiv((__hmul(rate,dw[n-1])) , (__hadd(hsqrt(gsum[n-1]), eps)));
dw[n-1] =__hmul(dw[n-1],dwalpha);
}
}`,
	}
}

//Adam is a weight trainer
func Adam() Kernel {
	return Kernel{
		Name: `Adam`,
		Code: `extern "C" __global__ void Adam(const int n,
			float *w,
			float *gsum,
			float *xsum,
			float *dw,
			const float rate,
			const float beta1,
			const float beta2,
			const float eps,
			const float denombeta1,
			const float denombeta2,
			const float dwalpha)
{

CUDA_GRID_LOOP_X(i, n)
{

gsum[i] = (beta1 * gsum[i]) + ((1.0 - beta1) * dw[i]);
float gsumt = gsum[i] /denombeta1;
xsum[i] = (beta2 * xsum[i]) + ((1.0 - beta2) * (dw[i] * dw[i]));
float xsumt = xsum[i] / denombeta2;
w[i] += -(rate * gsumt) / (sqrtf(xsumt) + eps);
dw[i]=  dwalpha*dw[i]; //smoothing factor
}

}`,
	}
}

//AdamFP16 is a weight updater probably needs fixed
func AdamFP16() Kernel {
	return Kernel{
		Name: `AdamFP16`,
		Code: `extern "C" __global__ void AdamFP16(const int n,
			__half *w,
			__half *gsum,
			__half *xsum,
			__half *dw,
			const __half rate,
			const __half beta1,
			const __half beta2,
			const __half eps,
			const __half denombeta1,
			const __half denombeta2,
			const __half dwalpha)
{
int n2=n/2;
__half2 *w2=(__half2*)w,*dw2=(__half2*)dw,*gsum2=(__half2*)gsum,*xsum2=(__half2*)xsum;
const __half2 rate2=__halves2half2(rate,rate);
const __half2 eps2=__halves2half2(eps,eps);
const __half2 dwalpha2=__halves2half2(dwalpha,dwalpha);
const __half2 beta12=__halves2half2(beta1,beta1);
const __half2 beta22=__halves2half2(beta2,beta2);
const __half one1 = __float2half(1.0);
const __half2 one2=__halves2half2(one1,one1);
StartAxis(stx,x)
CUDA_GRID_LOOP_X(i, n2)
{
gsum2[i] =__hfma2(__hsub2(one2,beta12),dw2[i],__hmul2(beta12,gsum2[i]));
__half2 gsumt = __h2div(gsum2[i] ,__halves2half2(denombeta1,denombeta1));
xsum2[i] = __hfma2(beta22 , xsum2[i], __hmul2(__hsub2(one2, beta22), __hmul2(dw2[i] , dw2[i])));
__half2 xsumt = __h2div(xsum2[i] , __halves2half2(denombeta2,denombeta2));
w2[i]=__hsub2(w2[i],__h2div(__hmul2(rate2,gsumt),__hadd2(h2sqrt(xsumt),eps2)));
dw2[i]=  __hmul2(dwalpha2,dw2[i]);
}

if (stx==0 && (n%2)){
const int i = n-1;
gsum[i] =__hfma(__hsub(one1,beta1),dw[i],__hmul(beta1,gsum[i]));
__half gsumt = __hdiv(gsum[i] ,denombeta1);
xsum[i] = __hfma(beta2 , xsum[i], __hmul(__hsub(one1, beta2), __hmul(dw[i] , dw[i])));
__half xsumt = __hdiv(xsum[i] , denombeta2);
w[i]=__hsub(w[i],__hdiv(__hmul(rate,gsumt),__hadd(hsqrt(xsumt),eps)));
dw[i]=  __hmul(dwalpha,dw[i]);
}
}`,
	}
}

//AdaDelta is a weight trainer
func AdaDelta() Kernel {
	return Kernel{
		Name: `AdaDelta`,
		Code: `extern "C" __global__ void AdaDelta(const int length,
			float *weights,   //weights input and output
			float *gsum,      //storage
			float *xsum,      //storage
			float *dw,        //input and will have to set to zero
			const float rate, //input
			const float eps,
			const float ro,
			const float dwalpha)
{

CUDA_GRID_LOOP_X(i, length)
{

gsum[i] = (ro * gsum[i]) + ((1.0-ro)*dw[i] * dw[i]);
const float dx = sqrtf((xsum[i]+eps)/(gsum[i]+eps))*dw[i];
xsum[i]=(ro*xsum[i])+((1-ro)*dx*dx);
weights[i] -= dx;
dw[i] = dw[i]*dwalpha;
}
}`,
	}
}

//AdaDeltaFP16 is a weight trainer
func AdaDeltaFP16() Kernel {
	return Kernel{
		Name: `AdaDeltaFP16`,
		Code: `extern "C" __global__ void AdaDeltaFP16(const int n,
			__half *w,   //weights input and output
			__half *gsum,      //storage
			__half *xsum,      //storage
			__half *dw,        //input and will have to set to zero
			const __half rate, //input
			const __half eps,
			 const __half ro,
			const __half dwalpha)
{
StartAxis(stx,x)
int n2=n/2;
__half2 *w2=(__half2*)w,*dw2=(__half2*)dw,*gsum2=(__half2*)gsum,*xsum2=(__half2*)xsum;
const __half2 rate2=__halves2half2(rate,rate);
const __half2 eps2=__halves2half2(eps,eps);
const __half2 ro2=__halves2half2(ro,ro);
const __half one1 = __float2half(1.0);
const __half2 one2=__halves2half2(one1,one1);
const __half2 dwalpha2=__halves2half2(dwalpha,dwalpha);
CUDA_GRID_LOOP_X(i, n2)
{
gsum2[i]= __hfma2(__hsub2(one2,ro2),__hmul2(dw2[i],dw2[i]),__hmul2(ro2,gsum2[i]));
const __half2 dx2= __hmul2(h2sqrt(__h2div(__hadd2(xsum2[i],eps2),__hadd2(gsum2[i],eps2))),dw2[i]);
xsum2[i]= __hfma2(__hsub2(one2,ro2),__hmul2(dx2,dx2),__hmul2(ro2,xsum2[i]));
w2[i] =__hsub2(w2[i],dx2);
dw2[i] =  __hmul2(dw2[i],dwalpha2);
}

if (stx ==0 &&(n%2)){
int i = n-1;
gsum[i]= __hfma(__hsub(one1,ro),__hmul(dw[i],dw[i]),__hmul(ro,gsum[i]));
const __half dx= __hmul(hsqrt(__hdiv(__hadd(xsum[i],eps),__hadd(gsum[i],eps))),dw[i]);
xsum[i]= __hfma(__hsub(one1,ro),__hmul(dx,dx),__hmul(ro,xsum[i]));
w[i] =__hsub(w[i],dx);
dw[i] =  __hmul(dw[i],dwalpha);
}
}`,
	}
}
