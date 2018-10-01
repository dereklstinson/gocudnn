package gocudnn

import (
	"errors"

	"github.com/dereklstinson/GoCudnn/kernels"
)

type XActivationMode uint
type XActivationModeFlag struct {
}

func (x XActivationMode) tostringfwd(dtype DataType) string {
	dtf := DataTypeFlag{}
	var xaflg XActivationModeFlag
	if dtype != dtf.Float() {
		return "error XActivationMode - DataTypeNotSupported"
	}
	var ktf kernels.XtraKerns
	switch x {
	case xaflg.Leaky():
		return ktf.ForwardLeakyfloat()
	case xaflg.Parametric():
		return ktf.ForwardParametricfloat()
	}
	return "error"

}

func (x XActivationMode) tostringbwd(dtype DataType) string {
	dtf := DataTypeFlag{}
	var xaflg XActivationModeFlag
	if dtype != dtf.Float() {
		return "error XActivationMode - DataTypeNotSupported"
	}
	var ktf kernels.XtraKerns
	switch x {
	case xaflg.Leaky():
		return ktf.BackwardLeakyfloat()
	case xaflg.Parametric():
		return ktf.BackwardParametricfloat()
	}
	return "error"
}

//Leaky returns the leaky flag
func (x XActivationModeFlag) Leaky() XActivationMode {
	return XActivationMode(1)
}

//Parametric returns the Parametric flag
func (x XActivationModeFlag) Parametric() XActivationMode {
	return XActivationMode(2)
}

type XActivationD struct {
	data    DataType
	amode   XActivationMode
	dtype   DataType
	tmode   TrainingMode
	counter uint64
	fwdmode *Kernel
	bwdmode *Kernel
	rmodek  *Kernel
	tmodek  *Kernel
	coef    float64
}

//NewXActivationDescriptor - Creates a descriptor for the xtra functions made for gocudnn.
//Note: Only trainable activations will be trained.  tmode will be ignored for unsupported activations
//Note: Only functions requiring coef will get it.  coef will be ignored for unsupported activations
func (xtra Xtra) NewXActivationDescriptor(h *XHandle, amode XActivationMode, tmode TrainingMode, dtype DataType, coef float64) (*XActivationD, error) {

	ctr := uint64(1)
	var ktf kernels.XtraKerns
	switch amode {
	case XActivationModeFlag{}.Parametric():
		fwdmode, err := Cuda{}.MakeKernel(amode.tostringfwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		bwdmode, err := Cuda{}.MakeKernel(amode.tostringbwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		rmodek, err := Cuda{}.MakeKernel(ktf.Batch(), h.mod)
		if err != nil {
			return nil, err
		}
		tmodek, err := Cuda{}.MakeKernel(tmode.tostring(), h.mod)
		if err != nil {
			return nil, err
		}
		return &XActivationD{
			fwdmode: fwdmode,
			bwdmode: bwdmode,
			rmodek:  rmodek,
			tmodek:  tmodek,
			amode:   amode,
			tmode:   tmode,
			counter: ctr,
		}, nil
	default:
		fwdmode, err := Cuda{}.MakeKernel(amode.tostringfwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		bwdmode, err := Cuda{}.MakeKernel(amode.tostringbwd(dtype), h.mod)
		if err != nil {
			return nil, err
		}
		return &XActivationD{
			fwdmode: fwdmode,
			bwdmode: bwdmode,
			coef:    coef,
			amode:   amode,
		}, nil
	}

}

//ForwardProp does the feed forward operation alphas can be nil iff it is not supported (leaky right now).  otherwise it needs to be the same size as x and y
func (xA *XActivationD) ForwardProp(h *XHandle, blocksize uint32, xD *TensorD, x Memer, yD *TensorD, y Memer, alphas Memer) error {
	dtype, _, _, err := xD.GetDescrptor()
	sizeinbytes, err := xD.GetSizeInBytes()
	if err != nil {
		return err
	}
	length := FindLength(sizeinbytes, dtype)
	gridsize := kernels.SimpleGridCalculator(blocksize, length)

	switch xA.amode {
	case XActivationModeFlag{}.Leaky():
		return xA.fwdmode.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, h.s, length, x, y, float32(xA.coef))
	case XActivationModeFlag{}.Parametric():
		return xA.fwdmode.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, h.s, length, x, y, alphas)
	}
	return errors.New("Unsupported XActivationMode")
}

//BackProp does the feed forward operation alphas and dalphas can be nil iff it is not supported (leaky right now).  otherwise it needs to be the same size as x and y(parametric right now)
func (xA *XActivationD) BackProp(h *XHandle, blocksize uint32, dxD *TensorD, dx Memer, dyD *TensorD, dy Memer, alphas, dalphas Memer) error {
	dtype, _, _, err := dxD.GetDescrptor()
	sizeinbytes, err := dxD.GetSizeInBytes()
	if err != nil {
		return err
	}
	length := FindLength(sizeinbytes, dtype)
	gridsize := kernels.SimpleGridCalculator(blocksize, length)

	switch xA.amode {
	case XActivationModeFlag{}.Leaky():
		return xA.bwdmode.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, h.s, length, dx, dy, float32(xA.coef))
	case XActivationModeFlag{}.Parametric():

		if alphas == nil || dalphas == nil {

			return errors.New("alphas or daphas are nil")

		}
		return xA.bwdmode.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, h.s, length, dx, dy, alphas, dalphas)
	}
	return errors.New("Unsupported XActivationMode")
}

//UpdateAlphas will update the alphas using the optimizer specified.  Adagrad doesn't use xsum so that can be nil if using adagrad.
func (xA *XActivationD) UpdateAlphas(h *XHandle, blocksize uint32, batch int, dxD *TensorD, alphas, dalphas, xsum, gsum Memer, t TrainingParams) error {
	var dtf DataTypeFlag
	dtype, _, _, err := dxD.GetDescrptor()
	if dtype != dtf.Float() {
		return errors.New("only supports Float or float32 data type")
	}
	sizeinbytes, err := dxD.GetSizeInBytes()
	if err != nil {
		return err
	}
	length := FindLength(sizeinbytes, dtype)
	gridsize := kernels.SimpleGridCalculator(blocksize, length)
	if xA.rmodek == nil {
		return errors.New("regularization mode not set this is internal and if not using parmetric activation then you shouldn't update the alphas")
	}

	fbatch := float32(batch)

	err = xA.rmodek.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, h.s, length, dalphas, fbatch)
	if err != nil {
		return err
	}
	switch xA.tmode {
	case TrainingModeFlag{}.Adam():
		ctr := float32(xA.counter)
		err = xA.tmodek.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, h.s, length, alphas, gsum, xsum, dalphas, t.rate, t.beta1, t.beta2, t.eps, ctr)

		///void adamfloat(const int length,float *w,float *gsum,float *xsum,float *dw,const float rate,const float beta1,const float beta2,const float eps,const float counter){
		xA.counter++
		return err
	case TrainingModeFlag{}.AdaDelta():
		return xA.tmodek.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, h.s, length, alphas, gsum, xsum, dalphas, t.rate, t.eps)
	case TrainingModeFlag{}.AdaGrad():
		return xA.tmodek.Launch(gridsize, 1, 1, blocksize, 1, 1, 0, h.s, length, alphas, dalphas, gsum, t.rate, t.eps)

	}

	return errors.New("Unsupported Update")
}

/*

extern "C" __global__
void adadeltafloat( const int length,
                    float *weights, //weights input and output
                    float *gsum, //storage
                    float *xsum, //storage
                    float *dw, //input and will have to set to zero
                    const float rate, //input
                    const float eps){



            int section = blockIdx.x;
            int index = threadIdx.x;
            int cell = section*blockDim.x +index;
if(cell<length){
    gsum[cell]= gsum[cell]+(dw[cell]*dw[cell]);
    weights[cell]= -(rate*dw[cell])/(sqrtf(gsum[cell])+eps);
    dw[cell]=0.0;


}

}

extern "C" __global__
void adagradfloat(const int length,
                  float *weights, //weights input and output
                  float *dw, //input and will have to set to zero
                  float *gsum, //storage
                  const float rate, //input
                  const float eps){ //input


    int section = blockIdx.x;
    int index = threadIdx.x;
    int cell = section*blockDim.x +index;
    if (cell<length){
        int holder = gsum[cell];
        gsum[cell]= holder +(dw[cell]*dw[cell]);
        weights[cell]= -(rate*dw[cell])/(sqrtf(gsum[cell])+eps);
        dw[cell]=0.0;
    }

}

extern "C" __global__
void adamfloat(const int length,
               float *w,
               float *gsum,
               float *xsum,
               float *dw,
               const float rate,
               const float beta1,
               const float beta2,
               const float eps,
               const float counter){


    int i = (blockIdx.y*gridDim.x*blockDim.x) +
    (blockIdx.x*blockDim.x) +
    threadIdx.x;

if (i<length){
     gsum[i]=(beta1*gsum[i]) +((1.0-beta1)*dw[i]);
    float gsumt = 0;
    gsumt = gsum[i]/(1.0- powf(beta1,counter));
     xsum[i]= (beta2*xsum[i])+((1.0 -beta2)*(dw[i]*dw[i]));
    float xsumt =0;
    xsumt = xsum[i]/(1.0 - powf(beta2,counter));
    dw[i]=0.0;

}

}

extern "C" __global__
void batchregfloat(
    const int length,
    float *dw, //input and output
    const float batch) {// should be an int but just send it as a float
        int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
    if (i<length){

    dw[i]= dw[i]/batch;
    }

}

extern "C" __global__
void backwardleakyfloat(const int length,float *dx,float *dy, const float alpha){
int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
if (i<length){
    if (dy[i]>0.0){
        dx[i]=1.0;
    }else{
        dx[i]=alpha;
    }

}

}
extern "C" __global__
void backwardParametricfloat(const int length,float *dx,float *dy,  float *alpha, float *dalpha){

    int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;

if (i<length){
    if (dy[i]>0.0){
        dx[i]=1.0;
    }else{
        dx[i]=alpha[i];
        dalpha[i]+=alpha[i]*dy[i];
    }
}
}

extern "C" __global__
void forwardParametricfloat(const int length ,float *x,float *y,  float *alpha){

int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
if (i<length){
    if (x[i]>0.0){
        y[i]=x[i];
    }else{
        y[i]=alpha[i]*x[i];
    }

}


extern "C" __global__
void forwardleakyfloat(const int length,float *x,float *y, const float alpha){

int i=  (blockIdx.y*gridDim.x*blockDim.x) +(blockIdx.x*blockDim.x) + threadIdx.x;
if (i<length){
    if (x[i]>0.0){
        y[i]=x[i];
    }else{
        y[i]=alpha*x[i];
    }

}
*/
