package gocudnn

/*
#include <cudnn.h>

cudnnStatus_t ConcatActivationForward4DNCHW(
								 cudnnHandle_t handle,
								 cudnnActivationDescriptor_t        activationDesc,
								 const void						    *alpha,
								 const void 						*beta,
								 const cudnnTensorDescriptor_t       xDescA,
                                 const void                         *xA,
						         const cudnnTensorDescriptor_t       xDescB,
                                 const void                         *xB,
	      						 const cudnnTensorDescriptor_t       yDescC, //Descriptor describing All of y.
								 void                               *y){


		int nA,nB,cA,cB,hA,hB,wA,wB,sn,sc,sh,sw ,cC;
		cudnnDataType_t dtype;
        cudnnStatus_t status;
	status=	cudnnGetTensor4dDescriptor(xDescA,&dtype,&nA,&cA,&hA,&wA,&sn,&sc,&sh,&sw);
  	if (status!=CUDNN_STATUS_SUCCESS){
	  return status;
	  }
	      int chwA = cA*hA*wA;
  	status=	cudnnGetTensor4dDescriptor(xDescB,&dtype,&nB,&cB,&hB,&wB,&sn,&sc,&sh,&sw);
	if (status!=CUDNN_STATUS_SUCCESS){
	  return status;
	  }
	  int chwB=cB*hB*wB;

  	if (nA!=nB || hA!=hB || wA!=wB){
	  return CUDNN_STATUS_BAD_PARAM;
	}


	cudnnTensorDescriptor_t tempdescXA,tempdescXB;
	status= cudnnCreateTensorDescriptor(&tempdescXA);
			if (status!=CUDNN_STATUS_SUCCESS){
              return status;
        	}

	status= cudnnSetTensor4dDescriptor(tempdescXA,  CUDNN_TENSOR_NCHW,dtype,1,cA,hA,wA);
          	if (status!=CUDNN_STATUS_SUCCESS){
              return status;
        	}

	status= cudnnCreateTensorDescriptor(&tempdescXB);
			if (status!=CUDNN_STATUS_SUCCESS){
              return status;
        	}

	status= cudnnSetTensor4dDescriptor(tempdescXB,  CUDNN_TENSOR_NCHW,dtype,1,cB,hB,wB);
          	if (status!=CUDNN_STATUS_SUCCESS){
              return status;
        	}
   status=	cudnnGetTensor4dDescriptor(yDescC,&dtype,&nB,&cC,&hB,&wB,&sn,&sc,&sh,&sw);
    if (nA!=nB || hA!=hB || wA!=wB || cC!=cA+cB){
	  return CUDNN_STATUS_BAD_PARAM;
	}
	int chwC=hB*wB*cC;
    int chwCstep=chwA+chwC;

        for (int i =0;i<nA;i++){

       status =cudnnActivationForward(handle,activationDesc,alpha,tempdescXA,xA+(i*chwA),beta,tempdescXA,y +(i*chwC));
       status =cudnnActivationForward(handle,activationDesc,alpha,tempdescXB,xB+(i*chwB),beta,tempdescXB,y +(i*chwCstep));


        	  }
cudnnDestroyTensorDescriptor(tempdescXB);
return cudnnDestroyTensorDescriptor(tempdescXA);

	}









*/
import "C"

//Activation is a helper func that is used is activation type processes
type Activation struct {
	Flgs ActivationModeFlag
}

//ActivationD is an opaque struct that holds the description of an activation operation.
type ActivationD struct {
	descriptor C.cudnnActivationDescriptor_t
	coef       C.double //This is used for ceiling for clipped relu, and alpha for elu
}

//NewActivationDescriptor creates and sets and returns an activation descriptor in ActivationD and the error
func (new Activation) NewActivationDescriptor(mode ActivationMode, nan PropagationNAN, coef float64) (*ActivationD, error) {

	var descriptor C.cudnnActivationDescriptor_t
	err := Status(C.cudnnCreateActivationDescriptor(&descriptor)).error("NewActivationDescriptor-create")
	if err != nil {
		return nil, err
	}

	err = Status(C.cudnnSetActivationDescriptor(descriptor, mode.c(), nan.c(), C.double(coef))).error("NewActivationDescriptor-set")
	if err != nil {
		return nil, err
	}
	return &ActivationD{
		descriptor: descriptor,
		coef:       C.double(coef),
	}, err
}

//GetDescriptor gets the descriptor info for ActivationD
func (a *ActivationD) GetDescriptor() (ActivationMode, PropagationNAN, float64, error) {
	var coef C.double
	var mode C.cudnnActivationMode_t
	var nan C.cudnnNanPropagation_t
	err := Status(C.cudnnGetActivationDescriptor(a.descriptor, &mode, &nan, &coef)).error("GetDescriptor")
	return ActivationMode(mode), PropagationNAN(nan), float64(coef), err
}

//DestroyDescriptor destroys the activation descriptor
func (a *ActivationD) DestroyDescriptor() error {
	return Status(C.cudnnDestroyActivationDescriptor(a.descriptor)).error("DestroyDescriptor")
}

//ActivationForward does the forward activation function yrtn is returned and changed.
func (a *ActivationD) Forward(
	handle *Handle,
	alpha CScalar,
	xD *TensorD,
	x Memer,
	beta CScalar,
	yD *TensorD,
	yrtn Memer) error {
	return Status(C.cudnnActivationForward(handle.x, a.descriptor, alpha.CPtr(), xD.descriptor, x.Ptr(), beta.CPtr(), yD.descriptor, yrtn.Ptr())).error("ActivationForward")
}

//ActivationBackward does the activation backward method
func (a *ActivationD) Backward(
	handle *Handle,
	alpha CScalar,
	yD *TensorD,
	y Memer,
	dyD *TensorD,
	dy Memer,
	xD *TensorD,
	x Memer,
	beta CScalar,
	dxD *TensorD,
	dx Memer) error {
	return Status(C.cudnnActivationBackward(handle.x, a.descriptor, alpha.CPtr(), yD.descriptor, y.Ptr(), dyD.descriptor, dy.Ptr(), xD.descriptor, x.Ptr(), beta.CPtr(), dxD.descriptor, dx.Ptr())).error("ActivationBackward")
}

//ActivationModeFlag is used to "safely" pass flags by the use of methods.
//If it was me I would call it like:
//var ActMode gocudnn.ActivationModeFlag
//Then use the methods like ActMode.Sigmoid() to pass the sigmoid flag.
type ActivationModeFlag struct {
}

//ActivationMode is used for activation discriptor flags use ActivationModeFlag struct to pass flags through its methods
type ActivationMode C.cudnnActivationMode_t

//Sigmoid returnsActivationMode(C.CUDNN_ACTIVATION_SIGMOID)
func (a ActivationModeFlag) Sigmoid() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_SIGMOID)
}

//Relu returns ActivationMode(C.CUDNN_ACTIVATION_RELU)
func (a ActivationModeFlag) Relu() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_RELU)
}

//Tanh returns ActivationMode(C.CUDNN_ACTIVATION_TANH)
func (a ActivationModeFlag) Tanh() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_TANH)
}

//ClippedRelu returns  ActivationMode(C.CUDNN_ACTIVATION_CLIPPED_RELU)
func (a ActivationModeFlag) ClippedRelu() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_CLIPPED_RELU)
}

//Elu returns  ActivationMode(C.CUDNN_ACTIVATION_ELU)
func (a ActivationModeFlag) Elu() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_ELU)
}

//Identity returns  ActivationMode(C.CUDNN_ACTIVATION_IDENTITY)
func (a ActivationModeFlag) Identity() ActivationMode {
	return ActivationMode(C.CUDNN_ACTIVATION_IDENTITY)
}
func (a ActivationMode) c() C.cudnnActivationMode_t { return C.cudnnActivationMode_t(a) }
