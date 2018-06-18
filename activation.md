---
description: >-
  This page covers activation structs, flags, functions, methods, and handle
  methods used for Activation.
---

# Activation

## type ActivationD struct

ActivationD is an opaque struct that holds the description of an activation operation.

### func\(\*ActivationD\) GetDescriptor\(\)\(ActivationMode,PropagationNan,CDouble,error\)

GetDescriptor method returns the values that were used when making the descriptor through the **NewActivationDescriptor** function.

### func\(\*ActivationD\)DestroyDescriptor\(\)\(error\)

DestroyDescriptor destroys the ActivationD. 

## func NewActivationDescriptor \(mode ActivationMode, nan PropagationNan, coef CDouble\)\(\*ActivationD, error\)

This function creates and sets an Activation Descriptor and returns an \*ActivationD and error.'

* mode - specifies activation mode by passing an **ActivationMode** Flag
* nan - specifies the nan propigation mode by passing a **PropagationNAN** flag
* coef - is a number that will specify the clipping threshold when the **ActivationMode** is set to **ClippedRelu**, or if **ActivationMode** is set to **Elu** then it will be the alpha coefficient. 

 

## type ActivationModeFlag struct

ActivationModeFlag is a nil struct that is used to pass ActivationMode flags through methods.  Explanation will be given for Identity method because it is a special case.  Methods include:

* **func \(ActivationModeFlag\) Sigmoid\(\) ActivationMode**
* **func \(ActivationModeFlag\) Relu\(\) ActivationMode**
* **func \(ActivationModeFlag\) Tanh\(\) ActivationMode**
* **func \(ActivationModeFlag\) ClippedRelu\(\) ActivationMode**
* **func \(ActivationModeFlag\) Elu\(\) ActivationMode**
* f**unc \(ActivationModeFlag\) Identity\(\) ActivationMode**
  * This can only be used for ConvolutionBiasActivationForward\(\)
  * Will not work for ActivationForward\(\) or ActivationBackward\(\)

