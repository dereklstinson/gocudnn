package gocudnn

/*
#include <curand.h>
*/
import "C"
import "fmt"

/*

Note, I probably grabbed way to many flags from curand.h

*/
/*

  CURAND METHOD

*/

//CuRandMethod are used for flags
type CuRandMethod C.curandMethod_t

func (m CuRandMethod) c() C.curandMethod_t {
	return C.curandMethod_t(m)
}

//CuRandMethodFlag passes CuRandMethod flags through methods
type CuRandMethodFlag struct {
}

//Best returns best flag
func (m CuRandMethodFlag) Best() CuRandMethod {
	return CuRandMethod(C.CURAND_CHOOSE_BEST)
}

//ITR returns ITR flag
func (m CuRandMethodFlag) ITR() CuRandMethod {
	return CuRandMethod(C.CURAND_ITR)
}

//KNUTH returns KNUTH flag
func (m CuRandMethodFlag) KNUTH() CuRandMethod {
	return CuRandMethod(C.CURAND_KNUTH)
}

//HITR returns HITR flag
func (m CuRandMethodFlag) HITR() CuRandMethod {
	return CuRandMethod(C.CURAND_HITR)
}

//M1 returns M1 flag
func (m CuRandMethodFlag) M1() CuRandMethod {
	return CuRandMethod(C.CURAND_M1)
}

//M2 returns M2 flag
func (m CuRandMethodFlag) M2() CuRandMethod {
	return CuRandMethod(C.CURAND_M2)
}

//SEARCH returns SEARCH flag
func (m CuRandMethodFlag) SEARCH() CuRandMethod {
	return CuRandMethod(C.CURAND_BINARY_SEARCH)
}

//GAUSS returns GAUSS flag
func (m CuRandMethodFlag) GAUSS() CuRandMethod {
	return CuRandMethod(C.CURAND_DISCRETE_GAUSS)
}

//Rejection returns Rejection flag
func (m CuRandMethodFlag) Rejection() CuRandMethod {
	return CuRandMethod(C.CURAND_REJECTION)
}

//DeviceAPI returns DeviceAPI flag
func (m CuRandMethodFlag) DeviceAPI() CuRandMethod {
	return CuRandMethod(C.CURAND_DEVICE_API)
}

//FastRejection returns FastRejection flag
func (m CuRandMethodFlag) FastRejection() CuRandMethod {
	return CuRandMethod(C.CURAND_FAST_REJECTION)
}

//Third returns Third flag
func (m CuRandMethodFlag) Third() CuRandMethod {
	return CuRandMethod(C.CURAND_3RD)
}

//Definition returns Definition flag
func (m CuRandMethodFlag) Definition() CuRandMethod {
	return CuRandMethod(C.CURAND_DEFINITION)
}

//Poisson returns Poisson flag
func (m CuRandMethodFlag) Poisson() CuRandMethod {
	return CuRandMethod(C.CURAND_POISSON)
}

/*

CURAND choice of direction vector set


*/

//CuRandDirectionVectorSet are used for flags
type CuRandDirectionVectorSet C.curandDirectionVectorSet_t

func (o CuRandDirectionVectorSet) c() C.curandDirectionVectorSet_t {
	return C.curandDirectionVectorSet_t(o)
}

//CuRandDirectionVectorSetFlag pass CuRandDirectionVectorSet flags through methods
type CuRandDirectionVectorSetFlag struct {
}

//Vector32JoeKuo6 -- Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
func (d CuRandDirectionVectorSetFlag) Vector32JoeKuo6() CuRandDirectionVectorSet {
	return CuRandDirectionVectorSet(C.CURAND_DIRECTION_VECTORS_32_JOEKUO6)
}

//ScrambledVector32JoeKuo6 --Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000
func (d CuRandDirectionVectorSetFlag) ScrambledVector32JoeKuo6() CuRandDirectionVectorSet {
	return CuRandDirectionVectorSet(C.CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
}

//Vector64JoeKuo6 -- Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
func (d CuRandDirectionVectorSetFlag) Vector64JoeKuo6() CuRandDirectionVectorSet {
	return CuRandDirectionVectorSet(C.CURAND_DIRECTION_VECTORS_64_JOEKUO6)
}

//ScrambledVector64JoeKuo6 -- Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
func (d CuRandDirectionVectorSetFlag) ScrambledVector64JoeKuo6() CuRandDirectionVectorSet {
	return CuRandDirectionVectorSet(C.CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)
}

/*

Ordering Flags


*/

//CuRandOrdering are flags for CURAND ordering of results in memory
type CuRandOrdering C.curandOrdering_t

func (o CuRandOrdering) c() C.curandOrdering_t {
	return C.curandOrdering_t(o)
}

//CuRandOrderingFlag returns CuRandOrdering through methods
type CuRandOrderingFlag struct {
}

//PseudoBest returns PseudoBest flag
func (o CuRandOrderingFlag) PseudoBest() CuRandOrdering {
	return CuRandOrdering(C.CURAND_ORDERING_PSEUDO_BEST)
}

//PseudoDefault returns PseudoDefault flag
func (o CuRandOrderingFlag) PseudoDefault() CuRandOrdering {
	return CuRandOrdering(C.CURAND_ORDERING_PSEUDO_DEFAULT)
}

//PseudoSeeded returns PseudoSeeded flag
func (o CuRandOrderingFlag) PseudoSeeded() CuRandOrdering {
	return CuRandOrdering(C.CURAND_ORDERING_PSEUDO_SEEDED)
}

//QuasiDefault returns QuasiDefault flag
func (o CuRandOrderingFlag) QuasiDefault() CuRandOrdering {
	return CuRandOrdering(C.CURAND_ORDERING_QUASI_DEFAULT)
}

/*


CuRand Status/Error stuff


*/

//This handles the errors of curand
type curandstatus C.curandStatus_t

func (c curandstatus) error(Comment string) error {
	if c == curandstatus(C.CURAND_STATUS_SUCCESS) {
		return nil
	}
	return fmt.Errorf("%s -- %s", Comment, c.Error())
}

//Error is the error for currand
func (c curandstatus) Error() string {
	switch c {
	case curandstatus(C.CURAND_STATUS_VERSION_MISMATCH):
		return "CURAND_STATUS_VERSION_MISMATCH"
	case curandstatus(C.CURAND_STATUS_NOT_INITIALIZED):
		return "CURAND_STATUS_NOT_INITIALIZED"
	case curandstatus(C.CURAND_STATUS_ALLOCATION_FAILED):
		return "CURAND_STATUS_ALLOCATION_FAILED"
	case curandstatus(C.CURAND_STATUS_TYPE_ERROR):
		return "CURAND_STATUS_TYPE_ERROR"
	case curandstatus(C.CURAND_STATUS_OUT_OF_RANGE):
		return "CURAND_STATUS_OUT_OF_RANGE"
	case curandstatus(C.CURAND_STATUS_LENGTH_NOT_MULTIPLE):
		return "CURAND_STATUS_LENGTH_NOT_MULTIPLE"
	case curandstatus(C.CURAND_STATUS_DOUBLE_PRECISION_REQUIRED):
		return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"
	case curandstatus(C.CURAND_STATUS_LAUNCH_FAILURE):
		return "CURAND_STATUS_LAUNCH_FAILURE"
	case curandstatus(C.CURAND_STATUS_PREEXISTING_FAILURE):
		return "CURAND_STATUS_PREEXISTING_FAILURE"
	case curandstatus(C.CURAND_STATUS_INITIALIZATION_FAILED):
		return "CURAND_STATUS_INITIALIZATION_FAILED"
	case curandstatus(C.CURAND_STATUS_ARCH_MISMATCH):
		return "CURAND_STATUS_ARCH_MISMATCH"
	case curandstatus(C.CURAND_STATUS_INTERNAL_ERROR):
		return "CURAND_STATUS_INTERNAL_ERROR"
	default:
		return "Unknown Error. Problably got a CURAND_STATUS_SUCCESS and haven't accounted for it in the function  "
	}
}
