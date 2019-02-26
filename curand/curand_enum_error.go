package curand

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

//Method are used for flags and are passed through methds
type Method C.curandMethod_t

func (m Method) c() C.curandMethod_t {
	return C.curandMethod_t(m)
}

//Best returns best flag
func (m Method) Best() Method {
	return Method(C.CURAND_CHOOSE_BEST)
}

//ITR returns ITR flag
func (m Method) ITR() Method {
	return Method(C.CURAND_ITR)
}

//KNUTH returns KNUTH flag
func (m Method) KNUTH() Method {
	return Method(C.CURAND_KNUTH)
}

//HITR returns HITR flag
func (m Method) HITR() Method {
	return Method(C.CURAND_HITR)
}

//M1 returns M1 flag
func (m Method) M1() Method {
	return Method(C.CURAND_M1)
}

//M2 returns M2 flag
func (m Method) M2() Method {
	return Method(C.CURAND_M2)
}

//SEARCH returns SEARCH flag
func (m Method) SEARCH() Method {
	return Method(C.CURAND_BINARY_SEARCH)
}

//GAUSS returns GAUSS flag
func (m Method) GAUSS() Method {
	return Method(C.CURAND_DISCRETE_GAUSS)
}

//Rejection returns Rejection flag
func (m Method) Rejection() Method {
	return Method(C.CURAND_REJECTION)
}

//DeviceAPI returns DeviceAPI flag
func (m Method) DeviceAPI() Method {
	return Method(C.CURAND_DEVICE_API)
}

//FastRejection returns FastRejection flag
func (m Method) FastRejection() Method {
	return Method(C.CURAND_FAST_REJECTION)
}

//Third returns Third flag
func (m Method) Third() Method {
	return Method(C.CURAND_3RD)
}

//Definition returns Definition flag
func (m Method) Definition() Method {
	return Method(C.CURAND_DEFINITION)
}

//Poisson returns Poisson flag
func (m Method) Poisson() Method {
	return Method(C.CURAND_POISSON)
}

/*

CURAND choice of direction vector set


*/

//DirectionVectorSet are used for flags
type DirectionVectorSet C.curandDirectionVectorSet_t

func (d DirectionVectorSet) c() C.curandDirectionVectorSet_t {
	return C.curandDirectionVectorSet_t(d)
}

//Vector32JoeKuo6 -- Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
func (d DirectionVectorSet) Vector32JoeKuo6() DirectionVectorSet {
	return DirectionVectorSet(C.CURAND_DIRECTION_VECTORS_32_JOEKUO6)
}

//ScrambledVector32JoeKuo6 --Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000
func (d DirectionVectorSet) ScrambledVector32JoeKuo6() DirectionVectorSet {
	return DirectionVectorSet(C.CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
}

//Vector64JoeKuo6 -- Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
func (d DirectionVectorSet) Vector64JoeKuo6() DirectionVectorSet {
	return DirectionVectorSet(C.CURAND_DIRECTION_VECTORS_64_JOEKUO6)
}

//ScrambledVector64JoeKuo6 -- Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
func (d DirectionVectorSet) ScrambledVector64JoeKuo6() DirectionVectorSet {
	return DirectionVectorSet(C.CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)
}

/*

Ordering Flags


*/

//Ordering are flags for CURAND ordering of results in memory flags are set through methods type holds
type Ordering C.curandOrdering_t

func (o Ordering) c() C.curandOrdering_t {
	return C.curandOrdering_t(o)
}

//PseudoBest returns PseudoBest flag
func (o Ordering) PseudoBest() Ordering {
	return Ordering(C.CURAND_ORDERING_PSEUDO_BEST)
}

//PseudoDefault returns PseudoDefault flag
func (o Ordering) PseudoDefault() Ordering {
	return Ordering(C.CURAND_ORDERING_PSEUDO_DEFAULT)
}

//PseudoSeeded returns PseudoSeeded flag
func (o Ordering) PseudoSeeded() Ordering {
	return Ordering(C.CURAND_ORDERING_PSEUDO_SEEDED)
}

//QuasiDefault returns QuasiDefault flag
func (o Ordering) QuasiDefault() Ordering {
	return Ordering(C.CURAND_ORDERING_QUASI_DEFAULT)
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
