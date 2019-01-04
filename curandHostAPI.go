package gocudnn

/*
#include <curand.h>
*/
import "C"

//CuRandGenerator is a random number generator for the device.
type CuRandGenerator struct {
	generator C.curandGenerator_t
	gentype   CuRandRngType
}

//CreateCuRandGenerator creates a curandgenerator
func CreateCuRandGenerator(gentype CuRandRngType) *CuRandGenerator {
	var generator C.curandGenerator_t
	err := curandstatus(C.curandCreateGenerator(&generator, gentype.c())).error("NewCuRandGenerator-create")
	if err != nil {
		panic(err)
	}
	return &CuRandGenerator{
		generator: generator,
		gentype:   gentype,
	}
}

//Destroy destroys the random generator
func (c CuRandGenerator) Destroy() error {
	return curandstatus(C.curandDestroyGenerator(c.generator)).error("curandDestroyGenerator")
}

//SetStream sets the a cuda stream for the curand generator
func (c CuRandGenerator) SetStream(stream *Stream) error {
	return curandstatus(C.curandSetStream(c.generator, stream.stream)).error("curandSetStream")
}

//SetPsuedoSeed sets the seed for the curand generator
func (c CuRandGenerator) SetPsuedoSeed(seed uint64) error {
	return curandstatus(C.curandSetPseudoRandomGeneratorSeed(c.generator, C.ulonglong(seed))).error("SetPsuedoSeed")
}

//Uint fills mem with random numbers
/*
From cuRAND documentation:
The curandGenerate() function is used to generate pseudo- or quasirandom bits of output for XORWOW, MRG32k3a, MTGP32, MT19937, Philox_4x32_10 and SOBOL32 generators. Each output element is a 32-bit unsigned int where all bits are random. For SOBOL64 generators, each output element is a 64-bit unsigned long long where all bits are random. curandGenerate() returns an error for SOBOL64 generators. Use curandGenerateLongLong() to generate 64 bit integers with the SOBOL64 generators.
//values need to be stored as an uint32
*/
func (c CuRandGenerator) Uint(mem *Malloced) error {
	return curandstatus(C.curandGenerate(c.generator, (*C.uint)(mem.ptr), mem.size.c())).error("Generate")
}

//Uint64 fills mem with  unsigned long long random numbers
/*
From cuRAND documentation:
The curandGenerate() function is used to generate pseudo- or quasirandom bits of output for XORWOW, MRG32k3a, MTGP32, MT19937, Philox_4x32_10 and SOBOL32 generators. Each output element is a 32-bit unsigned int where all bits are random. For SOBOL64 generators, each output element is a 64-bit unsigned long long where all bits are random. curandGenerate() returns an error for SOBOL64 generators. Use curandGenerateLongLong() to generate 64 bit integers with the SOBOL64 generators.
//values need to be stored as an uint32
*/
func (c CuRandGenerator) Uint64(mem *Malloced) error {
	return curandstatus(C.curandGenerateLongLong(c.generator, (*C.ulonglong)(mem.ptr), mem.size.c())).error("Generate")
}

//UniformFloat32 - generates uniform distributions in float32
/*
from cuRAND documentation:
The curandGenerateUniform() function is used to generate uniformly distributed floating point values between 0.0 and 1.0, where 0.0 is excluded and 1.0 is included.
*/
func (c CuRandGenerator) UniformFloat32(mem *Malloced) error {
	return curandstatus(C.curandGenerateUniform(c.generator, (*C.float)(mem.ptr), mem.size.c())).error("Generate")
}

//NormalFloat32 -generates a Normal distribution in float32
/*
from cuRAND documentation:
The curandGenerateNormal() function is used to generate normally distributed floating point values with the given mean and standard deviation.
*/
func (c CuRandGenerator) NormalFloat32(mem *Malloced, mean, std float32) error {
	return curandstatus(C.curandGenerateNormal(c.generator, (*C.float)(mem.ptr), mem.size.c(), C.float(mean), C.float(std))).error("NormalFloat32")
}

/*

Generator Flags


*/

//CuRandRngType holds CURAND generator type flags
type CuRandRngType C.curandRngType_t

func (rng CuRandRngType) c() C.curandRngType_t {
	return C.curandRngType_t(rng)
}

//CuRandRngTypeFlag returns CuRandRngType through methods
type CuRandRngTypeFlag struct {
}

//Test returns test flag
func (rng CuRandRngTypeFlag) Test() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_TEST)
}

//PseudoDefault returns PseudoDefault flag
func (rng CuRandRngTypeFlag) PseudoDefault() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_PSEUDO_DEFAULT)
}

//PseudoXORWOW returns PseudoXORWOW flag
func (rng CuRandRngTypeFlag) PseudoXORWOW() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_PSEUDO_XORWOW)
}

//PseudoMRG32K3A returns PseudoMRG32K3A flag
func (rng CuRandRngTypeFlag) PseudoMRG32K3A() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_PSEUDO_MRG32K3A)
}

//PseudoMTGP32 returns PseudoMTGP32 flag
func (rng CuRandRngTypeFlag) PseudoMTGP32() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_PSEUDO_MTGP32)
}

//PseudoMT19937 returns PseudoMT19937 flag
func (rng CuRandRngTypeFlag) PseudoMT19937() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_PSEUDO_MT19937)
}

//PseudoPhilox43210 returns PseudoPhilox43210 flag
func (rng CuRandRngTypeFlag) PseudoPhilox43210() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_PSEUDO_PHILOX4_32_10)
}

//QuasiDefault returns QuasiDefault flag
func (rng CuRandRngTypeFlag) QuasiDefault() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_QUASI_DEFAULT)
}

//QuasiSOBOL32 returns QuasiSOBOL32 flag
func (rng CuRandRngTypeFlag) QuasiSOBOL32() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_QUASI_SOBOL32)
}

//QuasiScrambledSOBOL32 returns QuasiScrambledSOBOL32 flag
func (rng CuRandRngTypeFlag) QuasiScrambledSOBOL32() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32)
}

//QuasiSOBOL64 returns QuasiSOBOL64 flag
func (rng CuRandRngTypeFlag) QuasiSOBOL64() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_QUASI_SOBOL64)
}

//QuasiScrambledSOBOL64 returns QuasiScrambledSOBOL64 flag
func (rng CuRandRngTypeFlag) QuasiScrambledSOBOL64() CuRandRngType {
	return CuRandRngType(C.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
}
