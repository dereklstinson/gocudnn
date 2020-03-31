package curand

/*
#include <curand.h>
*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/cutil"
)

//Generator is a random number generator for the device.
type Generator struct {
	w         *gocu.Worker
	generator C.curandGenerator_t
	gentype   RngType
	gogc      bool
}

//CreateGeneratorEx creates a generater where the operations will flow through a worker
func CreateGeneratorEx(w *gocu.Worker, gentype RngType) *Generator {
	if w == nil {
		return CreateGenerator(gentype)
	}
	var g = new(Generator)
	g.gentype = gentype
	g.gogc = true
	g.w = w

	err := w.Work(func() error {

		errsub := curandstatus(C.curandCreateGenerator(&g.generator, g.gentype.c())).error("NewGenerator-create")
		if errsub != nil {
			return errsub
		}

		runtime.SetFinalizer(g, curandDestroyGenerator)
		return nil
	})
	if err != nil {
		panic(err)
	}

	return g
}

//CreateGenerator creates a Generator
func CreateGenerator(gentype RngType) *Generator {
	var generator C.curandGenerator_t
	err := curandstatus(C.curandCreateGenerator(&generator, gentype.c())).error("NewGenerator-create")
	if err != nil {
		panic(err)
	}
	g := &Generator{
		generator: generator,
		gentype:   gentype,
		gogc:      true,
	}
	runtime.SetFinalizer(g, curandDestroyGenerator)
	return g
}

func curandDestroyGenerator(g *Generator) error {
	return curandstatus(C.curandDestroyGenerator(g.generator)).error("curandDestroyGenerator")
}

//Destroy destroys the random generator.  This doesn't do anything right now
func (c *Generator) Destroy() error {
	if c.gogc {
		return nil
	}
	return curandstatus(C.curandDestroyGenerator(c.generator)).error("(c *Generator) Destroy()")
}

//SetStream sets the a cuda stream for the curand generator
func (c *Generator) SetStream(stream gocu.Streamer) error {
	if c.w != nil {
		return c.w.Work(func() error {
			return curandstatus(C.curandSetStream(c.generator, C.cudaStream_t(stream.Ptr()))).error("(c *Generator) SetStream")

		})
	}
	return curandstatus(C.curandSetStream(c.generator, C.cudaStream_t(stream.Ptr()))).error("(c *Generator) SetStream")
}

//SetPsuedoSeed sets the seed for the curand generator
func (c *Generator) SetPsuedoSeed(seed uint64) error {
	if c.w != nil {
		return c.w.Work(func() error {
			return curandstatus(C.curandSetPseudoRandomGeneratorSeed(c.generator, C.ulonglong(seed))).error("(c *Generator) SetPsuedoSeed")
		})
	}
	return curandstatus(C.curandSetPseudoRandomGeneratorSeed(c.generator, C.ulonglong(seed))).error("(c *Generator) SetPsuedoSeed")
}

//Uint fills mem with random numbers
/*
From cuRAND documentation:
The curandGenerate() function is used to generate pseudo- or quasirandom bits of output for XORWOW, MRG32k3a, MTGP32, MT19937, Philox_4x32_10 and SOBOL32 generators. Each output element is a 32-bit unsigned int where all bits are random. For SOBOL64 generators, each output element is a 64-bit unsigned long long where all bits are random. curandGenerate() returns an error for SOBOL64 generators. Use curandGenerateLongLong() to generate 64 bit integers with the SOBOL64 generators.
//values need to be stored as an uint32
*/
func (c *Generator) Uint(mem cutil.Mem, sizeinbytes uint) error {
	if c.w != nil {
		return c.w.Work(func() error {
			return curandstatus(C.curandGenerate(c.generator, (*C.uint)(mem.Ptr()), C.size_t(sizeinbytes))).error("(c *Generator) Uint")
		})
	}
	return curandstatus(C.curandGenerate(c.generator, (*C.uint)(mem.Ptr()), C.size_t(sizeinbytes))).error("(c *Generator) Uint")
}

//Uint64 fills mem with  unsigned long long random numbers
/*
From cuRAND documentation:
The curandGenerate() function is used to generate pseudo- or quasirandom bits of output for XORWOW, MRG32k3a, MTGP32, MT19937, Philox_4x32_10 and SOBOL32 generators. Each output element is a 32-bit unsigned int where all bits are random. For SOBOL64 generators, each output element is a 64-bit unsigned long long where all bits are random. curandGenerate() returns an error for SOBOL64 generators. Use curandGenerateLongLong() to generate 64 bit integers with the SOBOL64 generators.
//values need to be stored as an uint32
*/
func (c *Generator) Uint64(mem cutil.Mem, sizeinbytes uint) error {
	if c.w != nil {
		return c.w.Work(func() error {
			return curandstatus(C.curandGenerateLongLong(c.generator, (*C.ulonglong)(mem.Ptr()), C.size_t(sizeinbytes))).error("(c *Generator) Uint64")
		})
	}
	return curandstatus(C.curandGenerateLongLong(c.generator, (*C.ulonglong)(mem.Ptr()), C.size_t(sizeinbytes))).error("(c *Generator) Uint64")
}

//UniformFloat32 - generates uniform distributions in float32
/*
from cuRAND documentation:
The curandGenerateUniform() function is used to generate uniformly distributed floating point values between 0.0 and 1.0, where 0.0 is excluded and 1.0 is included.
*/
func (c *Generator) UniformFloat32(mem cutil.Mem, sizeinbytes uint) error {
	if c.w != nil {
		return c.w.Work(func() error {
			return curandstatus(C.curandGenerateUniform(c.generator, (*C.float)(mem.Ptr()), C.size_t(sizeinbytes))).error("(c *Generator) UniformFloat32")
		})
	}
	return curandstatus(C.curandGenerateUniform(c.generator, (*C.float)(mem.Ptr()), C.size_t(sizeinbytes))).error("(c *Generator) UniformFloat32")
}

//NormalFloat32 -generates a Normal distribution in float32
/*
from cuRAND documentation:
The curandGenerateNormal() function is used to generate normally distributed floating point values with the given mean and standard deviation.
*/
func (c *Generator) NormalFloat32(mem cutil.Mem, sizeinbytes uint, mean, std float32) error {
	if c.w != nil {
		return c.w.Work(func() error {
			return curandstatus(C.curandGenerateNormal(c.generator, (*C.float)(mem.Ptr()), C.size_t(sizeinbytes), C.float(mean), C.float(std))).error("(c *Generator) NormalFloat32")
		})
	}
	return curandstatus(C.curandGenerateNormal(c.generator, (*C.float)(mem.Ptr()), C.size_t(sizeinbytes), C.float(mean), C.float(std))).error("(c *Generator) NormalFloat32")
}

/*

Generator Flags


*/

//RngType holds CURAND generator type flags
type RngType C.curandRngType_t

func (rng RngType) c() C.curandRngType_t {
	return C.curandRngType_t(rng)
}

//Test sets and returns test flag
func (rng *RngType) Test() RngType { *rng = RngType(C.CURAND_RNG_TEST); return *rng }

//PseudoDefault sets and returns PseudoDefault flag
func (rng *RngType) PseudoDefault() RngType { *rng = RngType(C.CURAND_RNG_PSEUDO_DEFAULT); return *rng }

//PseudoXORWOW sets and returns PseudoXORWOW flag
func (rng *RngType) PseudoXORWOW() RngType { *rng = RngType(C.CURAND_RNG_PSEUDO_XORWOW); return *rng }

//PseudoMRG32K3A sets and returns PseudoMRG32K3A flag
func (rng *RngType) PseudoMRG32K3A() RngType {
	*rng = RngType(C.CURAND_RNG_PSEUDO_MRG32K3A)
	return *rng
}

//PseudoMTGP32 sets and returns PseudoMTGP32 flag
func (rng *RngType) PseudoMTGP32() RngType { *rng = RngType(C.CURAND_RNG_PSEUDO_MTGP32); return *rng }

//PseudoMT19937 sets and  returns PseudoMT19937 flag
func (rng *RngType) PseudoMT19937() RngType { *rng = RngType(C.CURAND_RNG_PSEUDO_MT19937); return *rng }

//PseudoPhilox43210 sets and returns PseudoPhilox43210 flag
func (rng *RngType) PseudoPhilox43210() RngType {
	*rng = RngType(C.CURAND_RNG_PSEUDO_PHILOX4_32_10)
	return *rng
}

//QuasiDefault sets and returns QuasiDefault flag
func (rng *RngType) QuasiDefault() RngType { *rng = RngType(C.CURAND_RNG_QUASI_DEFAULT); return *rng }

//QuasiSOBOL32 sets and returns QuasiSOBOL32 flag
func (rng *RngType) QuasiSOBOL32() RngType { *rng = RngType(C.CURAND_RNG_QUASI_SOBOL32); return *rng }

//QuasiScrambledSOBOL32 sets and returns QuasiScrambledSOBOL32 flag
func (rng *RngType) QuasiScrambledSOBOL32() RngType {
	*rng = RngType(C.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32)
	return *rng
}

//QuasiSOBOL64 sets and returns QuasiSOBOL64 flag
func (rng *RngType) QuasiSOBOL64() RngType { *rng = RngType(C.CURAND_RNG_QUASI_SOBOL64); return *rng }

//QuasiScrambledSOBOL64 sets and returns QuasiScrambledSOBOL64 flag
func (rng *RngType) QuasiScrambledSOBOL64() RngType {
	*rng = RngType(C.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
	return *rng
}
func (rng RngType) String() string {
	f := rng
	var s string
	switch rng {
	case f.PseudoDefault():
		s = "PseudoDefault"
	case f.PseudoMRG32K3A():
		s = "PseudoMRG32K3A"
	case f.PseudoMTGP32():
		s = "PseudoMTGP32"
	case f.PseudoPhilox43210():
		s = "PseudoPhilox43210"
	case f.PseudoXORWOW():
		s = "PseudoXORWOW"
	case f.QuasiDefault():
		s = "QuasiDefault"
	case f.QuasiSOBOL32():
		s = "QuasiSOBOL32"
	case f.QuasiSOBOL64():
		s = "QuasiSOBOL64"
	case f.QuasiScrambledSOBOL32():
		s = "QuasiScrambledSOBOL32"
	case f.QuasiScrambledSOBOL64():
		s = "QuasiScrambledSOBOL64"
	default:
		s = "Unsupported Type"
	}
	return "RngType: " + s
}
