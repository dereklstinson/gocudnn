package npp

//#include <nppdefs.h>
import "C"
import (
	"unsafe"
)

//Flags is a special struct that contains all the flag types npp uses.
//Even though these types are flags themselves.
//They can return flags with different values within the same type through methods
type Flags struct {
	InterpolationMode  InterpolationMode
	BayerGridPosition  BayerGridPosition
	MaskSize           MaskSize
	DifferentialKernel DifferentialKernel
	Axis               Axis
	CmpOp              CmpOp
	RoundMode          RoundMode
	BorderType         BorderType
	HintAlgorithm      HintAlgorithm
	AlphaOp            AlphaOp
	ZCType             ZCType
	HuffmanTableType   HuffmanTableType
	Norm               Norm
}

//GetFlags returns a Flags struct.  This struct has all the flag types within it.
func GetFlags() Flags {
	return Flags{}
}

//InterpolationMode is a wrapper for interpolation flags
type InterpolationMode C.NppiInterpolationMode

func (n InterpolationMode) c() C.NppiInterpolationMode {
	return (C.NppiInterpolationMode)(n)
}

func (n InterpolationMode) cint() C.int {
	return (C.int)(n)
}

//UNDEFINED sets and returns InterpolationMode(C.NPPI_INTER_UNDEFINED)
//
func (n *InterpolationMode) UNDEFINED() InterpolationMode {
	*n = InterpolationMode(C.NPPI_INTER_UNDEFINED)
	return *n
}

//NN returns InterpolationMode(C.NPPI_INTER_NN)
/**<  Nearest neighbor filtering. */
func (n *InterpolationMode) NN() InterpolationMode {
	*n = InterpolationMode(C.NPPI_INTER_NN)
	return *n
}

//LINEAR sets and returns InterpolationMode(C.NPPI_INTER_LINEAR)
/**<  Linear interpolation. */
func (n *InterpolationMode) LINEAR() InterpolationMode {
	*n = InterpolationMode(C.NPPI_INTER_LINEAR)
	return *n
}

//CUBIC sets and returns InterpolationMode(C.NPPI_INTER_CUBIC)
/**<  Cubic interpolation. */
func (n *InterpolationMode) CUBIC() InterpolationMode {
	*n = InterpolationMode(C.NPPI_INTER_CUBIC)
	return *n
}

//BSPLINE sets and returns InterpolationMode(C.NPPI_INTER_CUBIC2P_BSPLINE)
/**<  Two-parameter cubic filter (B=1, C=0) */
func (n *InterpolationMode) BSPLINE() InterpolationMode {
	*n = InterpolationMode(C.NPPI_INTER_CUBIC2P_BSPLINE)
	return *n
}

//CATMULLROM sets and returns InterpolationMode(C.NPPI_INTER_CUBIC2P_CATMULLROM)
/**<  Two-parameter cubic filter (B=0, C=1/2) */
func (n *InterpolationMode) CATMULLROM() InterpolationMode {
	*n = InterpolationMode(C.NPPI_INTER_CUBIC2P_CATMULLROM)
	return *n
}

//B05C03 sets and returns InterpolationMode(C.NPPI_INTER_CUBIC2P_B05C03)
/**<  Two-parameter cubic filter (B=1/2, C=3/10) */
func (n *InterpolationMode) B05C03() InterpolationMode {
	*n = InterpolationMode(C.NPPI_INTER_CUBIC2P_B05C03)
	return *n
}

//SUPER sets and returns InterpolationMode(C.NPPI_INTER_SUPER)
/**<  Super sampling. */
func (n *InterpolationMode) SUPER() InterpolationMode {
	*n = InterpolationMode(C.NPPI_INTER_SUPER)
	return *n
}

//LANCZOS sets and returns InterpolationMode(C.NPPI_INTER_LANCZOS)
/**<  Lanczos filtering. */
func (n *InterpolationMode) LANCZOS() InterpolationMode {
	*n = InterpolationMode(C.NPPI_INTER_LANCZOS)
	return *n
}

//LANCZ0S3ADVANCED sets and returns InterpolationMode(C.NPPI_INTER_LANCZOS3_ADVANCED)
/**<  Generic Lanczos filtering with order 3. */
func (n *InterpolationMode) LANCZ0S3ADVANCED() InterpolationMode {
	*n = InterpolationMode(C.NPPI_INTER_LANCZOS3_ADVANCED)
	return *n
}

//SMOOTHEDGE sets and returns InterpolationMode(C.NPPI_SMOOTH_EDGE)
/**<  Smooth edge filtering. */
func (n *InterpolationMode) SMOOTHEDGE() InterpolationMode {
	*n = InterpolationMode(C.NPPI_SMOOTH_EDGE)
	return *n
}

//BayerGridPosition is used as flags. Contains methods for different flags
type BayerGridPosition C.NppiBayerGridPosition

func (b BayerGridPosition) c() C.NppiBayerGridPosition { return C.NppiBayerGridPosition(b) }

//BGGR sets and returns BayerGridPosition(C.NPPI_BAYER_BGGR)
func (b *BayerGridPosition) BGGR() BayerGridPosition {
	*b = BayerGridPosition(C.NPPI_BAYER_BGGR)
	return *b
}

//RGGB sets and returns BayerGridPosition(NPPI_BAYER_RGGB)
func (b *BayerGridPosition) RGGB() BayerGridPosition {
	*b = BayerGridPosition(C.NPPI_BAYER_RGGB)
	return *b
}

//GRBG sets and returns BayerGridPosition(C.NPPI_BAYER_GRBG)
func (b *BayerGridPosition) GRBG() BayerGridPosition {
	*b = BayerGridPosition(C.NPPI_BAYER_GRBG)
	return *b
}

//GBRG sets and returns BayerGridPosition(NPPI_BAYER_GBRG)
func (b *BayerGridPosition) GBRG() BayerGridPosition {
	*b = BayerGridPosition(C.NPPI_BAYER_GBRG)
	return *b
}

//MaskSize has methods that are flags that are Fixed filter-kernel sizes.
type MaskSize C.NppiMaskSize

func (m MaskSize) c() C.NppiMaskSize { return C.NppiMaskSize(m) }

//Size1x3 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size1x3() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_1_X_3); return *m }

//Size1x5 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size1x5() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_1_X_5); return *m }

//Size3x1 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size3x1() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_3_X_1); return *m }

//Size5x1 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size5x1() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_5_X_1); return *m }

//Size3x3 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size3x3() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_3_X_3); return *m }

//Size5x5 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size5x5() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_5_X_5); return *m }

//Size7x7 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size7x7() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_7_X_7); return *m }

//Size9x9 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size9x9() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_9_X_9); return *m }

//Size11x11 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size11x11() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_11_X_11); return *m }

//Size13x13 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size13x13() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_13_X_13); return *m }

//Size15x15 sets and returns the NPP flag wrapped in MaskSize
func (m *MaskSize) Size15x15() MaskSize { *m = MaskSize(C.NPP_MASK_SIZE_15_X_15); return *m }

/**
 * Differential Filter types
 */

//DifferentialKernel wraps a C.NppiDifferentialKernel
type DifferentialKernel C.NppiDifferentialKernel

func (d DifferentialKernel) c() C.NppiDifferentialKernel { return C.NppiDifferentialKernel(d) }

//SOBEL sets and returns  DifferentialKernel(C.NPP_FILTER_SOBEL) flag
func (d *DifferentialKernel) SOBEL() DifferentialKernel {
	*d = DifferentialKernel(C.NPP_FILTER_SOBEL)
	return *d
}

//SCHARR sets and returns DifferentialKernel(C.NPP_FILTER_SCHARR) flag
func (d *DifferentialKernel) SCHARR() DifferentialKernel {
	*d = DifferentialKernel(C.NPP_FILTER_SCHARR)
	return *d
}

/*
 *
 * Float16
 *
 */

//Float16 is a half used by npp.
type Float16 C.Npp16f

func (n *Float16) cptr() *C.Npp16f {
	return (*C.Npp16f)(n)
}

//Ptr returns an unsafe pointer to this variable location. This is so it can be used with other cuda libraries like (cudnn, cudart, cuda, and such)
func (n *Float16) Ptr() unsafe.Pointer {
	return unsafe.Pointer(n)
}

/*
//DPtr returns an double pointer used for allocating memory on device
func (n *Float16) DPtr() *unsafe.Pointer {
	x := unsafe.Pointer(n)
	return (*unsafe.Pointer)(&x)
}
*/

/*
func convertUint64toCNpp64uarray(x []Uint64) []C.Npp64u {
	y := make([]C.Npp64u, len(x))
	for i := range x {
		y[i] = C.Npp64u(x[i])
	}
	return y
}
func convertCNpp64utoUint64array(x []C.Npp64u) []Uint64 {
	y := make([]Uint64, len(x))
	for i := range x {
		y[i] = Uint64(x[i])
	}
	return y
}
func convertUint32toCNpp32uarray(x []Uint32) []C.Npp32u {
	y := make([]C.Npp32u, len(x))
	for i := range x {
		y[i] = C.Npp32u(x[i])
	}
	return y
}
func convertCNpp32utoUint32array(x []C.Npp32u) []Uint32 {
	y := make([]Uint32, len(x))
	for i := range x {
		y[i] = Uint32(x[i])
	}
	return y
}

func convertUint16toCNpp16uarray(x []Uint16) []C.Npp16u {
	y := make([]C.Npp16u, len(x))
	for i := range x {
		y[i] = C.Npp16u(x[i])
	}
	return y
}
func convertCNpp16utoUint16array(x []C.Npp16u) []Uint16 {
	y := make([]Uint16, len(x))
	for i := range x {
		y[i] = Uint16(x[i])
	}
	return y
}

func convertNpp8utoCNpp8uarray(x []Uint8) []C.Npp8u {
	y := make([]C.Npp8u, len(x))
	for i := range x {
		y[i] = C.Npp8u(x[i].c())
	}
	return y
}
func convertCNpp8utoNpp8uarray(x []C.Npp8u) []Uint8 {
	y := make([]Uint8, len(x))
	for i := range x {
		y[i].wrap(&x[i])
	}
	return y
}
*/

/*


//#define NPP_MIN_8U      ( 0 )                        /**<  Minimum 8-bit unsigned integer */
//#define NPP_MAX_8U      ( 255 )                      /**<  Maximum 8-bit unsigned integer */
//#define NPP_MIN_16U     ( 0 )                        /**<  Minimum 16-bit unsigned integer */
//#define NPP_MAX_16U     ( 65535 )                    /**<  Maximum 16-bit unsigned integer */
//#define NPP_MIN_32U     ( 0 )                        /**<  Minimum 32-bit unsigned integer */
//#define NPP_MAX_32U     ( 4294967295U )              /**<  Maximum 32-bit unsigned integer */
//#define NPP_MIN_64U     ( 0 )                        /**<  Minimum 64-bit unsigned integer */
//#define NPP_MAX_64U     ( 18446744073709551615ULL )  /**<  Maximum 64-bit unsigned integer */

//#define NPP_MIN_8S      (-127 - 1 )                  /**<  Minimum 8-bit signed integer */
//#define NPP_MAX_8S      ( 127 )                      /**<  Maximum 8-bit signed integer */
//#define NPP_MIN_16S     (-32767 - 1 )                /**<  Minimum 16-bit signed integer */
//#define NPP_MAX_16S     ( 32767 )                    /**<  Maximum 16-bit signed integer */
//#define NPP_MIN_32S     (-2147483647 - 1 )           /**<  Minimum 32-bit signed integer */
//#define NPP_MAX_32S     ( 2147483647 )               /**<  Maximum 32-bit signed integer */
//#define NPP_MAX_64S     ( 9223372036854775807LL )    /**<  Maximum 64-bit signed integer */
//#define NPP_MIN_64S     (-9223372036854775807LL - 1) /**<  Minimum 64-bit signed integer */

//#define NPP_MINABS_32F  ( 1.175494351e-38f )         /**<  Smallest positive 32-bit floating point value */
//#define NPP_MAXABS_32F  ( 3.402823466e+38f )         /**<  Largest  positive 32-bit floating point value */
//#define NPP_MINABS_64F  ( 2.2250738585072014e-308 )  /**<  Smallest positive 64-bit floating point value */
//#define NPP_MAXABS_64F  ( 1.7976931348623158e+308 )  /**<  Largest  positive 64-bit floating point value */

//Point is a 2d point
type Point C.NppiPoint

func (n Point) c() C.NppiPoint { return (C.NppiPoint)(n) }
func (n *Point) cptr() *C.NppiPoint {
	return (*C.NppiPoint)(n)
}

//Set sets the Point
func (n *Point) Set(x, y int32) {
	n.x = (C.int)(x)
	n.y = (C.int)(y)
}

//Get gets the Point's x and y
func (n *Point) Get() (x, y int32) {
	return (int32)(n.x), (int32)(n.y)
}

/*
typedef struct
{
    int x;
    int y;
} NppiPoint;
*/

//PolarPoint is a 2D Polar Point
type PolarPoint C.NppPointPolar

func (n PolarPoint) c() C.NppPointPolar      { return (C.NppPointPolar)(n) }
func (n *PolarPoint) cptr() *C.NppPointPolar { return (*C.NppPointPolar)(n) }

//Set sets the polar cordinates
func (n *PolarPoint) Set(rho, theta float32) {
	n.rho = (C.Npp32f)(rho)
	n.theta = (C.Npp32f)(theta)
}

//Get gets the polar coordinates
func (n *PolarPoint) Get() (rho, theta float32) {
	return (float32)(n.rho), (float32)(n.theta)
}

/*

typedef struct {
    Npp32f rho;
    Npp32f theta;
} NppPointPolar;
*/

//Size -2D Size represents the size of a a rectangular region in two space.
type Size C.NppiSize

func (n Size) c() C.NppiSize {
	return (C.NppiSize)(n)
}
func (n *Size) cptr() *C.NppiSize {
	return (*C.NppiSize)(n)
}

/*
typedef struct
{
    int width;
    int height;
} NppiSize;
*/

//Set sets the width and Height
func (n *Size) Set(w, h int32) {
	n.width = C.int(w)
	n.height = C.int(h)

}

//Get returns the width and Height
func (n *Size) Get() (w, h int32) {
	w = int32(n.width)
	h = int32(n.height)
	return w, h
}

//Rect - 2D Rectangle This struct contains position and size information of a rectangle in two space.
// The rectangle's position is usually signified by the coordinate of its upper-left corner.
type Rect C.NppiRect

func (n Rect) c() C.NppiRect {
	return (C.NppiRect)(n)
}
func (n *Rect) cptr() *C.NppiRect {
	return (*C.NppiRect)(n)
}

//Set sets the NppiRect's values
func (n *Rect) Set(x, y, w, h int32) {
	n.x = (C.int)(x)
	n.y = (C.int)(y)
	n.width = (C.int)(w)
	n.height = (C.int)(h)
}

//Get gets the NppiRect's values
func (n *Rect) Get() (x, y, w, h int32) {
	x = (int32)(n.x)
	y = (int32)(n.y)
	w = (int32)(n.width)
	h = (int32)(n.height)
	return x, y, w, h
}

/*
typedef struct
{
    int x;         //  x-coordinate of upper left corner (lowest memory address).
    int y;       // y-coordinate of upper left corner (lowest memory address).
    int width;    // Rectangle width.
    int height;   // Rectangle height.
} NppiRect;
*/

//Axis enums NpiiAxis
type Axis C.NppiAxis

//Horizontal sets and returns  Axis(C.NPP_HORIZONTAL_AXIS)
func (n *Axis) Horizontal() Axis {
	*n = Axis(C.NPP_HORIZONTAL_AXIS)
	return *n
}

//Vertical sets and returns  Axis(C.NPP_VERTICAL_AXIS)
func (n *Axis) Vertical() Axis {
	*n = Axis(C.NPP_VERTICAL_AXIS)
	return *n
}

//Both sets and returns  Axis(C.NPP_BOTH_AXIS)
func (n *Axis) Both() Axis {
	*n = Axis(C.NPP_BOTH_AXIS)
	return *n
}
func (n Axis) c() C.NppiAxis {
	return (C.NppiAxis)(n)
}

//CmpOp is a flag type used for comparisons
type CmpOp C.NppCmpOp

//Less is < sets and returns CmpOp(C.NPP_CMP_LESS)
func (n *CmpOp) Less() CmpOp { *n = CmpOp(C.NPP_CMP_LESS); return *n }

//LessEq is <= sets and returns CmpOp(C.NPP_CMP_LESS_EQ)
func (n *CmpOp) LessEq() CmpOp { *n = CmpOp(C.NPP_CMP_LESS_EQ); return *n }

//Eq is = sets and returns CmpOp(C.NPP_CMP_EQ)
func (n *CmpOp) Eq() CmpOp { *n = CmpOp(C.NPP_CMP_EQ); return *n }

//GreaterEq is >= sets and returns CmpOp(C.NPP_CMP_GREATER_EQ)
func (n *CmpOp) GreaterEq() CmpOp { *n = CmpOp(C.NPP_CMP_GREATER_EQ); return *n }

//Greater is > sets and returns CmpOp(C.NPP_CMP_GREATER)
func (n *CmpOp) Greater() CmpOp { *n = CmpOp(C.NPP_CMP_GREATER); return *n }

/**
NppRoundMode go wrapper for roundimg modes description from original header
 * Rounding Modes
 *
 * The enumerated rounding modes are used by a large number of NPP primitives
 * to allow the user to specify the method by which fractional values are converted
 * to integer values. Also see \ref rounding_modes.
 *
 * For NPP release 5.5 new names for the three rounding modes are introduced that are
 * based on the naming conventions for rounding modes set forth in the IEEE-754
 * floating-point standard. Developers are encouraged to use the new, longer names
 * to be future proof as the legacy names will be deprecated in subsequent NPP releases.
 *
*/

//RoundMode methods return the rounding mode flags
type RoundMode C.NppRoundMode

func (r *RoundMode) c() C.NppRoundMode {
	return C.NppRoundMode(*r)
}

//RndNear will round to the nearest number
//method sets and returns RoundMode(C.NPP_RND_NEAR)
func (r *RoundMode) RndNear() RoundMode { *r = RoundMode(C.NPP_RND_NEAR); return *r }

/*RndFinancial -From Original Header
 * Round according to financial rule.
 * All fractional numbers are rounded to their nearest integer. The ambiguous
 * cases (i.e. \<integer\>.5) are rounded away from zero.
 * E.g.
 * - roundFinancial(0.4)  = 0
 * - roundFinancial(0.5)  = 1
 * - roundFinancial(-1.5) = -2
 */
//method sets and returns RoundMode(C.NPP_RND_FINANCIAL)
func (r *RoundMode) RndFinancial() RoundMode { *r = RoundMode(C.NPP_RND_FINANCIAL); return *r }

/*RndZero - From Original Header
 * Round towards zero (truncation).
 * All fractional numbers of the form \<integer\>.\<decimals\> are truncated to
 * \<integer\>.
 * - roundZero(1.5) = 1
 * - roundZero(1.9) = 1
 * - roundZero(-2.5) = -2
 */
//method sets and returns RoundMode(C.NPP_RND_ZERO)
func (r *RoundMode) RndZero() RoundMode { *r = RoundMode(C.NPP_RND_ZERO); return *r }

/*
 * Other rounding modes supported by IEEE-754 (2008) floating-point standard:
 *
 * - NPP_ROUND_TOWARD_INFINITY // ceiling
 * - NPP_ROUND_TOWARD_NEGATIVE_INFINITY // floor
 *
 */

//BorderType is a flag type used to set the type of boarder.  Flags are passed through methods
type BorderType C.NppiBorderType

func (b BorderType) c() C.NppiBorderType {
	return C.NppiBorderType(b)
}

//Undefined sets and returns BorderType(C.NPP_BORDER_UNDEFINED)
func (b *BorderType) Undefined() BorderType { *b = BorderType(C.NPP_BORDER_UNDEFINED); return *b }

//None sets and returns BorderType(C.NPP_BORDER_NONE)
func (b *BorderType) None() BorderType { *b = BorderType(C.NPP_BORDER_NONE); return *b }

//Constant sets and returns BorderType(C.NPP_BORDER_CONSTANT)
func (b *BorderType) Constant() BorderType { *b = BorderType(C.NPP_BORDER_CONSTANT); return *b }

//Replicate sets and returns  BorderType(C.NPP_BORDER_REPLICATE)
func (b *BorderType) Replicate() BorderType { *b = BorderType(C.NPP_BORDER_REPLICATE); return *b }

//Wrap sets and returns BorderType(C.NPP_BORDER_WRAP)
func (b *BorderType) Wrap() BorderType { *b = BorderType(C.NPP_BORDER_WRAP); return *b }

//Mirror returns BorderType(C.NPP_BORDER_MIRROR)
func (b *BorderType) Mirror() BorderType { *b = BorderType(C.NPP_BORDER_MIRROR); return *b }

//HintAlgorithm are flags
type HintAlgorithm C.NppHintAlgorithm

func (h HintAlgorithm) c() C.NppHintAlgorithm { return C.NppHintAlgorithm(h) }

//None sets and returns HintAlgorithm(C.NPP_ALG_HINT_NONE)
func (h *HintAlgorithm) None() HintAlgorithm { *h = HintAlgorithm(C.NPP_ALG_HINT_NONE); return *h }

//Fast sets and returns HintAlgorithm(C.NPP_ALG_HINT_FAST)
func (h *HintAlgorithm) Fast() HintAlgorithm { *h = HintAlgorithm(C.NPP_ALG_HINT_FAST); return *h }

//Accurate sets and returns HintAlgorithm(C.NPP_ALG_HINT_ACCURATE)
func (h *HintAlgorithm) Accurate() HintAlgorithm {
	*h = HintAlgorithm(C.NPP_ALG_HINT_ACCURATE)
	return *h
}

/*
 * Alpha composition controls.
 */

//AlphaOp contains methods used to pass flags for composition controlls
type AlphaOp C.NppiAlphaOp

func (a AlphaOp) c() C.NppiAlphaOp { return C.NppiAlphaOp(a) }

//AlphaOver sets and returns AlphaOp(C.NPPI_OP_ALPHA_OVER)}
func (a *AlphaOp) AlphaOver() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_OVER); return *a }

//AlphaIn sets and returns AlphaOp(C.NPPI_OP_ALPHA_IN)}
func (a *AlphaOp) AlphaIn() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_IN); return *a }

//AlphaOut sets and returns AlphaOp(C.NPPI_OP_ALPHA_OUT)}
func (a *AlphaOp) AlphaOut() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_OUT); return *a }

//AlphaAtop sets and returns AlphaOp(C.NPPI_OP_ALPHA_ATOP)}
func (a *AlphaOp) AlphaAtop() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_ATOP); return *a }

//AlphaXOR sets and returns AlphaOp(C.NPPI_OP_ALPHA_XOR)}
func (a *AlphaOp) AlphaXOR() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_XOR); return *a }

//AlphaPlus sets and returns AlphaOp(C.NPPI_OP_ALPHA_PLUS)}
func (a *AlphaOp) AlphaPlus() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_PLUS); return *a }

//AlphaOverPremul sets and returns AlphaOp(C.NPPI_OP_ALPHA_OVER_PREMUL)}
func (a *AlphaOp) AlphaOverPremul() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_OVER_PREMUL); return *a }

//AlphaInPremul sets and returns AlphaOp(C.NPPI_OP_ALPHA_IN_PREMUL)}
func (a *AlphaOp) AlphaInPremul() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_IN_PREMUL); return *a }

//AlphaOutPremul sets and returns AlphaOp(C.NPPI_OP_ALPHA_OUT_PREMUL)}
func (a *AlphaOp) AlphaOutPremul() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_OUT_PREMUL); return *a }

//AlphaAtopPremul sets and returns AlphaOp(C.NPPI_OP_ALPHA_ATOP_PREMUL)}
func (a *AlphaOp) AlphaAtopPremul() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_ATOP_PREMUL); return *a }

//AlphaXORPremul sets and returns AlphaOp(C.NPPI_OP_ALPHA_XOR_PREMUL)}
func (a *AlphaOp) AlphaXORPremul() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_XOR_PREMUL); return *a }

//AlphaPlusPremul sets and returns AlphaOp(C.NPPI_OP_ALPHA_PLUS_PREMUL)}
func (a *AlphaOp) AlphaPlusPremul() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_PLUS_PREMUL); return *a }

//AlphaPremul sets and returns AlphaOp(C.NPPI_OP_ALPHA_PREMUL)}
func (a *AlphaOp) AlphaPremul() AlphaOp { *a = AlphaOp(C.NPPI_OP_ALPHA_PREMUL); return *a }

//HOGConfig type defines the configuration parameters for the HOG descriptor
type HOGConfig C.NppiHOGConfig

func (h HOGConfig) c() C.NppiHOGConfig {
	return C.NppiHOGConfig(h)
}

func (h *HOGConfig) cptr() *C.NppiHOGConfig {
	return (*C.NppiHOGConfig)(h)
}

//Set sets the HOGConfig inner types
func (h *HOGConfig) Set(cellSize, histogramBlockSize, nHistogramBins int32, detectionWindowSize Size) {
	h.cellSize = (C.int)(cellSize)
	h.histogramBlockSize = (C.int)(histogramBlockSize)
	h.histogramBlockSize = (C.int)(nHistogramBins)
	h.detectionWindowSize = detectionWindowSize.c()
}

//Get gets the inner type values
func (h *HOGConfig) Get() (cellSize, histogramBlockSize, nHistogramBins int32, detectionWindowSize Size) {
	cellSize = (int32)(h.cellSize)
	histogramBlockSize = (int32)(h.histogramBlockSize)
	nHistogramBins = (int32)(h.nHistogramBins)
	detectionWindowSize = (Size)(h.detectionWindowSize)
	return cellSize, histogramBlockSize, nHistogramBins, detectionWindowSize
}

/*
typedef struct
{
    int      cellSize;              //square cell size (pixels).
    int      histogramBlockSize;    //square histogram block size (pixels).
    int      nHistogramBins;        //required number of histogram bins.
    NppiSize detectionWindowSize;   //detection window size (pixels).
} NppiHOGConfig;
*/

//#define NPP_HOG_MAX_CELL_SIZE                          (16) /**< max horizontal/vertical pixel size of cell.   */
//#define NPP_HOG_MAX_BLOCK_SIZE                         (64) /**< max horizontal/vertical pixel size of block.  */
//#define NPP_HOG_MAX_BINS_PER_CELL                      (16) /**< max number of histogram bins. */
//#define NPP_HOG_MAX_CELLS_PER_DESCRIPTOR              (256) /**< max number of cells in a descriptor window.   */
//#define NPP_HOG_MAX_OVERLAPPING_BLOCKS_PER_DESCRIPTOR (256) /**< max number of overlapping blocks in a descriptor window.   */
//#define NPP_HOG_MAX_DESCRIPTOR_LOCATIONS_PER_CALL     (128) /**< max number of descriptor window locations per function call.   */

//HaarClassifier32f is a structure used in Haar Classification
type HaarClassifier32f C.NppiHaarClassifier_32f

func (h HaarClassifier32f) c() C.NppiHaarClassifier_32f {
	return C.NppiHaarClassifier_32f(h)
}

func (h *HaarClassifier32f) cptr() *C.NppiHaarClassifier_32f {
	return (*C.NppiHaarClassifier_32f)(h)
}

//Set sets the HOGConfig inner types
func (h *HaarClassifier32f) Set(classifiers []*Int32, classifierStep int, classifierSize Size, counterDevice *Int32) {
	h.numClassifiers = (C.int)(len(classifiers))
	h.classifiers = classifiers[0].cptr()
	h.classifierStep = (C.size_t)(classifierStep)
	h.classifierSize = classifierSize.c()
	h.counterDevice = counterDevice.cptr()
}

/*
//Get gets the inner type values
func (h *HaarClassifier32f) Get()(classifiers []*Int32, classifierStep int, classifierSize Size, counterDevice *Int32) {

numofclassifiers :=	(C.int)h.numClassifiers
classifiers=make([]Int32,	numofclassifiers)
h.classifiers
	classifiers[0].cptr() = h.classifiers =
	h.classifierStep = (C.size_t)(classifierStep)
	h.classifierSize = classifierSize.c()
	h.counterDevice = counterDevice.cptr()
*/
/*
typedef struct
{
    int      numClassifiers;     // number of classifiers
    Npp32s * classifiers;        // packed classifier data 40 bytes each
    size_t   classifierStep;
    NppiSize classifierSize;
    Npp32s * counterDevice;
} NppiHaarClassifier_32f;
*/

//HaarBuffer is a buffer for algorithms that require a HaarBuffer
type HaarBuffer C.NppiHaarBuffer

func (h HaarBuffer) c() C.NppiHaarBuffer {
	return C.NppiHaarBuffer(h)
}

func (h *HaarBuffer) cptr() *C.NppiHaarBuffer {
	return (*C.NppiHaarBuffer)(h)
}

//Get gets the HaarBuffer inner values
func (h HaarBuffer) Get() (BufferSize int32, Buffer *Int32) {
	BufferSize = (int32)(h.haarBufferSize)
	Buffer.wrap(h.haarBuffer)
	return BufferSize, Buffer
}

//Set sets the HaarBuffer inner values
func (h *HaarBuffer) Set(buffsize int32, buffer *Int32) {
	h.haarBufferSize = (C.int)(buffsize)
	h.haarBuffer = buffer.cptr()
}

/*
typedef struct
{
    int      haarBufferSize;     //size of the buffer
    Npp32s * haarBuffer;        //buffer

} NppiHaarBuffer;
*/

//ZCType is a type that holds flags through methods
type ZCType C.NppsZCType

func (z ZCType) c() C.NppsZCType { return C.NppsZCType(z) }

//ZCR sets and returns sign change  -- returns and sets ZCType(C.nppZCR)
func (z *ZCType) ZCR() ZCType { *z = ZCType(C.nppZCR); return *z }

//ZCXor sets and returns sign change XOR  -- returns and sets ZCType(C.nppZCXor)
func (z *ZCType) ZCXor() ZCType { *z = ZCType(C.nppZCXor); return *z }

//ZCC sets and returns sign change count_0  -- returns and sets ZCType(C.nppZCC)
func (z *ZCType) ZCC() ZCType { *z = ZCType(C.nppZCC); return *z }

//HuffmanTableType is a type used for HuffmanTableType flags flags are passed by methods
type HuffmanTableType C.NppiHuffmanTableType

func (h HuffmanTableType) c() C.NppiHuffmanTableType { return C.NppiHuffmanTableType(h) }

//DCTable - DC Table flag -- returns and sets HuffmanTableType(C.nppiDCTable)
func (h *HuffmanTableType) DCTable() HuffmanTableType { *h = HuffmanTableType(C.nppiDCTable); return *h }

//ACTable - AC Table flag  -- returns and sets HuffmanTableType(C.nppiACTable)
func (h *HuffmanTableType) ACTable() HuffmanTableType { *h = HuffmanTableType(C.nppiACTable); return *h }

//Norm is used for norm flags where needed Norm will return flags through methods
type Norm C.NppiNorm

func (n Norm) c() C.NppiNorm { return C.NppiNorm(n) }

//Inf maximum
func (n *Norm) Inf() Norm { *n = Norm(C.nppiNormInf); return *n }

//L1 sum
func (n *Norm) L1() Norm { *n = Norm(C.nppiNormL1); return *n }

//L2 square root of sum of squares
func (n *Norm) L2() Norm { *n = Norm(C.nppiNormL2); return *n }
