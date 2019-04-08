package npp

/*
#include<npps_arithmetic_and_logical_operations.h>
*/
import "C"

func AddC8uISfs(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_8u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_8u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func AddC8uSfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_8u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_8u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

/*
func AddC16uISfsCtx(Npp16u nValue, Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddC_16u_ISfs_Ctx(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddC16uISfs(Npp16u nValue, Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddC_16u_ISfs(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddC16uSfsCtx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddC_16u_Sfs_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddC16uSfs( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddC_16u_Sfs( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddC16sISfsCtx(Npp16s nValue, Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddC_16s_ISfs_Ctx(Npp16s nValue, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddC16sISfs(Npp16s nValue, Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddC_16s_ISfs(Npp16s nValue, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddC16sSfsCtx( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddC_16s_Sfs_Ctx( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddC16sSfs( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddC_16s_Sfs( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddC16scISfsCtx(Npp16sc nValue, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddC16scISfs(Npp16sc nValue, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddC_16sc_ISfs(Npp16sc nValue, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddC16scSfsCtx( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddC_16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddC16scSfs( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddC_16sc_Sfs( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddC32sISfsCtx(Npp32s nValue, Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddC_32s_ISfs_Ctx(Npp32s nValue, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddC32sISfs(Npp32s nValue, Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddC_32s_ISfs(Npp32s nValue, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddC32sSfsCtx( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddC_32s_Sfs_Ctx( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddC32sSfs( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddC_32s_Sfs( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddC32scISfsCtx(Npp32sc nValue, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddC_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddC32scISfs(Npp32sc nValue, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddC_32sc_ISfs(Npp32sc nValue, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddC32scSfsCtx( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddC_32sc_Sfs_Ctx( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddC32scSfs( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddC_32sc_Sfs( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddC32fICtx(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddC_32f_I_Ctx(nValue Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func AddC32fI(nValue Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsAddC_32f_I(nValue Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func AddC32fCtx( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddC_32f_Ctx( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func AddC32f( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsAddC_32f( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength))).ToError()
}
func AddC32fcICtx(nValue *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddC_32fc_I_Ctx(nValue *Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func AddC32fcI(nValue *Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsAddC_32fc_I(nValue *Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func AddC32fcCtx( pSrc *Float32Complex, nValue *Float32, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddC_32fc_Ctx( pSrc *Float32Complex, nValue *Float32, pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func AddC32fc( pSrc *Float32Complex, nValue *Float32, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsAddC_32fc( pSrc *Float32Complex, nValue *Float32, pDst.cptr(), (C.int)(nLength))).ToError()
}
func AddC64fICtx(Npp64f nValue, Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddC_64f_I_Ctx(Npp64f nValue, Npp64f * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func AddC64fI(Npp64f nValue, Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsAddC_64f_I(Npp64f nValue, Npp64f * pSrcDst, (C.int)(nLength))).ToError()
}
func AddC64fCtx( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddC_64f_Ctx( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func AddC64f( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, nLength int32) error{
  return status(C.nppsAddC_64f( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func AddC64fcICtx(Npp64fc nValue, Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddC_64fc_I_Ctx(Npp64fc nValue, Npp64fc * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func AddC64fcI(Npp64fc nValue, Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsAddC_64fc_I(Npp64fc nValue, Npp64fc * pSrcDst, (C.int)(nLength))).ToError()
}
func AddC64fcCtx( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddC_64fc_Ctx( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func AddC64fc( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsAddC_64fc( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, (C.int)(nLength))).ToError()
}
func AddProductC32fCtx( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddProductC_32f_Ctx( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func AddProductC32f( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsAddProductC_32f( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength))).ToError()
}
func MulC8uISfsCtx(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_8u_ISfs_Ctx(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC8uISfs(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_8u_ISfs(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC8uSfsCtx(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_8u_Sfs_Ctx(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC8uSfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_8u_Sfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC16uISfsCtx(Npp16u nValue, Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_16u_ISfs_Ctx(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC16uISfs(Npp16u nValue, Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_16u_ISfs(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC16uSfsCtx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_16u_Sfs_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC16uSfs( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_16u_Sfs( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC16sISfsCtx(Npp16s nValue, Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_16s_ISfs_Ctx(Npp16s nValue, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC16sISfs(Npp16s nValue, Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_16s_ISfs(Npp16s nValue, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC16sSfsCtx( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_16s_Sfs_Ctx( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC16sSfs( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_16s_Sfs( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC16scISfsCtx(Npp16sc nValue, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC16scISfs(Npp16sc nValue, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_16sc_ISfs(Npp16sc nValue, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC16scSfsCtx( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC16scSfs( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_16sc_Sfs( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC32sISfsCtx(Npp32s nValue, Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_32s_ISfs_Ctx(Npp32s nValue, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC32sISfs(Npp32s nValue, Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_32s_ISfs(Npp32s nValue, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC32sSfsCtx( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_32s_Sfs_Ctx( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC32sSfs( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_32s_Sfs( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC32scISfsCtx(Npp32sc nValue, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC32scISfs(Npp32sc nValue, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_32sc_ISfs(Npp32sc nValue, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC32scSfsCtx( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_32sc_Sfs_Ctx( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC32scSfs( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_32sc_Sfs( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC32fICtx(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMulC_32f_I_Ctx(nValue Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func MulC32fI(nValue Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsMulC_32f_I(nValue Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func MulC32fCtx( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMulC_32f_Ctx( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func MulC32f( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsMulC_32f( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength))).ToError()
}
func MulCLow32f16sCtx( pSrc *Float32, nValue Float32, Npp16s * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMulC_Low_32f16s_Ctx( pSrc *Float32, nValue Float32, Npp16s * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func MulCLow32f16s( pSrc *Float32, nValue Float32, Npp16s * pDst, nLength int32) error{
  return status(C.nppsMulC_Low_32f16s( pSrc *Float32, nValue Float32, Npp16s * pDst, (C.int)(nLength))).ToError()
}
func MulC32f16sSfsCtx( pSrc *Float32, nValue Float32, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_32f16s_Sfs_Ctx( pSrc *Float32, nValue Float32, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC32f16sSfs( pSrc *Float32, nValue Float32, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_32f16s_Sfs( pSrc *Float32, nValue Float32, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC32fcICtx(nValue *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMulC_32fc_I_Ctx(nValue *Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func MulC32fcI(nValue *Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsMulC_32fc_I(nValue *Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func MulC32fcCtx( pSrc *Float32Complex, nValue *Float32, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMulC_32fc_Ctx( pSrc *Float32Complex, nValue *Float32, pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func MulC32fc( pSrc *Float32Complex, nValue *Float32, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsMulC_32fc( pSrc *Float32Complex, nValue *Float32, pDst.cptr(), (C.int)(nLength))).ToError()
}
func MulC64fICtx(Npp64f nValue, Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMulC_64f_I_Ctx(Npp64f nValue, Npp64f * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func MulC64fI(Npp64f nValue, Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsMulC_64f_I(Npp64f nValue, Npp64f * pSrcDst, (C.int)(nLength))).ToError()
}
func MulC64fCtx( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMulC_64f_Ctx( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func MulC64f( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, nLength int32) error{
  return status(C.nppsMulC_64f( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func MulC64f64sISfsCtx(Npp64f nValue, Npp64s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMulC_64f64s_ISfs_Ctx(Npp64f nValue, Npp64s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulC64f64sISfs(Npp64f nValue, Npp64s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMulC_64f64s_ISfs(Npp64f nValue, Npp64s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulC64fcICtx(Npp64fc nValue, Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMulC_64fc_I_Ctx(Npp64fc nValue, Npp64fc * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func MulC64fcI(Npp64fc nValue, Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsMulC_64fc_I(Npp64fc nValue, Npp64fc * pSrcDst, (C.int)(nLength))).ToError()
}
func MulC64fcCtx( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMulC_64fc_Ctx( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func MulC64fc( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsMulC_64fc( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, (C.int)(nLength))).ToError()
}
func SubC8uISfsCtx(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_8u_ISfs_Ctx(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC8uISfs(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_8u_ISfs(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC8uSfsCtx(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_8u_Sfs_Ctx(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC8uSfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_8u_Sfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC16uISfsCtx(Npp16u nValue, Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_16u_ISfs_Ctx(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC16uISfs(Npp16u nValue, Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_16u_ISfs(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC16uSfsCtx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_16u_Sfs_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC16uSfs( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_16u_Sfs( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC16sISfsCtx(Npp16s nValue, Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_16s_ISfs_Ctx(Npp16s nValue, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC16sISfs(Npp16s nValue, Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_16s_ISfs(Npp16s nValue, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC16sSfsCtx( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_16s_Sfs_Ctx( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC16sSfs( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_16s_Sfs( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC16scISfsCtx(Npp16sc nValue, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC16scISfs(Npp16sc nValue, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_16sc_ISfs(Npp16sc nValue, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC16scSfsCtx( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC16scSfs( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_16sc_Sfs( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC32sISfsCtx(Npp32s nValue, Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_32s_ISfs_Ctx(Npp32s nValue, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC32sISfs(Npp32s nValue, Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_32s_ISfs(Npp32s nValue, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC32sSfsCtx( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_32s_Sfs_Ctx( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC32sSfs( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_32s_Sfs( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC32scISfsCtx(Npp32sc nValue, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC32scISfs(Npp32sc nValue, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_32sc_ISfs(Npp32sc nValue, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC32scSfsCtx( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubC_32sc_Sfs_Ctx( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubC32scSfs( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubC_32sc_Sfs( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubC32fICtx(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubC_32f_I_Ctx(nValue Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func SubC32fI(nValue Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsSubC_32f_I(nValue Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func SubC32fCtx( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubC_32f_Ctx( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func SubC32f( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsSubC_32f( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength))).ToError()
}
func SubC32fcICtx(nValue *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubC_32fc_I_Ctx(nValue *Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func SubC32fcI(nValue *Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsSubC_32fc_I(nValue *Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func SubC32fcCtx( pSrc *Float32Complex, nValue *Float32, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubC_32fc_Ctx( pSrc *Float32Complex, nValue *Float32, pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func SubC32fc( pSrc *Float32Complex, nValue *Float32, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsSubC_32fc( pSrc *Float32Complex, nValue *Float32, pDst.cptr(), (C.int)(nLength))).ToError()
}
func SubC64fICtx(Npp64f nValue, Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubC_64f_I_Ctx(Npp64f nValue, Npp64f * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func SubC64fI(Npp64f nValue, Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsSubC_64f_I(Npp64f nValue, Npp64f * pSrcDst, (C.int)(nLength))).ToError()
}
func SubC64fCtx( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubC_64f_Ctx( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func SubC64f( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, nLength int32) error{
  return status(C.nppsSubC_64f( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func SubC64fcICtx(Npp64fc nValue, Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubC_64fc_I_Ctx(Npp64fc nValue, Npp64fc * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func SubC64fcI(Npp64fc nValue, Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsSubC_64fc_I(Npp64fc nValue, Npp64fc * pSrcDst, (C.int)(nLength))).ToError()
}
func SubC64fcCtx( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubC_64fc_Ctx( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func SubC64fc( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsSubC_64fc( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, (C.int)(nLength))).ToError()
}
func SubCRev8uISfsCtx(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_8u_ISfs_Ctx(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev8uISfs(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_8u_ISfs(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev8uSfsCtx(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_8u_Sfs_Ctx(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev8uSfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_8u_Sfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev16uISfsCtx(Npp16u nValue, Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_16u_ISfs_Ctx(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev16uISfs(Npp16u nValue, Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_16u_ISfs(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev16uSfsCtx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_16u_Sfs_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev16uSfs( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_16u_Sfs( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev16sISfsCtx(Npp16s nValue, Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_16s_ISfs_Ctx(Npp16s nValue, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev16sISfs(Npp16s nValue, Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_16s_ISfs(Npp16s nValue, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev16sSfsCtx( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_16s_Sfs_Ctx( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev16sSfs( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_16s_Sfs( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev16scISfsCtx(Npp16sc nValue, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev16scISfs(Npp16sc nValue, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_16sc_ISfs(Npp16sc nValue, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev16scSfsCtx( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev16scSfs( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_16sc_Sfs( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev32sISfsCtx(Npp32s nValue, Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_32s_ISfs_Ctx(Npp32s nValue, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev32sISfs(Npp32s nValue, Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_32s_ISfs(Npp32s nValue, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev32sSfsCtx( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_32s_Sfs_Ctx( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev32sSfs( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_32s_Sfs( Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev32scISfsCtx(Npp32sc nValue, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev32scISfs(Npp32sc nValue, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_32sc_ISfs(Npp32sc nValue, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev32scSfsCtx( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_32sc_Sfs_Ctx( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func SubCRev32scSfs( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSubCRev_32sc_Sfs( Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func SubCRev32fICtx(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_32f_I_Ctx(nValue Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func SubCRev32fI(nValue Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsSubCRev_32f_I(nValue Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func SubCRev32fCtx( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_32f_Ctx( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func SubCRev32f( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsSubCRev_32f( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength))).ToError()
}
func SubCRev32fcICtx(nValue *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_32fc_I_Ctx(nValue *Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func SubCRev32fcI(nValue *Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsSubCRev_32fc_I(nValue *Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func SubCRev32fcCtx( pSrc *Float32Complex, nValue *Float32, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_32fc_Ctx( pSrc *Float32Complex, nValue *Float32, pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func SubCRev32fc( pSrc *Float32Complex, nValue *Float32, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsSubCRev_32fc( pSrc *Float32Complex, nValue *Float32, pDst.cptr(), (C.int)(nLength))).ToError()
}
func SubCRev64fICtx(Npp64f nValue, Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_64f_I_Ctx(Npp64f nValue, Npp64f * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func SubCRev64fI(Npp64f nValue, Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsSubCRev_64f_I(Npp64f nValue, Npp64f * pSrcDst, (C.int)(nLength))).ToError()
}
func SubCRev64fCtx( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_64f_Ctx( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func SubCRev64f( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, nLength int32) error{
  return status(C.nppsSubCRev_64f( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func SubCRev64fcICtx(Npp64fc nValue, Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_64fc_I_Ctx(Npp64fc nValue, Npp64fc * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func SubCRev64fcI(Npp64fc nValue, Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsSubCRev_64fc_I(Npp64fc nValue, Npp64fc * pSrcDst, (C.int)(nLength))).ToError()
}
func SubCRev64fcCtx( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSubCRev_64fc_Ctx( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func SubCRev64fc( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsSubCRev_64fc( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, (C.int)(nLength))).ToError()
}
func DivC8uISfsCtx(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDivC_8u_ISfs_Ctx(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func DivC8uISfs(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDivC_8u_ISfs(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func DivC8uSfsCtx(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDivC_8u_Sfs_Ctx(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func DivC8uSfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDivC_8u_Sfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func DivC16uISfsCtx(Npp16u nValue, Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDivC_16u_ISfs_Ctx(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func DivC16uISfs(Npp16u nValue, Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDivC_16u_ISfs(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func DivC16uSfsCtx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDivC_16u_Sfs_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func DivC16uSfs( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDivC_16u_Sfs( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func DivC16sISfsCtx(Npp16s nValue, Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDivC_16s_ISfs_Ctx(Npp16s nValue, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func DivC16sISfs(Npp16s nValue, Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDivC_16s_ISfs(Npp16s nValue, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func DivC16sSfsCtx( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDivC_16s_Sfs_Ctx( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func DivC16sSfs( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDivC_16s_Sfs( Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func DivC16scISfsCtx(Npp16sc nValue, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDivC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func DivC16scISfs(Npp16sc nValue, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDivC_16sc_ISfs(Npp16sc nValue, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func DivC16scSfsCtx( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDivC_16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func DivC16scSfs( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDivC_16sc_Sfs( Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
*/
func DivC32fI(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_32f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_32f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func DivC32f(pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_32f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_32f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func DivC32fcI(nValue Float32Complex, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_32fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_32fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

func DivC32fc(pSrc *Float32Complex, nValue Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_32fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_32fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

/*
func DivC64fICtx(Npp64f nValue, Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDivC_64f_I_Ctx(Npp64f nValue, Npp64f * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func DivC64fI(Npp64f nValue, Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsDivC_64f_I(Npp64f nValue, Npp64f * pSrcDst, (C.int)(nLength))).ToError()
}
func DivC64fCtx( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDivC_64f_Ctx( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func DivC64f( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, nLength int32) error{
  return status(C.nppsDivC_64f( Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func DivC64fcICtx(Npp64fc nValue, Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDivC_64fc_I_Ctx(Npp64fc nValue, Npp64fc * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func DivC64fcI(Npp64fc nValue, Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsDivC_64fc_I(Npp64fc nValue, Npp64fc * pSrcDst, (C.int)(nLength))).ToError()
}
func DivC64fcCtx( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDivC_64fc_Ctx( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func DivC64fc( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsDivC_64fc( Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, (C.int)(nLength))).ToError()
}
func DivCRev16uICtx(Npp16u nValue, Npp16u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDivCRev_16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func DivCRev16uI(Npp16u nValue, Npp16u * pSrcDst, nLength int32) error{
  return status(C.nppsDivCRev_16u_I(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength))).ToError()
}
func DivCRev16uCtx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDivCRev_16u_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func DivCRev16u( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32) error{
  return status(C.nppsDivCRev_16u( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func DivCRev32fICtx(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDivCRev_32f_I_Ctx(nValue Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func DivCRev32fI(nValue Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsDivCRev_32f_I(nValue Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func DivCRev32fCtx( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDivCRev_32f_Ctx( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func DivCRev32f( pSrc *Float32, nValue Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsDivCRev_32f( pSrc *Float32, nValue Float32, pDst *Float32, (C.int)(nLength))).ToError()
}
func Add16sCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16s_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Add16s( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32) error{
  return status(C.nppsAdd_16s( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength))).ToError()
}
func Add16uCtx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16u_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Add16u( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32) error{
  return status(C.nppsAdd_16u( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func Add32uCtx( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_32u_Ctx( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Add32u( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, nLength int32) error{
  return status(C.nppsAdd_32u( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, (C.int)(nLength))).ToError()
}
func Add32fCtx( pSrc *Float321,pSrc *Float322, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_32f_Ctx( pSrc *Float321,pSrc *Float322, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Add32f( pSrc *Float321,pSrc *Float322, pDst *Float32, nLength int32) error{
  return status(C.nppsAdd_32f( pSrc *Float321,pSrc *Float322, pDst *Float32, (C.int)(nLength))).ToError()
}
func Add64fCtx( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_64f_Ctx( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Add64f( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, nLength int32) error{
  return status(C.nppsAdd_64f( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func Add32fcCtx( pSrc1 *Float32Complex,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_32fc_Ctx( pSrc1.cptr(),pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func Add32fc( pSrc1 *Float32Complex,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsAdd_32fc( pSrc1.cptr(),pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
}
func Add64fcCtx( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_64fc_Ctx( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Add64fc( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsAdd_64fc( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, (C.int)(nLength))).ToError()
}
func Add8u16uCtx(pSrc *Uint81,Npp8u * pSrc2, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_8u16u_Ctx(pSrc *Uint81,Npp8u * pSrc2, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Add8u16u(pSrc *Uint81,Npp8u * pSrc2, Npp16u * pDst, nLength int32) error{
  return status(C.nppsAdd_8u16u(pSrc *Uint81,Npp8u * pSrc2, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func Add16s32fCtx( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16s32f_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Add16s32f( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, nLength int32) error{
  return status(C.nppsAdd_16s32f( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, (C.int)(nLength))).ToError()
}
func Add8uSfsCtx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_8u_Sfs_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Add8uSfs(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_8u_Sfs(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Add16uSfsCtx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16u_Sfs_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Add16uSfs( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_16u_Sfs( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Add16sSfsCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16s_Sfs_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Add16sSfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_16s_Sfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Add32sSfsCtx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_32s_Sfs_Ctx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Add32sSfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_32s_Sfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Add64sSfsCtx( Npp64s * pSrc1,Npp64s * pSrc2, Npp64s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_64s_Sfs_Ctx( Npp64s * pSrc1,Npp64s * pSrc2, Npp64s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Add64sSfs( Npp64s * pSrc1,Npp64s * pSrc2, Npp64s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_64s_Sfs( Npp64s * pSrc1,Npp64s * pSrc2, Npp64s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Add16scSfsCtx( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16sc_Sfs_Ctx( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Add16scSfs( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_16sc_Sfs( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Add32scSfsCtx( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_32sc_Sfs_Ctx( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Add32scSfs( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_32sc_Sfs( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Add16sICtx( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16s_I_Ctx( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Add16sI( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32) error{
  return status(C.nppsAdd_16s_I( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength))).ToError()
}
func Add32fICtx( pSrc *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_32f_I_Ctx( pSrc *Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Add32fI( pSrc *Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsAdd_32f_I( pSrc *Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func Add64fICtx( Npp64f * pSrc, Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_64f_I_Ctx( Npp64f * pSrc, Npp64f * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Add64fI( Npp64f * pSrc, Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsAdd_64f_I( Npp64f * pSrc, Npp64f * pSrcDst, C)).ToError()
}
func Add32fcICtx( pSrc *Float32Complex, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_32fc_I_Ctx( pSrc *Float32Complex, pSrcDst *Float32, C, ctx.c())).ToError()
}
func Add32fcI( pSrc *Float32Complex, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsAdd_32fc_I( pSrc *Float32Complex, pSrcDst *Float32, C)).ToError()
}
func Add64fcICtx( Npp64fc * pSrc, Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_64fc_I_Ctx( Npp64fc * pSrc, Npp64fc * pSrcDst, C, ctx.c())).ToError()
}
func Add64fcI( Npp64fc * pSrc, Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsAdd_64fc_I( Npp64fc * pSrc, Npp64fc * pSrcDst, C)).ToError()
}
func Add16s32sICtx( Npp16s * pSrc, Npp32s * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16s32s_I_Ctx( Npp16s * pSrc, Npp32s * pSrcDst, C, ctx.c())).ToError()
}
func Add16s32sI( Npp16s * pSrc, Npp32s * pSrcDst, nLength int32) error{
  return status(C.nppsAdd_16s32s_I( Npp16s * pSrc, Npp32s * pSrcDst, C)).ToError()
}
func Add8uISfsCtx(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_8u_ISfs_Ctx(pSrc *Uint8, pSrcDst *Uint8, C, nScaleFactor int32, ctx.c())).ToError()
}
func Add8uISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_8u_ISfs(pSrc *Uint8, pSrcDst *Uint8, C, nScaleFactor int32)).ToError()
}
func Add16uISfsCtx( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16u_ISfs_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, C, nScaleFactor int32, ctx.c())).ToError()
}
func Add16uISfs( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_16u_ISfs( Npp16u * pSrc, Npp16u * pSrcDst, C, nScaleFactor int32)).ToError()
}
func Add16sISfsCtx( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16s_ISfs_Ctx( Npp16s * pSrc, Npp16s * pSrcDst, C, nScaleFactor int32, ctx.c())).ToError()
}
func Add16sISfs( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_16s_ISfs( Npp16s * pSrc, Npp16s * pSrcDst, C, nScaleFactor int32)).ToError()
}
func Add32sISfsCtx( Npp32s * pSrc, Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_32s_ISfs_Ctx( Npp32s * pSrc, Npp32s * pSrcDst, C, nScaleFactor int32, ctx.c())).ToError()
}
func Add32sISfs( Npp32s * pSrc, Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_32s_ISfs( Npp32s * pSrc, Npp32s * pSrcDst, C, nScaleFactor int32)).ToError()
}
func Add16scISfsCtx( Npp16sc * pSrc, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_16sc_ISfs_Ctx( Npp16sc * pSrc, Npp16sc * pSrcDst, C, nScaleFactor int32, ctx.c())).ToError()
}
func Add16scISfs( Npp16sc * pSrc, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_16sc_ISfs( Npp16sc * pSrc, Npp16sc * pSrcDst, C, nScaleFactor int32)).ToError()
}
func Add32scISfsCtx( Npp32sc * pSrc, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAdd_32sc_ISfs_Ctx( Npp32sc * pSrc, Npp32sc * pSrcDst, C, nScaleFactor int32, ctx.c())).ToError()
}
func Add32scISfs( Npp32sc * pSrc, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAdd_32sc_ISfs( Npp32sc * pSrc, Npp32sc * pSrcDst, C, nScaleFactor int32)).ToError()
}
func AddProduct32fCtx( pSrc *Float321,pSrc *Float322, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddProduct_32f_Ctx( pSrc *Float321,pSrc *Float322, pDst *Float32, C, ctx.c())).ToError()
}
func AddProduct32f( pSrc *Float321,pSrc *Float322, pDst *Float32, nLength int32) error{
  return status(C.nppsAddProduct_32f( pSrc *Float321,pSrc *Float322, pDst *Float32, (C.int)(nLength))).ToError()
}
func AddProduct64fCtx( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddProduct_64f_Ctx( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func AddProduct64f( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, nLength int32) error{
  return status(C.nppsAddProduct_64f( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func AddProduct32fcCtx( pSrc1 *Float32Complex,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddProduct_32fc_Ctx( pSrc1.cptr(),pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func AddProduct32fc( pSrc1 *Float32Complex,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsAddProduct_32fc( pSrc1.cptr(),pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
}
func AddProduct64fcCtx( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAddProduct_64fc_Ctx( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func AddProduct64fc( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsAddProduct_64fc( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, (C.int)(nLength))).ToError()
}
func AddProduct16sSfsCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddProduct_16s_Sfs_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddProduct16sSfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddProduct_16s_Sfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddProduct32sSfsCtx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddProduct_32s_Sfs_Ctx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddProduct32sSfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddProduct_32s_Sfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func AddProduct16s32sSfsCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsAddProduct_16s32s_Sfs_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func AddProduct16s32sSfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsAddProduct_16s32s_Sfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul16sCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_16s_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Mul16s( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32) error{
  return status(C.nppsMul_16s( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength))).ToError()
}
func Mul32fCtx( pSrc *Float321,pSrc *Float322, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_32f_Ctx( pSrc *Float321,pSrc *Float322, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Mul32f( pSrc *Float321,pSrc *Float322, pDst *Float32, nLength int32) error{
  return status(C.nppsMul_32f( pSrc *Float321,pSrc *Float322, pDst *Float32, (C.int)(nLength))).ToError()
}
func Mul64fCtx( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_64f_Ctx( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Mul64f( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, nLength int32) error{
  return status(C.nppsMul_64f( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func Mul32fcCtx( pSrc1 *Float32Complex,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_32fc_Ctx( pSrc1.cptr(),pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func Mul32fc( pSrc1 *Float32Complex,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsMul_32fc( pSrc1.cptr(),pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
}
func Mul64fcCtx( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_64fc_Ctx( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Mul64fc( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsMul_64fc( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, (C.int)(nLength))).ToError()
}
func Mul8u16uCtx(pSrc *Uint81,Npp8u * pSrc2, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_8u16u_Ctx(pSrc *Uint81,Npp8u * pSrc2, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Mul8u16u(pSrc *Uint81,Npp8u * pSrc2, Npp16u * pDst, nLength int32) error{
  return status(C.nppsMul_8u16u(pSrc *Uint81,Npp8u * pSrc2, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func Mul16s32fCtx( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_16s32f_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Mul16s32f( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, nLength int32) error{
  return status(C.nppsMul_16s32f( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, (C.int)(nLength))).ToError()
}
func Mul32f32fcCtx( pSrc *Float321,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_32f32fc_Ctx( pSrc *Float321,pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func Mul32f32fc( pSrc *Float321,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsMul_32f32fc( pSrc *Float321,pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
}
func Mul8uSfsCtx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_8u_Sfs_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul8uSfs(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_8u_Sfs(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul16uSfsCtx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_16u_Sfs_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul16uSfs( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_16u_Sfs( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul16sSfsCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_16s_Sfs_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul16sSfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_16s_Sfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul32sSfsCtx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_32s_Sfs_Ctx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul32sSfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_32s_Sfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul16scSfsCtx( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_16sc_Sfs_Ctx( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul16scSfs( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_16sc_Sfs( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul32scSfsCtx( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_32sc_Sfs_Ctx( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul32scSfs( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_32sc_Sfs( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul16u16sSfsCtx( Npp16u * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_16u16s_Sfs_Ctx( Npp16u * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul16u16sSfs( Npp16u * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_16u16s_Sfs( Npp16u * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul16s32sSfsCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_16s32s_Sfs_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul16s32sSfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_16s32s_Sfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul32s32scSfsCtx( Npp32s * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_32s32sc_Sfs_Ctx( Npp32s * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul32s32scSfs( Npp32s * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_32s32sc_Sfs( Npp32s * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func MulLow32sSfsCtx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_Low_32s_Sfs_Ctx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func MulLow32sSfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_Low_32s_Sfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul16sICtx( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_16s_I_Ctx( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Mul16sI( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32) error{
  return status(C.nppsMul_16s_I( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength))).ToError()
}
func Mul32fICtx( pSrc *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_32f_I_Ctx( pSrc *Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Mul32fI( pSrc *Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsMul_32f_I( pSrc *Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func Mul64fICtx( Npp64f * pSrc, Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_64f_I_Ctx( Npp64f * pSrc, Npp64f * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Mul64fI( Npp64f * pSrc, Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsMul_64f_I( Npp64f * pSrc, Npp64f * pSrcDst, (C.int)(nLength))).ToError()
}
func Mul32fcICtx( pSrc *Float32Complex, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_32fc_I_Ctx( pSrc *Float32Complex, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Mul32fcI( pSrc *Float32Complex, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsMul_32fc_I( pSrc *Float32Complex, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func Mul64fcICtx( Npp64fc * pSrc, Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_64fc_I_Ctx( Npp64fc * pSrc, Npp64fc * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Mul64fcI( Npp64fc * pSrc, Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsMul_64fc_I( Npp64fc * pSrc, Npp64fc * pSrcDst, (C.int)(nLength))).ToError()
}
func Mul32f32fcICtx( pSrc *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsMul_32f32fc_I_Ctx( pSrc *Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Mul32f32fcI( pSrc *Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsMul_32f32fc_I( pSrc *Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func Mul8uISfsCtx(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_8u_ISfs_Ctx(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul8uISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_8u_ISfs(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul16uISfsCtx( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_16u_ISfs_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul16uISfs( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_16u_ISfs( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul16sISfsCtx( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_16s_ISfs_Ctx( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul16sISfs( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_16s_ISfs( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul32sISfsCtx( Npp32s * pSrc, Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_32s_ISfs_Ctx( Npp32s * pSrc, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul32sISfs( Npp32s * pSrc, Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_32s_ISfs( Npp32s * pSrc, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul16scISfsCtx( Npp16sc * pSrc, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_16sc_ISfs_Ctx( Npp16sc * pSrc, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul16scISfs( Npp16sc * pSrc, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_16sc_ISfs( Npp16sc * pSrc, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul32scISfsCtx( Npp32sc * pSrc, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_32sc_ISfs_Ctx( Npp32sc * pSrc, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul32scISfs( Npp32sc * pSrc, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_32sc_ISfs( Npp32sc * pSrc, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Mul32s32scISfsCtx( Npp32s * pSrc, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsMul_32s32sc_ISfs_Ctx( Npp32s * pSrc, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Mul32s32scISfs( Npp32s * pSrc, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsMul_32s32sc_ISfs( Npp32s * pSrc, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub16sCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_16s_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Sub16s( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32) error{
  return status(C.nppsSub_16s( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength))).ToError()
}
func Sub32fCtx( pSrc *Float321,pSrc *Float322, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_32f_Ctx( pSrc *Float321,pSrc *Float322, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Sub32f( pSrc *Float321,pSrc *Float322, pDst *Float32, nLength int32) error{
  return status(C.nppsSub_32f( pSrc *Float321,pSrc *Float322, pDst *Float32, (C.int)(nLength))).ToError()
}
func Sub64fCtx( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_64f_Ctx( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Sub64f( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, nLength int32) error{
  return status(C.nppsSub_64f( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func Sub32fcCtx( pSrc1 *Float32Complex,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_32fc_Ctx( pSrc1.cptr(),pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func Sub32fc( pSrc1 *Float32Complex,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsSub_32fc( pSrc1.cptr(),pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
}
func Sub64fcCtx( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_64fc_Ctx( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Sub64fc( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsSub_64fc( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, (C.int)(nLength))).ToError()
}
func Sub16s32fCtx( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_16s32f_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Sub16s32f( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, nLength int32) error{
  return status(C.nppsSub_16s32f( Npp16s * pSrc1,Npp16s * pSrc2, pDst *Float32, (C.int)(nLength))).ToError()
}
func Sub8uSfsCtx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_8u_Sfs_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub8uSfs(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_8u_Sfs(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub16uSfsCtx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_16u_Sfs_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub16uSfs( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_16u_Sfs( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub16sSfsCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_16s_Sfs_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub16sSfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_16s_Sfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub32sSfsCtx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_32s_Sfs_Ctx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub32sSfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_32s_Sfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub16scSfsCtx( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_16sc_Sfs_Ctx( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub16scSfs( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_16sc_Sfs( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub32scSfsCtx( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_32sc_Sfs_Ctx( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub32scSfs( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_32sc_Sfs( Npp32sc * pSrc1,Npp32sc * pSrc2, Npp32sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub16sICtx( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_16s_I_Ctx( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Sub16sI( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32) error{
  return status(C.nppsSub_16s_I( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength))).ToError()
}
func Sub32fICtx( pSrc *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_32f_I_Ctx( pSrc *Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Sub32fI( pSrc *Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsSub_32f_I( pSrc *Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func Sub64fICtx( Npp64f * pSrc, Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_64f_I_Ctx( Npp64f * pSrc, Npp64f * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Sub64fI( Npp64f * pSrc, Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsSub_64f_I( Npp64f * pSrc, Npp64f * pSrcDst, (C.int)(nLength))).ToError()
}
func Sub32fcICtx( pSrc *Float32Complex, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_32fc_I_Ctx( pSrc *Float32Complex, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Sub32fcI( pSrc *Float32Complex, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsSub_32fc_I( pSrc *Float32Complex, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func Sub64fcICtx( Npp64fc * pSrc, Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSub_64fc_I_Ctx( Npp64fc * pSrc, Npp64fc * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Sub64fcI( Npp64fc * pSrc, Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsSub_64fc_I( Npp64fc * pSrc, Npp64fc * pSrcDst, (C.int)(nLength))).ToError()
}
func Sub8uISfsCtx(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_8u_ISfs_Ctx(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub8uISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_8u_ISfs(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub16uISfsCtx( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_16u_ISfs_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub16uISfs( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_16u_ISfs( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub16sISfsCtx( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_16s_ISfs_Ctx( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub16sISfs( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_16s_ISfs( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub32sISfsCtx( Npp32s * pSrc, Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_32s_ISfs_Ctx( Npp32s * pSrc, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub32sISfs( Npp32s * pSrc, Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_32s_ISfs( Npp32s * pSrc, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub16scISfsCtx( Npp16sc * pSrc, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_16sc_ISfs_Ctx( Npp16sc * pSrc, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub16scISfs( Npp16sc * pSrc, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_16sc_ISfs( Npp16sc * pSrc, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Sub32scISfsCtx( Npp32sc * pSrc, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSub_32sc_ISfs_Ctx( Npp32sc * pSrc, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Sub32scISfs( Npp32sc * pSrc, Npp32sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSub_32sc_ISfs( Npp32sc * pSrc, Npp32sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div8uSfsCtx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_8u_Sfs_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div8uSfs(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_8u_Sfs(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div16uSfsCtx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_16u_Sfs_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div16uSfs( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_16u_Sfs( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div16sSfsCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_16s_Sfs_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div16sSfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_16s_Sfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div32sSfsCtx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_32s_Sfs_Ctx( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div32sSfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_32s_Sfs( Npp32s * pSrc1,Npp32s * pSrc2, Npp32s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div16scSfsCtx( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_16sc_Sfs_Ctx( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div16scSfs( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_16sc_Sfs( Npp16sc * pSrc1,Npp16sc * pSrc2, Npp16sc * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div32s16sSfsCtx( Npp16s * pSrc1,Npp32s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_32s16s_Sfs_Ctx( Npp16s * pSrc1,Npp32s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div32s16sSfs( Npp16s * pSrc1,Npp32s * pSrc2, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_32s16s_Sfs( Npp16s * pSrc1,Npp32s * pSrc2, Npp16s * pDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div32fCtx( pSrc *Float321,pSrc *Float322, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDiv_32f_Ctx( pSrc *Float321,pSrc *Float322, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Div32f( pSrc *Float321,pSrc *Float322, pDst *Float32, nLength int32) error{
  return status(C.nppsDiv_32f( pSrc *Float321,pSrc *Float322, pDst *Float32, (C.int)(nLength))).ToError()
}
func Div64fCtx( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDiv_64f_Ctx( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Div64f( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, nLength int32) error{
  return status(C.nppsDiv_64f( Npp64f * pSrc1,Npp64f * pSrc2, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func Div32fcCtx( pSrc1 *Float32Complex,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDiv_32fc_Ctx( pSrc1.cptr(),pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
func Div32fc( pSrc1 *Float32Complex,pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsDiv_32fc( pSrc1.cptr(),pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
}
func Div64fcCtx( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDiv_64fc_Ctx( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Div64fc( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsDiv_64fc( Npp64fc * pSrc1,Npp64fc * pSrc2, Npp64fc * pDst, (C.int)(nLength))).ToError()
}
func Div8uISfsCtx(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_8u_ISfs_Ctx(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div8uISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_8u_ISfs(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div16uISfsCtx( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_16u_ISfs_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div16uISfs( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_16u_ISfs( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div16sISfsCtx( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_16s_ISfs_Ctx( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div16sISfs( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_16s_ISfs( Npp16s * pSrc, Npp16s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div16scISfsCtx( Npp16sc * pSrc, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_16sc_ISfs_Ctx( Npp16sc * pSrc, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div16scISfs( Npp16sc * pSrc, Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_16sc_ISfs( Npp16sc * pSrc, Npp16sc * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div32sISfsCtx( Npp32s * pSrc, Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_32s_ISfs_Ctx( Npp32s * pSrc, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32, ctx.c())).ToError()
}
func Div32sISfs( Npp32s * pSrc, Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsDiv_32s_ISfs( Npp32s * pSrc, Npp32s * pSrcDst, (C.int)(nLength), nScaleFactor int32)).ToError()
}
func Div32fICtx( pSrc *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDiv_32f_I_Ctx( pSrc *Float32, pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func Div32fI( pSrc *Float32, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsDiv_32f_I( pSrc *Float32, pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func Div64fICtx( Npp64f * pSrc, Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDiv_64f_I_Ctx( Npp64f * pSrc, Npp64f * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Div64fI( Npp64f * pSrc, Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsDiv_64f_I( Npp64f * pSrc, Npp64f * pSrcDst, nLength int32)).ToError()
}
func Div32fcICtx( pSrc *Float32Complex, pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDiv_32fc_I_Ctx( pSrc *Float32Complex, pSrcDst *Float32, nLength int32, ctx.c())).ToError()
}
func Div32fcI( pSrc *Float32Complex, pSrcDst *Float32, nLength int32) error{
  return status(C.nppsDiv_32fc_I( pSrc *Float32Complex, pSrcDst *Float32, nLength int32)).ToError()
}
func Div64fcICtx( Npp64fc * pSrc, Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsDiv_64fc_I_Ctx( Npp64fc * pSrc, Npp64fc * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Div64fcI( Npp64fc * pSrc, Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsDiv_64fc_I( Npp64fc * pSrc, Npp64fc * pSrcDst, nLength int32)).ToError()
}
func DivRound8uSfsCtx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_Round_8u_Sfs_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx.c())).ToError()
}
func DivRound8uSfs(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, NppRoundMode nRndMode, nScaleFactor int32) error{
  return status(C.nppsDiv_Round_8u_Sfs(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, NppRoundMode nRndMode, nScaleFactor int32)).ToError()
}
func DivRound16uSfsCtx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_Round_16u_Sfs_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx.c())).ToError()
}
func DivRound16uSfs( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32) error{
  return status(C.nppsDiv_Round_16u_Sfs( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32)).ToError()
}
func DivRound16sSfsCtx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_Round_16s_Sfs_Ctx( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx.c())).ToError()
}
func DivRound16sSfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32) error{
  return status(C.nppsDiv_Round_16s_Sfs( Npp16s * pSrc1,Npp16s * pSrc2, Npp16s * pDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32)).ToError()
}
func DivRound8uISfsCtx(pSrc *Uint8, pSrcDst *Uint8, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_Round_8u_ISfs_Ctx(pSrc *Uint8, pSrcDst *Uint8, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx.c())).ToError()
}
func DivRound8uISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, NppRoundMode nRndMode, nScaleFactor int32) error{
  return status(C.nppsDiv_Round_8u_ISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, NppRoundMode nRndMode, nScaleFactor int32)).ToError()
}
func DivRound16uISfsCtx( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_Round_16u_ISfs_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx.c())).ToError()
}
func DivRound16uISfs( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32) error{
  return status(C.nppsDiv_Round_16u_ISfs( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32)).ToError()
}
func DivRound16sISfsCtx( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsDiv_Round_16s_ISfs_Ctx( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32, ctx.c())).ToError()
}
func DivRound16sISfs( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32) error{
  return status(C.nppsDiv_Round_16s_ISfs( Npp16s * pSrc, Npp16s * pSrcDst, nLength int32, NppRoundMode nRndMode, nScaleFactor int32)).ToError()
}
func Abs16sCtx( Npp16s * pSrc, Npp16s * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAbs_16s_Ctx( Npp16s * pSrc, Npp16s * pDst, nLength int32, ctx.c())).ToError()
}
func Abs16s( Npp16s * pSrc, Npp16s * pDst, nLength int32) error{
  return status(C.nppsAbs_16s( Npp16s * pSrc, Npp16s * pDst, nLength int32)).ToError()
}
func Abs32s_Ctx( Npp32s * pSrc, Npp32s * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAbs_32s_Ctx( Npp32s * pSrc, Npp32s * pDst, nLength int32, ctx.c())).ToError()
}
func Abs32s( Npp32s * pSrc, Npp32s * pDst, nLength int32) error{
  return status(C.nppsAbs_32s( Npp32s * pSrc, Npp32s * pDst, nLength int32)).ToError()
}
func Abs32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAbs_32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx.c())).ToError()
}
func Abs32f( pSrc *Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsAbs_32f( pSrc *Float32, pDst *Float32, nLength int32)).ToError()
}
func Abs64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAbs_64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx.c())).ToError()
}
func Abs64f( Npp64f * pSrc, Npp64f * pDst, nLength int32) error{
  return status(C.nppsAbs_64f( Npp64f * pSrc, Npp64f * pDst, nLength int32)).ToError()
}
func Abs16s_I_Ctx(Npp16s * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAbs_16s_I_Ctx(Npp16s * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Abs16s_I(Npp16s * pSrcDst, nLength int32) error{
  return status(C.nppsAbs_16s_I(Npp16s * pSrcDst, nLength int32)).ToError()
}
func Abs32s_I_Ctx(Npp32s * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAbs_32s_I_Ctx(Npp32s * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Abs32s_I(Npp32s * pSrcDst, nLength int32) error{
  return status(C.nppsAbs_32s_I(Npp32s * pSrcDst, nLength int32)).ToError()
}
func Abs32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAbs_32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx.c())).ToError()
}
func Abs32f_I(pSrcDst *Float32, nLength int32) error{
  return status(C.nppsAbs_32f_I(pSrcDst *Float32, nLength int32)).ToError()
}
func Abs64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAbs_64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Abs64f_I(Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsAbs_64f_I(Npp64f * pSrcDst, nLength int32)).ToError()
}
func Sqr32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqr_32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx.c())).ToError()
}
func Sqr32f( pSrc *Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsSqr_32f( pSrc *Float32, pDst *Float32, nLength int32)).ToError()
}
func Sqr64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqr_64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx.c())).ToError()
}
func Sqr64f( Npp64f * pSrc, Npp64f * pDst, nLength int32) error{
  return status(C.nppsSqr_64f( Npp64f * pSrc, Npp64f * pDst, nLength int32)).ToError()
}
func Sqr32fc_Ctx( pSrc *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqr_32fc_Ctx( pSrc *Float32Complex, pDst.cptr(), nLength int32, ctx.c())).ToError()
}
func Sqr32fc( pSrc *Float32Complex, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsSqr_32fc( pSrc *Float32Complex, pDst.cptr(), nLength int32)).ToError()
}
func Sqr64fc_Ctx( Npp64fc * pSrc, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqr_64fc_Ctx( Npp64fc * pSrc, Npp64fc * pDst, nLength int32, ctx.c())).ToError()
}
func Sqr64fc( Npp64fc * pSrc, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsSqr_64fc( Npp64fc * pSrc, Npp64fc * pDst, nLength int32)).ToError()
}
func Sqr32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqr_32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx.c())).ToError()
}
func Sqr32f_I(pSrcDst *Float32, nLength int32) error{
  return status(C.nppsSqr_32f_I(pSrcDst *Float32, nLength int32)).ToError()
}
func Sqr64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqr_64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Sqr64f_I(Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsSqr_64f_I(Npp64f * pSrcDst, nLength int32)).ToError()
}
func Sqr32fc_I_Ctx(pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqr_32fc_I_Ctx(pSrcDst *Float32, nLength int32, ctx.c())).ToError()
}
func Sqr32fc_I(pSrcDst *Float32, nLength int32) error{
  return status(C.nppsSqr_32fc_I(pSrcDst *Float32, nLength int32)).ToError()
}
func Sqr64fc_I_Ctx(Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqr_64fc_I_Ctx(Npp64fc * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Sqr64fc_I(Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsSqr_64fc_I(Npp64fc * pSrcDst, nLength int32)).ToError()
}
func Sqr8u_Sfs_Ctx(pSrc *Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqr_8u_Sfs_Ctx(pSrc *Uint8, pDst *Uint8, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqr8u_Sfs(pSrc *Uint8, pDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqr_8u_Sfs(pSrc *Uint8, pDst *Uint8, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqr16u_Sfs_Ctx( Npp16u * pSrc, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqr_16u_Sfs_Ctx( Npp16u * pSrc, Npp16u * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqr16u_Sfs( Npp16u * pSrc, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqr_16u_Sfs( Npp16u * pSrc, Npp16u * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqr16s_Sfs_Ctx( Npp16s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqr_16s_Sfs_Ctx( Npp16s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqr16s_Sfs( Npp16s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqr_16s_Sfs( Npp16s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqr16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqr_16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqr16sc_Sfs( Npp16sc * pSrc, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqr_16sc_Sfs( Npp16sc * pSrc, Npp16sc * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqr8u_ISfs_Ctx(pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqr_8u_ISfs_Ctx(pSrcDst *Uint8, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqr8u_ISfs(pSrcDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqr_8u_ISfs(pSrcDst *Uint8, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqr16u_ISfs_Ctx(Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqr_16u_ISfs_Ctx(Npp16u * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqr16u_ISfs(Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqr_16u_ISfs(Npp16u * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqr16s_ISfs_Ctx(Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqr_16s_ISfs_Ctx(Npp16s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqr16s_ISfs(Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqr_16s_ISfs(Npp16s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqr16sc_ISfs_Ctx(Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqr_16sc_ISfs_Ctx(Npp16sc * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqr16sc_ISfs(Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqr_16sc_ISfs(Npp16sc * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx.c())).ToError()
}
func Sqrt32f( pSrc *Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsSqrt_32f( pSrc *Float32, pDst *Float32, nLength int32)).ToError()
}
func Sqrt64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx.c())).ToError()
}
func Sqrt64f( Npp64f * pSrc, Npp64f * pDst, nLength int32) error{
  return status(C.nppsSqrt_64f( Npp64f * pSrc, Npp64f * pDst, nLength int32)).ToError()
}
func Sqrt32fc_Ctx( pSrc *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_32fc_Ctx( pSrc *Float32Complex, pDst.cptr(), nLength int32, ctx.c())).ToError()
}
func Sqrt32fc( pSrc *Float32Complex, pDst *Float32Complex, nLength int32) error{
  return status(C.nppsSqrt_32fc( pSrc *Float32Complex, pDst.cptr(), nLength int32)).ToError()
}
func Sqrt64fc_Ctx( Npp64fc * pSrc, Npp64fc * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_64fc_Ctx( Npp64fc * pSrc, Npp64fc * pDst, nLength int32, ctx.c())).ToError()
}
func Sqrt64fc( Npp64fc * pSrc, Npp64fc * pDst, nLength int32) error{
  return status(C.nppsSqrt_64fc( Npp64fc * pSrc, Npp64fc * pDst, nLength int32)).ToError()
}
func Sqrt32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx.c())).ToError()
}
func Sqrt32f_I(pSrcDst *Float32, nLength int32) error{
  return status(C.nppsSqrt_32f_I(pSrcDst *Float32, nLength int32)).ToError()
}
func Sqrt64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Sqrt64f_I(Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsSqrt_64f_I(Npp64f * pSrcDst, nLength int32)).ToError()
}
func Sqrt32fc_I_Ctx(pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_32fc_I_Ctx(pSrcDst *Float32, nLength int32, ctx.c())).ToError()
}
func Sqrt32fc_I(pSrcDst *Float32, nLength int32) error{
  return status(C.nppsSqrt_32fc_I(pSrcDst *Float32, nLength int32)).ToError()
}
func Sqrt64fc_I_Ctx(Npp64fc * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_64fc_I_Ctx(Npp64fc * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Sqrt64fc_I(Npp64fc * pSrcDst, nLength int32) error{
  return status(C.nppsSqrt_64fc_I(Npp64fc * pSrcDst, nLength int32)).ToError()
}
func Sqrt8u_Sfs_Ctx(pSrc *Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_8u_Sfs_Ctx(pSrc *Uint8, pDst *Uint8, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt8u_Sfs(pSrc *Uint8, pDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_8u_Sfs(pSrc *Uint8, pDst *Uint8, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt16u_Sfs_Ctx( Npp16u * pSrc, Npp16u * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_16u_Sfs_Ctx( Npp16u * pSrc, Npp16u * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt16u_Sfs( Npp16u * pSrc, Npp16u * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_16u_Sfs( Npp16u * pSrc, Npp16u * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt16s_Sfs_Ctx( Npp16s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_16s_Sfs_Ctx( Npp16s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt16s_Sfs( Npp16s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_16s_Sfs( Npp16s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt16sc_Sfs( Npp16sc * pSrc, Npp16sc * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_16sc_Sfs( Npp16sc * pSrc, Npp16sc * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt64s_Sfs_Ctx( Npp64s * pSrc, Npp64s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_64s_Sfs_Ctx( Npp64s * pSrc, Npp64s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt64s_Sfs( Npp64s * pSrc, Npp64s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_64s_Sfs( Npp64s * pSrc, Npp64s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt32s16s_Sfs_Ctx( Npp32s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_32s16s_Sfs_Ctx( Npp32s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt32s16s_Sfs( Npp32s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_32s16s_Sfs( Npp32s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt64s16s_Sfs_Ctx( Npp64s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_64s16s_Sfs_Ctx( Npp64s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt64s16s_Sfs( Npp64s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_64s16s_Sfs( Npp64s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt8u_ISfs_Ctx(pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_8u_ISfs_Ctx(pSrcDst *Uint8, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt8u_ISfs(pSrcDst *Uint8, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_8u_ISfs(pSrcDst *Uint8, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt16u_ISfs_Ctx(Npp16u * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_16u_ISfs_Ctx(Npp16u * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt16u_ISfs(Npp16u * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_16u_ISfs(Npp16u * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt16s_ISfs_Ctx(Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_16s_ISfs_Ctx(Npp16s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt16s_ISfs(Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_16s_ISfs(Npp16s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt16sc_ISfs_Ctx(Npp16sc * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_16sc_ISfs_Ctx(Npp16sc * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt16sc_ISfs(Npp16sc * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_16sc_ISfs(Npp16sc * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Sqrt64s_ISfs_Ctx(Npp64s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsSqrt_64s_ISfs_Ctx(Npp64s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Sqrt64s_ISfs(Npp64s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsSqrt_64s_ISfs(Npp64s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Cubrt32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsCubrt_32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx.c())).ToError()
}
func Cubrt32f( pSrc *Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsCubrt_32f( pSrc *Float32, pDst *Float32, nLength int32)).ToError()
}
func Cubrt32s16s_Sfs_Ctx( Npp32s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsCubrt_32s16s_Sfs_Ctx( Npp32s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Cubrt32s16s_Sfs( Npp32s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsCubrt_32s16s_Sfs( Npp32s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Exp32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsExp_32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx.c())).ToError()
}
func Exp32f( pSrc *Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsExp_32f( pSrc *Float32, pDst *Float32, nLength int32)).ToError()
}
func Exp64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsExp_64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx.c())).ToError()
}
func Exp64f( Npp64f * pSrc, Npp64f * pDst, nLength int32) error{
  return status(C.nppsExp_64f( Npp64f * pSrc, Npp64f * pDst, nLength int32)).ToError()
}
func Exp32f64f_Ctx( pSrc *Float32, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsExp_32f64f_Ctx( pSrc *Float32, Npp64f * pDst, nLength int32, ctx.c())).ToError()
}
func Exp32f64f( pSrc *Float32, Npp64f * pDst, nLength int32) error{
  return status(C.nppsExp_32f64f( pSrc *Float32, Npp64f * pDst, nLength int32)).ToError()
}
func Exp32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsExp_32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx.c())).ToError()
}
func Exp32f_I(pSrcDst *Float32, nLength int32) error{
  return status(C.nppsExp_32f_I(pSrcDst *Float32, nLength int32)).ToError()
}
func Exp64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsExp_64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Exp64f_I(Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsExp_64f_I(Npp64f * pSrcDst, nLength int32)).ToError()
}
func Exp16s_Sfs_Ctx( Npp16s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsExp_16s_Sfs_Ctx( Npp16s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Exp16s_Sfs( Npp16s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsExp_16s_Sfs( Npp16s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Exp32s_Sfs_Ctx( Npp32s * pSrc, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsExp_32s_Sfs_Ctx( Npp32s * pSrc, Npp32s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Exp32s_Sfs( Npp32s * pSrc, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsExp_32s_Sfs( Npp32s * pSrc, Npp32s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Exp64s_Sfs_Ctx( Npp64s * pSrc, Npp64s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsExp_64s_Sfs_Ctx( Npp64s * pSrc, Npp64s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Exp64s_Sfs( Npp64s * pSrc, Npp64s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsExp_64s_Sfs( Npp64s * pSrc, Npp64s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Exp16s_ISfs_Ctx(Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsExp_16s_ISfs_Ctx(Npp16s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Exp16s_ISfs(Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsExp_16s_ISfs(Npp16s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Exp32s_ISfs_Ctx(Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsExp_32s_ISfs_Ctx(Npp32s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Exp32s_ISfs(Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsExp_32s_ISfs(Npp32s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Exp64s_ISfs_Ctx(Npp64s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsExp_64s_ISfs_Ctx(Npp64s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Exp64s_ISfs(Npp64s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsExp_64s_ISfs(Npp64s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Ln32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLn_32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx.c())).ToError()
}
func Ln32f( pSrc *Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsLn_32f( pSrc *Float32, pDst *Float32, nLength int32)).ToError()
}
func Ln64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLn_64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx.c())).ToError()
}
func Ln64f( Npp64f * pSrc, Npp64f * pDst, nLength int32) error{
  return status(C.nppsLn_64f( Npp64f * pSrc, Npp64f * pDst, nLength int32)).ToError()
}
func Ln64f32f_Ctx( Npp64f * pSrc, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLn_64f32f_Ctx( Npp64f * pSrc, pDst *Float32, nLength int32, ctx.c())).ToError()
}
func Ln64f32f( Npp64f * pSrc, pDst *Float32, nLength int32) error{
  return status(C.nppsLn_64f32f( Npp64f * pSrc, pDst *Float32, nLength int32)).ToError()
}
func Ln32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLn_32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx.c())).ToError()
}
func Ln32f_I(pSrcDst *Float32, nLength int32) error{
  return status(C.nppsLn_32f_I(pSrcDst *Float32, nLength int32)).ToError()
}
func Ln64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLn_64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx.c())).ToError()
}
func Ln64f_I(Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsLn_64f_I(Npp64f * pSrcDst, nLength int32)).ToError()
}
func Ln16s_Sfs_Ctx( Npp16s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsLn_16s_Sfs_Ctx( Npp16s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Ln16s_Sfs( Npp16s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsLn_16s_Sfs( Npp16s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Ln32s_Sfs_Ctx( Npp32s * pSrc, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsLn_32s_Sfs_Ctx( Npp32s * pSrc, Npp32s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Ln32s_Sfs( Npp32s * pSrc, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsLn_32s_Sfs( Npp32s * pSrc, Npp32s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Ln32s16s_Sfs_Ctx( Npp32s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsLn_32s16s_Sfs_Ctx( Npp32s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Ln32s16s_Sfs( Npp32s * pSrc, Npp16s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsLn_32s16s_Sfs( Npp32s * pSrc, Npp16s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Ln16s_ISfs_Ctx(Npp16s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsLn_16s_ISfs_Ctx(Npp16s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Ln16s_ISfs(Npp16s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsLn_16s_ISfs(Npp16s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func Ln32s_ISfs_Ctx(Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsLn_32s_ISfs_Ctx(Npp32s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func Ln32s_ISfs(Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.nppsLn_32s_ISfs(Npp32s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func TenLogTen32sSfs_Ctx( Npp32s * pSrc, Npp32s * pDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.npps10Log10_32s_Sfs_Ctx( Npp32s * pSrc, Npp32s * pDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func TenLogTen32sSfs( Npp32s * pSrc, Npp32s * pDst, nLength int32, nScaleFactor int32) error{
  return status(C.npps10Log10_32s_Sfs( Npp32s * pSrc, Npp32s * pDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func TenLogTen32sISfs_Ctx(Npp32s * pSrcDst, nLength int32, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.npps10Log10_32s_ISfs_Ctx(Npp32s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}
func TenLogTen32sISfs(Npp32s * pSrcDst, nLength int32, nScaleFactor int32) error{
  return status(C.npps10Log10_32s_ISfs(Npp32s * pSrcDst, (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
}
func SumLnGetBufferSize32fCtx(nLength int32, int * hpBufferSize, ctx *StreamContext) error{
  return status(C.nppsSumLnGetBufferSize_32f_Ctx((C.int)(nLength), int * hpBufferSize, ctx.c())).ToError()
}
func SumLnGetBufferSize32f(nLength int32, int * hpBufferSize) error{
  return status(C.nppsSumLnGetBufferSize_32f((C.int)(nLength), int * hpBufferSize)).ToError()
}
func  SumLn32f_Ctx( pSrc *Float32, nLength int32, pDst *Float32, Npp8u * pDeviceBuffer, ctx *StreamContext) error{
  return status(C.nppsSumLn_32f_Ctx( pSrc *Float32, (C.int)(nLength), pDst *Float32, Npp8u * pDeviceBuffer, ctx.c())).ToError()
}
func  SumLn32f( pSrc *Float32, nLength int32, pDst *Float32, Npp8u * pDeviceBuffer) error{
  return status(C.nppsSumLn_32f( pSrc *Float32, (C.int)(nLength), pDst *Float32, Npp8u * pDeviceBuffer)).ToError()
}
func SumLnGetBufferSize_64f_Ctx(nLength int32, int * hpBufferSize, ctx *StreamContext) error{
  return status(C.nppsSumLnGetBufferSize_64f_Ctx((C.int)(nLength), int * hpBufferSize, ctx.c())).ToError()
}
func SumLnGetBufferSize_64f(nLength int32, int * hpBufferSize) error{
  return status(C.nppsSumLnGetBufferSize_64f((C.int)(nLength), int * hpBufferSize)).ToError()
}
func  SumLn64f_Ctx( Npp64f * pSrc, nLength int32, Npp64f * pDst, Npp8u * pDeviceBuffer, ctx *StreamContext) error{
  return status(C.nppsSumLn_64f_Ctx( Npp64f * pSrc, (C.int)(nLength), Npp64f * pDst, Npp8u * pDeviceBuffer, ctx.c())).ToError()
}
func  SumLn64f( Npp64f * pSrc, nLength int32, Npp64f * pDst, Npp8u * pDeviceBuffer) error{
  return status(C.nppsSumLn_64f( Npp64f * pSrc, (C.int)(nLength), Npp64f * pDst, Npp8u * pDeviceBuffer)).ToError()
}
func SumLnGetBufferSize_32f64f_Ctx(nLength int32, int * hpBufferSize, ctx *StreamContext) error{
  return status(C.nppsSumLnGetBufferSize_32f64f_Ctx((C.int)(nLength), int * hpBufferSize, ctx.c())).ToError()
}
func SumLnGetBufferSize_32f64f(nLength int32, int * hpBufferSize) error{
  return status(C.nppsSumLnGetBufferSize_32f64f((C.int)(nLength), int * hpBufferSize)).ToError()
}
func  SumLn32f64f_Ctx( pSrc *Float32, nLength int32, Npp64f * pDst, Npp8u * pDeviceBuffer, ctx *StreamContext) error{
  return status(C.nppsSumLn_32f64f_Ctx( pSrc *Float32, (C.int)(nLength), Npp64f * pDst, Npp8u * pDeviceBuffer, ctx.c())).ToError()
}
func SumLn32f64f( pSrc *Float32, nLength int32, Npp64f * pDst, Npp8u * pDeviceBuffer) error{
  return status(C.nppsSumLn_32f64f( pSrc *Float32, (C.int)(nLength), Npp64f * pDst, Npp8u * pDeviceBuffer)).ToError()
}
func SumLnGetBufferSize_16s32f_Ctx(nLength int32, int * hpBufferSize, ctx *StreamContext) error{
  return status(C.nppsSumLnGetBufferSize_16s32f_Ctx((C.int)(nLength), int * hpBufferSize, ctx.c())).ToError()
}
func SumLnGetBufferSize_16s32f(nLength int32, int * hpBufferSize) error{
  return status(C.nppsSumLnGetBufferSize_16s32f((C.int)(nLength), int * hpBufferSize)).ToError()
}
func SumLn16s32f_Ctx( Npp16s * pSrc, nLength int32, pDst *Float32, Npp8u * pDeviceBuffer, ctx *StreamContext) error{
  return status(C.nppsSumLn_16s32f_Ctx( Npp16s * pSrc, (C.int)(nLength), pDst *Float32, Npp8u * pDeviceBuffer, ctx.c())).ToError()
}
func SumLn16s32f( Npp16s * pSrc, nLength int32, pDst *Float32, Npp8u * pDeviceBuffer) error{
  return status(C.nppsSumLn_16s32f( Npp16s * pSrc, (C.int)(nLength), pDst *Float32, Npp8u * pDeviceBuffer)).ToError()
}
func  Arctan32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsArctan_32f_Ctx( pSrc *Float32, pDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func  Arctan32f( pSrc *Float32, pDst *Float32, nLength int32) error{
  return status(C.nppsArctan_32f( pSrc *Float32, pDst *Float32, (C.int)(nLength))).ToError()
}
func  Arctan64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsArctan_64f_Ctx( Npp64f * pSrc, Npp64f * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func  Arctan64f( Npp64f * pSrc, Npp64f * pDst, nLength int32) error{
  return status(C.nppsArctan_64f( Npp64f * pSrc, Npp64f * pDst, (C.int)(nLength))).ToError()
}
func  Arctan32f_I_Ctx(pSrcDst *Float32, nLength int32, ctx *StreamContext) error{
  return status(C.nppsArctan_32f_I_Ctx(pSrcDst *Float32, (C.int)(nLength), ctx.c())).ToError()
}
func  Arctan32f_I(pSrcDst *Float32, nLength int32) error{
  return status(C.nppsArctan_32f_I(pSrcDst *Float32, (C.int)(nLength))).ToError()
}
func  Arctan64f_I_Ctx(Npp64f * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsArctan_64f_I_Ctx(Npp64f * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func  Arctan64f_I(Npp64f * pSrcDst, nLength int32) error{
  return status(C.nppsArctan_64f_I(Npp64f * pSrcDst, (C.int)(nLength))).ToError()
}
func Normalize32f_Ctx( pSrc *Float32, pDst *Float32, nLength int32, Npp32f vSub, Npp32f vDiv, ctx *StreamContext) error{
  return status(C.nppsNormalize_32f_Ctx( pSrc *Float32, pDst *Float32, (C.int)(nLength), Npp32f vSub, Npp32f vDiv, ctx.c())).ToError()
}
func Normalize32f( pSrc *Float32, pDst *Float32, nLength int32, Npp32f vSub, Npp32f vDiv) error{
  return status(C.nppsNormalize_32f( pSrc *Float32, pDst *Float32, (C.int)(nLength), Npp32f vSub, Npp32f vDiv)).ToError()
}
func Normalize32fc_Ctx( pSrc *Float32Complex, pDst *Float32Complex, nLength int32, Npp32fc vSub, Npp32f vDiv, ctx *StreamContext) error{
  return status(C.nppsNormalize_32fc_Ctx( pSrc *Float32Complex, pDst.cptr(), (C.int)(nLength), Npp32fc vSub, Npp32f vDiv, ctx.c())).ToError()
}
func Normalize32fc( pSrc *Float32Complex, pDst *Float32Complex, nLength int32, Npp32fc vSub, Npp32f vDiv) error{
  return status(C.nppsNormalize_32fc( pSrc *Float32Complex, pDst.cptr(), (C.int)(nLength), Npp32fc vSub, Npp32f vDiv)).ToError()
}
func Normalize64f_Ctx( Npp64f * pSrc, Npp64f * pDst, nLength int32, Npp64f vSub, Npp64f vDiv, ctx *StreamContext) error{
  return status(C.nppsNormalize_64f_Ctx( Npp64f * pSrc, Npp64f * pDst, (C.int)(nLength), Npp64f vSub, Npp64f vDiv, ctx.c())).ToError()
}
func Normalize64f( Npp64f * pSrc, Npp64f * pDst, nLength int32, Npp64f vSub, Npp64f vDiv) error{
  return status(C.nppsNormalize_64f( Npp64f * pSrc, Npp64f * pDst, (C.int)(nLength), Npp64f vSub, Npp64f vDiv)).ToError()
}
func Normalize64fc_Ctx( Npp64fc * pSrc, Npp64fc * pDst, nLength int32, Npp64fc vSub, Npp64f vDiv, ctx *StreamContext) error{
  return status(C.nppsNormalize_64fc_Ctx( Npp64fc * pSrc, Npp64fc * pDst, (C.int)(nLength), Npp64fc vSub, Npp64f vDiv, ctx.c())).ToError()
}
func Normalize64fc( Npp64fc * pSrc, Npp64fc * pDst, nLength int32, Npp64fc vSub, Npp64f vDiv) error{
  return status(C.nppsNormalize_64fc( Npp64fc * pSrc, Npp64fc * pDst, (C.int)(nLength), Npp64fc vSub, Npp64f vDiv)).ToError()
}
func Normalize16s_Sfs_Ctx( Npp16s * pSrc, Npp16s * pDst, nLength int32, Npp16s vSub, int vDiv, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsNormalize_16s_Sfs_Ctx( Npp16s * pSrc, Npp16s * pDst, (C.int)(nLength), Npp16s vSub, int vDiv, nScaleFactor int32, ctx.c())).ToError()
}
func Normalize16s_Sfs( Npp16s * pSrc, Npp16s * pDst, nLength int32, Npp16s vSub, int vDiv, nScaleFactor int32) error{
  return status(C.nppsNormalize_16s_Sfs( Npp16s * pSrc, Npp16s * pDst, (C.int)(nLength), Npp16s vSub, int vDiv, nScaleFactor int32)).ToError()
}
func Normalize16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc * pDst, nLength int32, Npp16sc vSub, int vDiv, nScaleFactor int32, ctx *StreamContext) error{
  return status(C.nppsNormalize_16sc_Sfs_Ctx( Npp16sc * pSrc, Npp16sc * pDst, (C.int)(nLength), Npp16sc vSub, int vDiv, nScaleFactor int32, ctx.c())).ToError()
}
func Normalize16sc_Sfs( Npp16sc * pSrc, Npp16sc * pDst, nLength int32, Npp16sc vSub, int vDiv, nScaleFactor int32) error{
  return status(C.nppsNormalize_16sc_Sfs( Npp16sc * pSrc, Npp16sc * pDst, (C.int)(nLength), Npp16sc vSub, int vDiv, nScaleFactor int32)).ToError()
}
func Cauchy32fI_Ctx(pSrcDst *Float32, nLength int32, Npp32f nParam, ctx *StreamContext) error{
  return status(C.nppsCauchy_32f_I_Ctx(pSrcDst *Float32, (C.int)(nLength), Npp32f nParam, ctx.c())).ToError()
}
func Cauchy32fI(pSrcDst *Float32, nLength int32, Npp32f nParam) error{
  return status(C.nppsCauchy_32f_I(pSrcDst *Float32, (C.int)(nLength), Npp32f nParam)).ToError()
}
func CauchyD32fI_Ctx(pSrcDst *Float32, nLength int32, Npp32f nParam, ctx *StreamContext) error{
  return status(C.nppsCauchyD_32f_I_Ctx(pSrcDst *Float32, (C.int)(nLength), Npp32f nParam, ctx.c())).ToError()
}
func CauchyD32fI(pSrcDst *Float32, nLength int32, Npp32f nParam) error{
  return status(C.nppsCauchyD_32f_I(pSrcDst *Float32, (C.int)(nLength), Npp32f nParam)).ToError()
}
func CauchyDD232fICtx(pSrcDst *Float32, Npp32f * pD2FVal, nLength int32, Npp32f nParam, ctx *StreamContext) error{
  return status(C.nppsCauchyDD2_32f_I_Ctx(pSrcDst *Float32, Npp32f * pD2FVal, (C.int)(nLength), Npp32f nParam, ctx.c())).ToError()
}
func CauchyDD232fI(pSrcDst *Float32, Npp32f * pD2FVal, nLength int32, Npp32f nParam) error{
  return status(C.nppsCauchyDD2_32f_I(pSrcDst *Float32, Npp32f * pD2FVal, (C.int)(nLength), Npp32f nParam)).ToError()
}
func AndC8uCtx(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAndC_8u_Ctx(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func AndC8u(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32) error{
  return status(C.nppsAndC_8u(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength))).ToError()
}
func AndC16u_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAndC_16u_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func AndC16u( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32) error{
  return status(C.nppsAndC_16u( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func AndC32u_Ctx( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAndC_32u_Ctx( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func AndC32u( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, nLength int32) error{
  return status(C.nppsAndC_32u( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, (C.int)(nLength))).ToError()
}
func AndC8u_I_Ctx(nValue Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAndC_8u_I_Ctx(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func AndC8u_I(nValue Uint8, pSrcDst *Uint8, nLength int32) error{
  return status(C.nppsAndC_8u_I(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength))).ToError()
}
func AndC16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAndC_16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func AndC16u_I(Npp16u nValue, Npp16u * pSrcDst, nLength int32) error{
  return status(C.nppsAndC_16u_I(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength))).ToError()
}
func AndC32u_I_Ctx(Npp32u nValue, Npp32u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAndC_32u_I_Ctx(Npp32u nValue, Npp32u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func AndC32u_I(Npp32u nValue, Npp32u * pSrcDst, nLength int32) error{
  return status(C.nppsAndC_32u_I(Npp32u nValue, Npp32u * pSrcDst, (C.int)(nLength))).ToError()
}
func And8u_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAnd_8u_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func And8u(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32) error{
  return status(C.nppsAnd_8u(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength))).ToError()
}
func And16u_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAnd_16u_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func And16u( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32) error{
  return status(C.nppsAnd_16u( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func And32u_Ctx( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAnd_32u_Ctx( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func And32u( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, nLength int32) error{
  return status(C.nppsAnd_32u( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, (C.int)(nLength))).ToError()
}
func And8u_I_Ctx(pSrc *Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAnd_8u_I_Ctx(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func And8u_I(pSrc *Uint8, pSrcDst *Uint8, nLength int32) error{
  return status(C.nppsAnd_8u_I(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength))).ToError()
}
func And16u_I_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAnd_16u_I_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func And16u_I( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32) error{
  return status(C.nppsAnd_16u_I( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength))).ToError()
}
func And32u_I_Ctx( Npp32u * pSrc, Npp32u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsAnd_32u_I_Ctx( Npp32u * pSrc, Npp32u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func And32u_I( Npp32u * pSrc, Npp32u * pSrcDst, nLength int32) error{
  return status(C.nppsAnd_32u_I( Npp32u * pSrc, Npp32u * pSrcDst, (C.int)(nLength))).ToError()
}
func OrC8u_Ctx(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOrC_8u_Ctx(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func OrC8u(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32) error{
  return status(C.nppsOrC_8u(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength))).ToError()
}
func OrC16u_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOrC_16u_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func OrC16u( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32) error{
  return status(C.nppsOrC_16u( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func OrC32u_Ctx( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOrC_32u_Ctx( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func OrC32u( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, nLength int32) error{
  return status(C.nppsOrC_32u( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, (C.int)(nLength))).ToError()
}
func OrC8u_I_Ctx(nValue Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOrC_8u_I_Ctx(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func OrC8u_I(nValue Uint8, pSrcDst *Uint8, nLength int32) error{
  return status(C.nppsOrC_8u_I(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength))).ToError()
}
func OrC16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOrC_16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func OrC16u_I(Npp16u nValue, Npp16u * pSrcDst, nLength int32) error{
  return status(C.nppsOrC_16u_I(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength))).ToError()
}
func OrC32u_I_Ctx(Npp32u nValue, Npp32u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOrC_32u_I_Ctx(Npp32u nValue, Npp32u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func OrC32u_I(Npp32u nValue, Npp32u * pSrcDst, nLength int32) error{
  return status(C.nppsOrC_32u_I(Npp32u nValue, Npp32u * pSrcDst, (C.int)(nLength))).ToError()
}
func Or8u_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOr_8u_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func Or8u(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32) error{
  return status(C.nppsOr_8u(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength))).ToError()
}
func Or16u_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOr_16u_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Or16u( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32) error{
  return status(C.nppsOr_16u( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func Or32u_Ctx( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOr_32u_Ctx( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Or32u( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, nLength int32) error{
  return status(C.nppsOr_32u( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, (C.int)(nLength))).ToError()
}
func Or8u_I_Ctx(pSrc *Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOr_8u_I_Ctx(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func Or8u_I(pSrc *Uint8, pSrcDst *Uint8, nLength int32) error{
  return status(C.nppsOr_8u_I(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength))).ToError()
}
func Or16u_I_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOr_16u_I_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Or16u_I( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32) error{
  return status(C.nppsOr_16u_I( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength))).ToError()
}
func Or32u_I_Ctx( Npp32u * pSrc, Npp32u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsOr_32u_I_Ctx( Npp32u * pSrc, Npp32u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Or32u_I( Npp32u * pSrc, Npp32u * pSrcDst, nLength int32) error{
  return status(C.nppsOr_32u_I( Npp32u * pSrc, Npp32u * pSrcDst, (C.int)(nLength))).ToError()
}
func XorC8u_Ctx(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXorC_8u_Ctx(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func XorC8u(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32) error{
  return status(C.nppsXorC_8u(pSrc *Uint8, nValue Uint8, pDst *Uint8, (C.int)(nLength))).ToError()
}
func XorC16u_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXorC_16u_Ctx( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func XorC16u( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, nLength int32) error{
  return status(C.nppsXorC_16u( Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func XorC32u_Ctx( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXorC_32u_Ctx( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func XorC32u( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, nLength int32) error{
  return status(C.nppsXorC_32u( Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, (C.int)(nLength))).ToError()
}
func XorC8u_I_Ctx(nValue Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXorC_8u_I_Ctx(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func XorC8u_I(nValue Uint8, pSrcDst *Uint8, nLength int32) error{
  return status(C.nppsXorC_8u_I(nValue Uint8, pSrcDst *Uint8, (C.int)(nLength))).ToError()
}
func XorC16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXorC_16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func XorC16u_I(Npp16u nValue, Npp16u * pSrcDst, nLength int32) error{
  return status(C.nppsXorC_16u_I(Npp16u nValue, Npp16u * pSrcDst, (C.int)(nLength))).ToError()
}
func XorC32u_I_Ctx(Npp32u nValue, Npp32u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXorC_32u_I_Ctx(Npp32u nValue, Npp32u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func XorC32u_I(Npp32u nValue, Npp32u * pSrcDst, nLength int32) error{
  return status(C.nppsXorC_32u_I(Npp32u nValue, Npp32u * pSrcDst, (C.int)(nLength))).ToError()
}
func Xor8u_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXor_8u_Ctx(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func Xor8u(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, nLength int32) error{
  return status(C.nppsXor_8u(pSrc *Uint81,Npp8u * pSrc2, pDst *Uint8, (C.int)(nLength))).ToError()
}
func Xor16u_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXor_16u_Ctx( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Xor16u( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, nLength int32) error{
  return status(C.nppsXor_16u( Npp16u * pSrc1,Npp16u * pSrc2, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func Xor32u_Ctx( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXor_32u_Ctx( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Xor32u( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, nLength int32) error{
  return status(C.nppsXor_32u( Npp32u * pSrc1,Npp32u * pSrc2, Npp32u * pDst, (C.int)(nLength))).ToError()
}
func Xor8u_I_Ctx(pSrc *Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXor_8u_I_Ctx(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func Xor8u_I(pSrc *Uint8, pSrcDst *Uint8, nLength int32) error{
  return status(C.nppsXor_8u_I(pSrc *Uint8, pSrcDst *Uint8, (C.int)(nLength))).ToError()
}
func Xor16u_I_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXor_16u_I_Ctx( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Xor16u_I( Npp16u * pSrc, Npp16u * pSrcDst, nLength int32) error{
  return status(C.nppsXor_16u_I( Npp16u * pSrc, Npp16u * pSrcDst, (C.int)(nLength))).ToError()
}
func Xor32u_I_Ctx( Npp32u * pSrc, Npp32u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsXor_32u_I_Ctx( Npp32u * pSrc, Npp32u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Xor32u_I( Npp32u * pSrc, Npp32u * pSrcDst, nLength int32) error{
  return status(C.nppsXor_32u_I( Npp32u * pSrc, Npp32u * pSrcDst, (C.int)(nLength))).ToError()
}
func Not8u_Ctx(pSrc *Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsNot_8u_Ctx(pSrc *Uint8, pDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func Not8u(pSrc *Uint8, pDst *Uint8, nLength int32) error{
  return status(C.nppsNot_8u(pSrc *Uint8, pDst *Uint8, (C.int)(nLength))).ToError()
}
func Not16u_Ctx( Npp16u * pSrc, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsNot_16u_Ctx( Npp16u * pSrc, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Not16u( Npp16u * pSrc, Npp16u * pDst, nLength int32) error{
  return status(C.nppsNot_16u( Npp16u * pSrc, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func Not32u_Ctx( Npp32u * pSrc, Npp32u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsNot_32u_Ctx( Npp32u * pSrc, Npp32u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func Not32u( Npp32u * pSrc, Npp32u * pDst, nLength int32) error{
  return status(C.nppsNot_32u( Npp32u * pSrc, Npp32u * pDst, (C.int)(nLength))).ToError()
}
func Not8u_I_Ctx(pSrcDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsNot_8u_I_Ctx(pSrcDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func Not8u_I(pSrcDst *Uint8, nLength int32) error{
  return status(C.nppsNot_8u_I(pSrcDst *Uint8, (C.int)(nLength))).ToError()
}
func Not16u_I_Ctx(Npp16u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsNot_16u_I_Ctx(Npp16u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Not16u_I(Npp16u * pSrcDst, nLength int32) error{
  return status(C.nppsNot_16u_I(Npp16u * pSrcDst, (C.int)(nLength))).ToError()
}
func Not32u_I_Ctx(Npp32u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsNot_32u_I_Ctx(Npp32u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func Not32u_I(Npp32u * pSrcDst, nLength int32) error{
  return status(C.nppsNot_32u_I(Npp32u * pSrcDst, (C.int)(nLength))).ToError()
}
func LShiftC8u_Ctx(pSrc *Uint8, int nValue, pDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLShiftC_8u_Ctx(pSrc *Uint8, int nValue, pDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func LShiftC8u(pSrc *Uint8, int nValue, pDst *Uint8, nLength int32) error{
  return status(C.nppsLShiftC_8u(pSrc *Uint8, int nValue, pDst *Uint8, (C.int)(nLength))).ToError()
}
func LShiftC16u_Ctx( Npp16u * pSrc, int nValue, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLShiftC_16u_Ctx( Npp16u * pSrc, int nValue, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func LShiftC16u( Npp16u * pSrc, int nValue, Npp16u * pDst, nLength int32) error{
  return status(C.nppsLShiftC_16u( Npp16u * pSrc, int nValue, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func LShiftC16s_Ctx( Npp16s * pSrc, int nValue, Npp16s * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLShiftC_16s_Ctx( Npp16s * pSrc, int nValue, Npp16s * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func LShiftC16s( Npp16s * pSrc, int nValue, Npp16s * pDst, nLength int32) error{
  return status(C.nppsLShiftC_16s( Npp16s * pSrc, int nValue, Npp16s * pDst, (C.int)(nLength))).ToError()
}
func LShiftC32u_Ctx( Npp32u * pSrc, int nValue, Npp32u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLShiftC_32u_Ctx( Npp32u * pSrc, int nValue, Npp32u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func LShiftC32u( Npp32u * pSrc, int nValue, Npp32u * pDst, nLength int32) error{
  return status(C.nppsLShiftC_32u( Npp32u * pSrc, int nValue, Npp32u * pDst, (C.int)(nLength))).ToError()
}
func LShiftC32s_Ctx( Npp32s * pSrc, int nValue, Npp32s * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLShiftC_32s_Ctx( Npp32s * pSrc, int nValue, Npp32s * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func LShiftC32s( Npp32s * pSrc, int nValue, Npp32s * pDst, nLength int32) error{
  return status(C.nppsLShiftC_32s( Npp32s * pSrc, int nValue, Npp32s * pDst, (C.int)(nLength))).ToError()
}
func LShiftC8u_I_Ctx(int nValue, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLShiftC_8u_I_Ctx(int nValue, pSrcDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func LShiftC8u_I(int nValue, pSrcDst *Uint8, nLength int32) error{
  return status(C.nppsLShiftC_8u_I(int nValue, pSrcDst *Uint8, (C.int)(nLength))).ToError()
}
func LShiftC16u_I_Ctx(int nValue, Npp16u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLShiftC_16u_I_Ctx(int nValue, Npp16u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func LShiftC16u_I(int nValue, Npp16u * pSrcDst, nLength int32) error{
  return status(C.nppsLShiftC_16u_I(int nValue, Npp16u * pSrcDst, (C.int)(nLength))).ToError()
}
func LShiftC16s_I_Ctx(int nValue, Npp16s * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLShiftC_16s_I_Ctx(int nValue, Npp16s * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func LShiftC16s_I(int nValue, Npp16s * pSrcDst, nLength int32) error{
  return status(C.nppsLShiftC_16s_I(int nValue, Npp16s * pSrcDst, (C.int)(nLength))).ToError()
}
func LShiftC32u_I_Ctx(int nValue, Npp32u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLShiftC_32u_I_Ctx(int nValue, Npp32u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func LShiftC32u_I(int nValue, Npp32u * pSrcDst, nLength int32) error{
  return status(C.nppsLShiftC_32u_I(int nValue, Npp32u * pSrcDst, (C.int)(nLength))).ToError()
}
func LShiftC32s_I_Ctx(int nValue, Npp32s * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsLShiftC_32s_I_Ctx(int nValue, Npp32s * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func LShiftC32s_I(int nValue, Npp32s * pSrcDst, nLength int32) error{
  return status(C.nppsLShiftC_32s_I(int nValue, Npp32s * pSrcDst, (C.int)(nLength))).ToError()
}
func RShiftC8u_Ctx(pSrc *Uint8, int nValue, pDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsRShiftC_8u_Ctx(pSrc *Uint8, int nValue, pDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func RShiftC8u(pSrc *Uint8, int nValue, pDst *Uint8, nLength int32) error{
  return status(C.nppsRShiftC_8u(pSrc *Uint8, int nValue, pDst *Uint8, (C.int)(nLength))).ToError()
}
func RShiftC16u_Ctx( Npp16u * pSrc, int nValue, Npp16u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsRShiftC_16u_Ctx( Npp16u * pSrc, int nValue, Npp16u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func RShiftC16u( Npp16u * pSrc, int nValue, Npp16u * pDst, nLength int32) error{
  return status(C.nppsRShiftC_16u( Npp16u * pSrc, int nValue, Npp16u * pDst, (C.int)(nLength))).ToError()
}
func RShiftC16s_Ctx( Npp16s * pSrc, int nValue, Npp16s * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsRShiftC_16s_Ctx( Npp16s * pSrc, int nValue, Npp16s * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func RShiftC16s( Npp16s * pSrc, int nValue, Npp16s * pDst, nLength int32) error{
  return status(C.nppsRShiftC_16s( Npp16s * pSrc, int nValue, Npp16s * pDst, (C.int)(nLength))).ToError()
}
func RShiftC32u_Ctx( Npp32u * pSrc, int nValue, Npp32u * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsRShiftC_32u_Ctx( Npp32u * pSrc, int nValue, Npp32u * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func RShiftC32u( Npp32u * pSrc, int nValue, Npp32u * pDst, nLength int32) error{
  return status(C.nppsRShiftC_32u( Npp32u * pSrc, int nValue, Npp32u * pDst, (C.int)(nLength))).ToError()
}
func RShiftC32s_Ctx( Npp32s * pSrc, int nValue, Npp32s * pDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsRShiftC_32s_Ctx( Npp32s * pSrc, int nValue, Npp32s * pDst, (C.int)(nLength), ctx.c())).ToError()
}
func RShiftC32s( Npp32s * pSrc, int nValue, Npp32s * pDst, nLength int32) error{
  return status(C.nppsRShiftC_32s( Npp32s * pSrc, int nValue, Npp32s * pDst, (C.int)(nLength))).ToError()
}
func RShiftC8u_I_Ctx(int nValue, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error{
  return status(C.nppsRShiftC_8u_I_Ctx(int nValue, pSrcDst *Uint8, (C.int)(nLength), ctx.c())).ToError()
}
func RShiftC8u_I(int nValue, pSrcDst *Uint8, nLength int32) error{
  return status(C.nppsRShiftC_8u_I(int nValue, pSrcDst *Uint8, (C.int)(nLength))).ToError()
}
func RShiftC16u_I_Ctx(int nValue, Npp16u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsRShiftC_16u_I_Ctx(int nValue, Npp16u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func RShiftC16u_I(int nValue, Npp16u * pSrcDst, nLength int32) error{
  return status(C.nppsRShiftC_16u_I(int nValue, Npp16u * pSrcDst, (C.int)(nLength))).ToError()
}
func RShiftC16s_I_Ctx(int nValue, Npp16s * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsRShiftC_16s_I_Ctx(int nValue, Npp16s * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func RShiftC16s_I(int nValue, Npp16s * pSrcDst, nLength int32) error{
  return status(C.nppsRShiftC_16s_I(int nValue, Npp16s * pSrcDst, (C.int)(nLength))).ToError()
}
func RShiftC32u_I_Ctx(int nValue, Npp32u * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsRShiftC_32u_I_Ctx(int nValue, Npp32u * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func RShiftC32u_I(int nValue, Npp32u * pSrcDst, nLength int32) error{
  return status(C.nppsRShiftC_32u_I(int nValue, Npp32u * pSrcDst, (C.int)(nLength))).ToError()
}
func RShiftC32s_I_Ctx(int nValue, Npp32s * pSrcDst, nLength int32, ctx *StreamContext) error{
  return status(C.nppsRShiftC_32s_I_Ctx(int nValue, Npp32s * pSrcDst, (C.int)(nLength), ctx.c())).ToError()
}
func RShiftC32s_I(int nValue, Npp32s * pSrcDst, nLength int32) error{
  return status(C.nppsRShiftC_32s_I(int nValue, Npp32s * pSrcDst, (C.int)(nLength))).ToError()
}
*/
