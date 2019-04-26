package npp

/*
#include<npps_arithmetic_and_logical_operations.h>
*/
import "C"

//AddC8uISfs can be found in cuda npp documentation
func AddC8uISfs(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_8u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_8u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC8uSfs can be found in cuda npp documentation
func AddC8uSfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_8u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_8u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC16uISfs can be found in cuda npp documentation
func AddC16uISfs(nValue Uint16, pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_16u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_16u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC16uSfs can be found in cuda npp documentation
func AddC16uSfs(pSrc *Uint16, nValue Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_16u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_16u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC16sISfs can be found in cuda npp documentation
func AddC16sISfs(nValue Int16, pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_16s_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_16s_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC16sSfs can be found in cuda npp documentation
func AddC16sSfs(pSrc *Int16, nValue Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_16s_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_16s_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC16scISfs can be found in cuda npp documentation
func AddC16scISfs(nValue Int16Complex, pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_16sc_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_16sc_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC16scSfs can be found in cuda npp documentation
func AddC16scSfs(pSrc *Int16Complex, nValue Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_16sc_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_16sc_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC32sISfs can be found in cuda npp documentation
func AddC32sISfs(nValue Int32, pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_32s_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_32s_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC32sSfs can be found in cuda npp documentation
func AddC32sSfs(pSrc *Int32, nValue Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_32s_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_32s_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC32scISfs can be found in cuda npp documentation
func AddC32scISfs(nValue Int32Complex, pSrcDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_32sc_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_32sc_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC32scSfs can be found in cuda npp documentation
func AddC32scSfs(pSrc *Int32Complex, nValue Int32Complex, pDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_32sc_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddC_32sc_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddC32fI can be found in cuda npp documentation
func AddC32fI(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_32f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddC_32f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddC32f can be found in cuda npp documentation
func AddC32f(pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_32f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddC_32f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddC32fcI can be found in cuda npp documentation
func AddC32fcI(nValue Float32Complex, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_32fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddC_32fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddC32fc can be found in cuda npp documentation
func AddC32fc(pSrc *Float32Complex, nValue Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_32fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddC_32fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddC64fI can be found in cuda npp documentation
func AddC64fI(nValue Float64, pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_64f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddC_64f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddC64f can be found in cuda npp documentation
func AddC64f(pSrc *Float64, nValue Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_64f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddC_64f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddC64fcI can be found in cuda npp documentation
func AddC64fcI(nValue Float64Complex, pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_64fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddC_64fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddC64fc can be found in cuda npp documentation
func AddC64fc(pSrc *Float64Complex, nValue Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddC_64fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()

	}
	return status(C.nppsAddC_64fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddProductC32f can be found in cuda npp documentation
func AddProductC32f(pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddProductC_32f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddProductC_32f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//MulC8uISfs can be found in cuda npp documentation
func MulC8uISfs(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_8u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_8u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC8uSfs can be found in cuda npp documentation
func MulC8uSfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_8u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_8u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC16uISfs can be found in cuda npp documentation
func MulC16uISfs(nValue Uint16, pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_16u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_16u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC16uSfs can be found in cuda npp documentation
func MulC16uSfs(pSrc *Uint16, nValue Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_16u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_16u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC16sISfs can be found in cuda npp documentation
func MulC16sISfs(nValue Int16, pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_16s_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_16s_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC16sSfs can be found in cuda npp documentation
func MulC16sSfs(pSrc *Int16, nValue Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_16s_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_16s_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC16scISfs can be found in cuda npp documentation
func MulC16scISfs(nValue Int16Complex, pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_16sc_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_16sc_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC16scSfs can be found in cuda npp documentation
func MulC16scSfs(pSrc *Int16Complex, nValue Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_16sc_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_16sc_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC32sISfs can be found in cuda npp documentation
func MulC32sISfs(nValue Int32, pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_32s_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_32s_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC32sSfs can be found in cuda npp documentation
func MulC32sSfs(pSrc *Int32, nValue Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_32s_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_32s_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC32scISfs can be found in cuda npp documentation
func MulC32scISfs(nValue Int32Complex, pSrcDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_32sc_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_32sc_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC32scSfs can be found in cuda npp documentation
func MulC32scSfs(pSrc *Int32Complex, nValue Int32Complex, pDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_32sc_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_32sc_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC32fI can be found in cuda npp documentation
func MulC32fI(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_32f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMulC_32f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//MulC32f can be found in cuda npp documentation
func MulC32f(pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_32f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMulC_32f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//MulCLow32f16s can be found in cuda npp documentation
func MulCLow32f16s(pSrc *Float32, nValue Float32, pDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_Low_32f16s(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMulC_Low_32f16s_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//MulC32f16sSfs can be found in cuda npp documentation
func MulC32f16sSfs(pSrc *Float32, nValue Float32, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_32f16s_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_32f16s_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC32fcI can be found in cuda npp documentation
func MulC32fcI(nValue Float32Complex, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_32fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMulC_32fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//MulC32fc can be found in cuda npp documentation
func MulC32fc(pSrc *Float32Complex, nValue Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_32fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMulC_32fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//MulC64fI can be found in cuda npp documentation
func MulC64fI(nValue Float64, pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_64f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMulC_64f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//MulC64f can be found in cuda npp documentation
func MulC64f(pSrc *Float64, nValue Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_64f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMulC_64f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//MulC64f64sISfs can be found in cuda npp documentation
func MulC64f64sISfs(nValue Float64, pDst *Int64, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_64f64s_ISfs(nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMulC_64f64s_ISfs_Ctx(nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulC64fcI can be found in cuda npp documentation
func MulC64fcI(nValue Float64Complex, pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_64fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMulC_64fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//MulC64fc can be found in cuda npp documentation
func MulC64fc(pSrc *Float64Complex, nValue Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMulC_64fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMulC_64fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubC8uISfs can be found in cuda npp documentation
func SubC8uISfs(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_8u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_8u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC8uSfs can be found in cuda npp documentation
func SubC8uSfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_8u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_8u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC16uISfs can be found in cuda npp documentation
func SubC16uISfs(nValue Uint16, pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_16u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_16u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC16uSfs can be found in cuda npp documentation
func SubC16uSfs(pSrc *Uint16, nValue Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_16u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_16u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC16sISfs can be found in cuda npp documentation
func SubC16sISfs(nValue Int16, pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_16s_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_16s_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC16sSfs can be found in cuda npp documentation
func SubC16sSfs(pSrc *Int16, nValue Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_16s_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_16s_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC16scISfs can be found in cuda npp documentation
func SubC16scISfs(nValue Int16Complex, pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_16sc_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_16sc_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC16scSfs can be found in cuda npp documentation
func SubC16scSfs(pSrc *Int16Complex, nValue Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_16sc_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_16sc_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC32sISfs can be found in cuda npp documentation
func SubC32sISfs(nValue Int32, pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_32s_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_32s_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC32sSfs can be found in cuda npp documentation
func SubC32sSfs(pSrc *Int32, nValue Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_32s_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_32s_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC32scISfs can be found in cuda npp documentation
func SubC32scISfs(nValue Int32Complex, pSrcDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_32sc_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_32sc_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC32scSfs can be found in cuda npp documentation
func SubC32scSfs(pSrc *Int32Complex, nValue Int32Complex, pDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_32sc_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubC_32sc_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubC32fI can be found in cuda npp documentation
func SubC32fI(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_32f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubC_32f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubC32f can be found in cuda npp documentation
func SubC32f(pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_32f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubC_32f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubC32fcI can be found in cuda npp documentation
func SubC32fcI(nValue Float32Complex, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_32fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubC_32fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubC32fc can be found in cuda npp documentation
func SubC32fc(pSrc *Float32Complex, nValue Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_32fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubC_32fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubC64fI can be found in cuda npp documentation
func SubC64fI(nValue Float64, pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_64f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubC_64f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubC64f can be found in cuda npp documentation
func SubC64f(pSrc *Float64, nValue Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_64f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubC_64f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubC64fcI can be found in cuda npp documentation
func SubC64fcI(nValue Float64Complex, pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_64fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubC_64fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubC64fc can be found in cuda npp documentation
func SubC64fc(pSrc *Float64Complex, nValue Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubC_64fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubC_64fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubCRev8uISfs can be found in cuda npp documentation
func SubCRev8uISfs(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_8u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_8u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev8uSfs can be found in cuda npp documentation
func SubCRev8uSfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_8u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_8u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev16uISfs can be found in cuda npp documentation
func SubCRev16uISfs(nValue Uint16, pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_16u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_16u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev16uSfs can be found in cuda npp documentation
func SubCRev16uSfs(pSrc *Uint16, nValue Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_16u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_16u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev16sISfs can be found in cuda npp documentation
func SubCRev16sISfs(nValue Int16, pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_16s_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_16s_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev16sSfs can be found in cuda npp documentation
func SubCRev16sSfs(pSrc *Int16, nValue Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_16s_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_16s_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev16scISfs can be found in cuda npp documentation
func SubCRev16scISfs(nValue Int16Complex, pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_16sc_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_16sc_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev16scSfs can be found in cuda npp documentation
func SubCRev16scSfs(pSrc *Int16Complex, nValue Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_16sc_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_16sc_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev32sISfs can be found in cuda npp documentation
func SubCRev32sISfs(nValue Int32, pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_32s_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_32s_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev32sSfs can be found in cuda npp documentation
func SubCRev32sSfs(pSrc *Int32, nValue Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_32s_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_32s_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev32scISfs can be found in cuda npp documentation
func SubCRev32scISfs(nValue Int32Complex, pSrcDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_32sc_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_32sc_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev32scSfs can be found in cuda npp documentation
func SubCRev32scSfs(pSrc *Int32Complex, nValue Int32Complex, pDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_32sc_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSubCRev_32sc_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SubCRev32fI can be found in cuda npp documentation
func SubCRev32fI(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_32f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubCRev_32f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubCRev32f can be found in cuda npp documentation
func SubCRev32f(pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_32f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubCRev_32f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubCRev32fcI can be found in cuda npp documentation
func SubCRev32fcI(nValue Float32Complex, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_32fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubCRev_32fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubCRev32fc can be found in cuda npp documentation
func SubCRev32fc(pSrc *Float32Complex, nValue Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_32fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubCRev_32fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubCRev64fI can be found in cuda npp documentation
func SubCRev64fI(nValue Float64, pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_64f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubCRev_64f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubCRev64f can be found in cuda npp documentation
func SubCRev64f(pSrc *Float64, nValue Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_64f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubCRev_64f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubCRev64fcI can be found in cuda npp documentation
func SubCRev64fcI(nValue Float64Complex, pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_64fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubCRev_64fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//SubCRev64fc can be found in cuda npp documentation
func SubCRev64fc(pSrc *Float64Complex, nValue Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSubCRev_64fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSubCRev_64fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivC8uISfs can be found in cuda npp documentation
func DivC8uISfs(nValue Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_8u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDivC_8u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivC8uSfs can be found in cuda npp documentation
func DivC8uSfs(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_8u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDivC_8u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivC16uISfs can be found in cuda npp documentation
func DivC16uISfs(nValue Uint16, pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_16u_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDivC_16u_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivC16uSfs can be found in cuda npp documentation
func DivC16uSfs(pSrc *Uint16, nValue Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_16u_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDivC_16u_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivC16sISfs can be found in cuda npp documentation
func DivC16sISfs(nValue Int16, pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_16s_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDivC_16s_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivC16sSfs can be found in cuda npp documentation
func DivC16sSfs(pSrc *Int16, nValue Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_16s_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDivC_16s_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivC16scISfs can be found in cuda npp documentation
func DivC16scISfs(nValue Int16Complex, pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_16sc_ISfs(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDivC_16sc_ISfs_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivC16scSfs can be found in cuda npp documentation
func DivC16scSfs(pSrc *Int16Complex, nValue Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_16sc_Sfs(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDivC_16sc_Sfs_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivC32fI can be found in cuda npp documentation
func DivC32fI(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_32f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_32f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivC32f can be found in cuda npp documentation
func DivC32f(pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_32f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_32f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivC32fcI can be found in cuda npp documentation
func DivC32fcI(nValue Float32Complex, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_32fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_32fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivC32fc can be found in cuda npp documentation
func DivC32fc(pSrc *Float32Complex, nValue Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_32fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_32fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivC64fI can be found in cuda npp documentation
func DivC64fI(nValue Float64, pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_64f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_64f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivC64f can be found in cuda npp documentation
func DivC64f(pSrc *Float64, nValue Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_64f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_64f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivC64fcI can be found in cuda npp documentation
func DivC64fcI(nValue Float64Complex, pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_64fc_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_64fc_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivC64fc can be found in cuda npp documentation
func DivC64fc(pSrc *Float64Complex, nValue Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivC_64fc(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivC_64fc_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivCRev16uI can be found in cuda npp documentation
func DivCRev16uI(nValue Uint16, pSrcDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivCRev_16u_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivCRev_16u_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivCRev16u can be found in cuda npp documentation
func DivCRev16u(pSrc *Uint16, nValue Uint16, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivCRev_16u(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivCRev_16u_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivCRev32fI can be found in cuda npp documentation
func DivCRev32fI(nValue Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivCRev_32f_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivCRev_32f_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivCRev32f can be found in cuda npp documentation
func DivCRev32f(pSrc *Float32, nValue Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDivCRev_32f(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDivCRev_32f_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add16s can be found in cuda npp documentation
func Add16s(pSrc1, pSrc2 *Int16, pDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16s(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_16s_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add16u can be found in cuda npp documentation
func Add16u(pSrc1 *Uint16, pSrc2 *Uint16, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_16u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add32u can be found in cuda npp documentation
func Add32u(pSrc1, pSrc2 *Uint32, pDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_32u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_32u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add32f can be found in cuda npp documentation
func Add32f(pSrc1, pSrc2 *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_32f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_32f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add64f can be found in cuda npp documentation
func Add64f(pSrc1, pSrc2 *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_64f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_64f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add32fc can be found in cuda npp documentation
func Add32fc(pSrc1 *Float32Complex, pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_32fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_32fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add64fc can be found in cuda npp documentation
func Add64fc(pSrc1, pSrc2 *Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_64fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_64fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add8u16u can be found in cuda npp documentation
func Add8u16u(pSrc1, pSrc2 *Uint8, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_8u16u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_8u16u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add16s32f can be found in cuda npp documentation
func Add16s32f(pSrc1, pSrc2 *Int16, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16s32f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_16s32f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add8uSfs can be found in cuda npp documentation
func Add8uSfs(pSrc1, pSrc2 *Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_8u_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_8u_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add16uSfs can be found in cuda npp documentation
func Add16uSfs(pSrc1, pSrc2 *Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16u_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_16u_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add16sSfs can be found in cuda npp documentation
func Add16sSfs(pSrc1, pSrc2 *Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_16s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add32sSfs can be found in cuda npp documentation
func Add32sSfs(pSrc1, pSrc2 *Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_32s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_32s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add64sSfs can be found in cuda npp documentation
func Add64sSfs(pSrc1, pSrc2 *Int64, pDst *Int64, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_64s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_64s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add16scSfs can be found in cuda npp documentation
func Add16scSfs(pSrc1, pSrc2 *Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16sc_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_16sc_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add32scSfs can be found in cuda npp documentation
func Add32scSfs(pSrc1, pSrc2 *Int32Complex, pDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_32sc_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_32sc_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add16sI can be found in cuda npp documentation
func Add16sI(pSrc *Int16, pSrcDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16s_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_16s_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add32fI can be found in cuda npp documentation
func Add32fI(pSrc *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_32f_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_32f_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add64fI can be found in cuda npp documentation
func Add64fI(pSrc *Float64, pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_64f_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_64f_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add32fcI can be found in cuda npp documentation
func Add32fcI(pSrc *Float32Complex, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_32fc_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_32fc_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add64fcI can be found in cuda npp documentation
func Add64fcI(pSrc *Float64Complex, pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_64fc_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_64fc_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add16s32sI can be found in cuda npp documentation
func Add16s32sI(pSrc *Int16, pSrcDst *Int32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16s32s_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAdd_16s32s_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Add8uISfs can be found in cuda npp documentation
func Add8uISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_8u_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_8u_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add16uISfs can be found in cuda npp documentation
func Add16uISfs(pSrc *Uint16, pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16u_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_16u_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add16sISfs can be found in cuda npp documentation
func Add16sISfs(pSrc *Int16, pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16s_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_16s_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add32sI can be found in cuda npp documentation
func Add32sI(pSrc *Int32, pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_32s_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}

	return status(C.nppsAdd_32s_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()

}

//Add16scISfs can be found in cuda npp documentation
func Add16scISfs(pSrc *Int16Complex, pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_16sc_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_16sc_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Add32scISfs can be found in cuda npp documentation
func Add32scISfs(pSrc *Int32Complex, pSrcDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAdd_32sc_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAdd_32sc_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddProduct32f can be found in cuda npp documentation
func AddProduct32f(pSrc1, pSrc2 *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddProduct_32f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddProduct_32f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddProduct64f can be found in cuda npp documentation
func AddProduct64f(pSrc1, pSrc2 *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddProduct_64f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddProduct_64f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddProduct32fc can be found in cuda npp documentation
func AddProduct32fc(pSrc1 *Float32Complex, pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddProduct_32fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddProduct_32fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddProduct64fc can be found in cuda npp documentation
func AddProduct64fc(pSrc1, pSrc2 *Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddProduct_64fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAddProduct_64fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AddProduct16sSfs can be found in cuda npp documentation
func AddProduct16sSfs(pSrc1, pSrc2 *Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddProduct_16s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddProduct_16s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddProduct32sSfs can be found in cuda npp documentation
func AddProduct32sSfs(pSrc1, pSrc2 *Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddProduct_32s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddProduct_32s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//AddProduct16s32sSfs can be found in cuda npp documentation
func AddProduct16s32sSfs(pSrc1, pSrc2 *Int16, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAddProduct_16s32s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsAddProduct_16s32s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul16s can be found in cuda npp documentation
func Mul16s(pSrc1, pSrc2 *Int16, pDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16s(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_16s_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul32f can be found in cuda npp documentation
func Mul32f(pSrc1, pSrc2 *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_32f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul64f can be found in cuda npp documentation
func Mul64f(pSrc1, pSrc2 *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_64f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_64f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul32fc can be found in cuda npp documentation
func Mul32fc(pSrc1 *Float32Complex, pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_32fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul64fc can be found in cuda npp documentation
func Mul64fc(pSrc1, pSrc2 *Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_64fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_64fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul8u16u can be found in cuda npp documentation
func Mul8u16u(pSrc1, pSrc2 *Uint8, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_8u16u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_8u16u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul16s32f can be found in cuda npp documentation
func Mul16s32f(pSrc1, pSrc2 *Int16, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16s32f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_16s32f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul32f32fc can be found in cuda npp documentation
func Mul32f32fc(pSrc1 *Float32, pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32f32fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_32f32fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul8uSfs can be found in cuda npp documentation
func Mul8uSfs(pSrc1, pSrc2 *Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_8u_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_8u_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul16uSfs can be found in cuda npp documentation
func Mul16uSfs(pSrc1, pSrc2 *Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16u_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_16u_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul16sSfs can be found in cuda npp documentation
func Mul16sSfs(pSrc1, pSrc2 *Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_16s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul32sSfs can be found in cuda npp documentation
func Mul32sSfs(pSrc1, pSrc2 *Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_32s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul16scSfs can be found in cuda npp documentation
func Mul16scSfs(pSrc1, pSrc2 *Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16sc_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_16sc_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul32scSfs can be found in cuda npp documentation
func Mul32scSfs(pSrc1, pSrc2 *Int32Complex, pDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32sc_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_32sc_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul16u16sSfs can be found in cuda npp documentation
func Mul16u16sSfs(pSrc1 *Uint16, pSrc2 *Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16u16s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_16u16s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul16s32sSfs can be found in cuda npp documentation
func Mul16s32sSfs(pSrc1, pSrc2 *Int16, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16s32s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_16s32s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul32s32scSfs can be found in cuda npp documentation
func Mul32s32scSfs(pSrc1 *Int32, pSrc2 *Int32Complex, pDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32s32sc_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_32s32sc_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//MulLow32sSfs can be found in cuda npp documentation
func MulLow32sSfs(pSrc1, pSrc2 *Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_Low_32s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_Low_32s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul16sI can be found in cuda npp documentation
func Mul16sI(pSrc *Int16, pSrcDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16s_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_16s_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul32fI can be found in cuda npp documentation
func Mul32fI(pSrc *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32f_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_32f_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul64fI can be found in cuda npp documentation
func Mul64fI(pSrc *Float64, pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_64f_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_64f_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul32fcI can be found in cuda npp documentation
func Mul32fcI(pSrc *Float32Complex, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32fc_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_32fc_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul64fcI can be found in cuda npp documentation
func Mul64fcI(pSrc *Float64Complex, pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_64fc_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_64fc_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul32f32fcI can be found in cuda npp documentation
func Mul32f32fcI(pSrc *Float32, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32f32fc_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_32f32fc_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Mul8uISfs can be found in cuda npp documentation
func Mul8uISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_8u_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_8u_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul16uISfs can be found in cuda npp documentation
func Mul16uISfs(pSrc *Uint16, pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16u_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_16u_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul16sISfs can be found in cuda npp documentation
func Mul16sISfs(pSrc *Int16, pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16s_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_16s_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul32sISfs can be found in cuda npp documentation
func Mul32sISfs(pSrc *Int32, pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32s_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_32s_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul16scISfs can be found in cuda npp documentation
func Mul16scISfs(pSrc *Int16Complex, pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_16sc_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_16sc_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul32scISfs can be found in cuda npp documentation
func Mul32scISfs(pSrc, pSrcDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32sc_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_32sc_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Mul32s32scISfs can be found in cuda npp documentation
func Mul32s32scISfs(pSrc *Int32, pSrcDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32s32sc_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_32s32sc_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub16s can be found in cuda npp documentation
func Sub16s(pSrc1, pSrc2 *Int16, pDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_16s(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_16s_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub32f can be found in cuda npp documentation
func Sub32f(pSrc1, pSrc2 *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_32f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_32f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub64f can be found in cuda npp documentation
func Sub64f(pSrc1, pSrc2 *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_64f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_64f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub32fc can be found in cuda npp documentation
func Sub32fc(pSrc1 *Float32Complex, pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_32fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_32fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub64fc can be found in cuda npp documentation
func Sub64fc(pSrc1, pSrc2 *Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_64fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_64fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub16s32f can be found in cuda npp documentation
func Sub16s32f(pSrc1, pSrc2 *Int16, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_16s32f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_16s32f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub8uSfs can be found in cuda npp documentation
func Sub8uSfs(pSrc1, pSrc2 *Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_8u_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_8u_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub16uSfs can be found in cuda npp documentation
func Sub16uSfs(pSrc1, pSrc2 *Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_16u_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_16u_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub16sSfs can be found in cuda npp documentation
func Sub16sSfs(pSrc1, pSrc2 *Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_16s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_16s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub32sSfs can be found in cuda npp documentation
func Sub32sSfs(pSrc1, pSrc2 *Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_32s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_32s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub16scSfs can be found in cuda npp documentation
func Sub16scSfs(pSrc1, pSrc2 *Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_16sc_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_16sc_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub32scSfs can be found in cuda npp documentation
func Sub32scSfs(pSrc1, pSrc2 *Int32Complex, pDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_32sc_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_32sc_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub16sI can be found in cuda npp documentation
func Sub16sI(pSrc *Int16, pSrcDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_16s_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_16s_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub32fI can be found in cuda npp documentation
func Sub32fI(pSrc *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_32f_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_32f_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub64fI can be found in cuda npp documentation
func Sub64fI(pSrc *Float64, pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_64f_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_64f_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub32fcI can be found in cuda npp documentation
func Sub32fcI(pSrc *Float32Complex, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_32fc_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_32fc_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub64fcI can be found in cuda npp documentation
func Sub64fcI(pSrc *Float64Complex, pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_64fc_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSub_64fc_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sub8uISfs can be found in cuda npp documentation
func Sub8uISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_8u_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_8u_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub16uISfs can be found in cuda npp documentation
func Sub16uISfs(pSrc *Uint16, pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_16u_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_16u_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub16sISfs can be found in cuda npp documentation
func Sub16sISfs(pSrc *Int16, pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_16s_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_16s_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub32sISfs can be found in cuda npp documentation
func Sub32sISfs(pSrc *Int32, pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_32s_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_32s_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub16scISfs can be found in cuda npp documentation
func Sub16scISfs(pSrc *Int16Complex, pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_16sc_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_16sc_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sub32scISfs can be found in cuda npp documentation
func Sub32scISfs(pSrc *Int32Complex, pSrcDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSub_32sc_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSub_32sc_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div8uSfs can be found in cuda npp documentation
func Div8uSfs(pSrc1, pSrc2 *Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_8u_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_8u_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div16uSfs can be found in cuda npp documentation
func Div16uSfs(pSrc1, pSrc2 *Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_16u_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_16u_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div16sSfs can be found in cuda npp documentation
func Div16sSfs(pSrc1, pSrc2 *Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_16s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_16s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div32sSfs can be found in cuda npp documentation
func Div32sSfs(pSrc1, pSrc2 *Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_32s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_32s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div16scSfs can be found in cuda npp documentation
func Div16scSfs(pSrc1, pSrc2 *Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_16sc_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_16sc_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div32s16sSfs can be found in cuda npp documentation
func Div32s16sSfs(pSrc1 *Int16, pSrc2 *Int32, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_32s16s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_32s16s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div32f can be found in cuda npp documentation
func Div32f(pSrc1, pSrc2 *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_32f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDiv_32f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Div64f can be found in cuda npp documentation
func Div64f(pSrc1, pSrc2 *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_64f(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDiv_64f_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Div32fc can be found in cuda npp documentation
func Div32fc(pSrc1 *Float32Complex, pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_32fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDiv_32fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Div64fc can be found in cuda npp documentation
func Div64fc(pSrc1, pSrc2 *Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_64fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDiv_64fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Div8uISfs can be found in cuda npp documentation
func Div8uISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_8u_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_8u_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div16uISfs can be found in cuda npp documentation
func Div16uISfs(pSrc *Uint16, pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_16u_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_16u_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div16sISfs can be found in cuda npp documentation
func Div16sISfs(pSrc *Int16, pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_16s_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_16s_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div16scISfs can be found in cuda npp documentation
func Div16scISfs(pSrc *Int16Complex, pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_16sc_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_16sc_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div32sISfs can be found in cuda npp documentation
func Div32sISfs(pSrc *Int32, pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_32s_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_32s_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Div32fI can be found in cuda npp documentation
func Div32fI(pSrc *Float32, pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_32f_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDiv_32f_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Div64fI can be found in cuda npp documentation
func Div64fI(pSrc *Float64, pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_64f_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDiv_64f_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Div32fcI can be found in cuda npp documentation
func Div32fcI(pSrc *Float32Complex, pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_32fc_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDiv_32fc_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Div64fcI can be found in cuda npp documentation
func Div64fcI(pSrc *Float64Complex, pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_64fc_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsDiv_64fc_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//DivRound8uSfs can be found in cuda npp documentation
func DivRound8uSfs(pSrc1, pSrc2 *Uint8, pDst *Uint8, nLength int32, rmode RoundMode, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_Round_8u_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_Round_8u_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivRound16uSfs can be found in cuda npp documentation
func DivRound16uSfs(pSrc1, pSrc2 *Uint16, pDst *Uint16, nLength int32, rmode RoundMode, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_Round_16u_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_Round_16u_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivRound16sSfs can be found in cuda npp documentation
func DivRound16sSfs(pSrc1, pSrc2 *Int16, pDst *Int16, nLength int32, rmode RoundMode, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_Round_16s_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_Round_16s_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivRound8uISfs can be found in cuda npp documentation
func DivRound8uISfs(pSrc *Uint8, pSrcDst *Uint8, nLength int32, rmode RoundMode, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_Round_8u_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_Round_8u_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivRound16uISfs can be found in cuda npp documentation
func DivRound16uISfs(pSrc *Uint16, pSrcDst *Uint16, nLength int32, rmode RoundMode, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_Round_16u_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_Round_16u_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//DivRound16sISfs can be found in cuda npp documentation
func DivRound16sISfs(pSrc *Int16, pSrcDst *Int16, nLength int32, rmode RoundMode, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_Round_16s_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_Round_16s_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), rmode.c(), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Abs16s can be found in cuda npp documentation
func Abs16s(pSrc *Int16, pDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAbs_16s(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAbs_16s_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Abs32s can be found in cuda npp documentation
func Abs32s(pSrc *Int32, pDst *Int32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAbs_32s(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAbs_32s_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Abs32f can be found in cuda npp documentation
func Abs32f(pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAbs_32f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAbs_32f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Abs64f can be found in cuda npp documentation
func Abs64f(pSrc *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAbs_64f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAbs_64f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Abs16sI can be found in cuda npp documentation
func Abs16sI(pSrcDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAbs_16s_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAbs_16s_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Abs32sI can be found in cuda npp documentation
func Abs32sI(pSrcDst *Int32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAbs_32s_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAbs_32s_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Abs32fI can be found in cuda npp documentation
func Abs32fI(pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAbs_32f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAbs_32f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Abs64fI can be found in cuda npp documentation
func Abs64fI(pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAbs_64f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAbs_64f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqr32f can be found in cuda npp documentation
func Sqr32f(pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_32f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqr_32f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqr64f can be found in cuda npp documentation
func Sqr64f(pSrc *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_64f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqr_64f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqr32fc can be found in cuda npp documentation
func Sqr32fc(pSrc *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_32fc(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqr_32fc_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqr64fc can be found in cuda npp documentation
func Sqr64fc(pSrc *Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_64fc(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqr_64fc_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqr32fI can be found in cuda npp documentation
func Sqr32fI(pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_32f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqr_32f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqr64fI can be found in cuda npp documentation
func Sqr64fI(pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_64f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqr_64f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqr32fcI can be found in cuda npp documentation
func Sqr32fcI(pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_32fc_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqr_32fc_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqr64fcI can be found in cuda npp documentation
func Sqr64fcI(pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_64fc_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqr_64fc_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqr8uSfs can be found in cuda npp documentation
func Sqr8uSfs(pSrc *Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_8u_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqr_8u_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqr16uSfs can be found in cuda npp documentation
func Sqr16uSfs(pSrc *Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_16u_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqr_16u_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqr16sSfs can be found in cuda npp documentation
func Sqr16sSfs(pSrc *Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_16s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqr_16s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqr16scSfs can be found in cuda npp documentation
func Sqr16scSfs(pSrc *Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_16sc_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqr_16sc_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqr8uISfs can be found in cuda npp documentation
func Sqr8uISfs(pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_8u_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqr_8u_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqr16uISfs can be found in cuda npp documentation
func Sqr16uISfs(pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_16u_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqr_16u_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqr16sISfs can be found in cuda npp documentation
func Sqr16sISfs(pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_16s_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqr_16s_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqr16scISfs can be found in cuda npp documentation
func Sqr16scISfs(pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_16sc_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqr_16sc_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt32f can be found in cuda npp documentation
func Sqrt32f(pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_32f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqrt_32f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqrt64f can be found in cuda npp documentation
func Sqrt64f(pSrc *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_64f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqrt_64f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqrt32fc can be found in cuda npp documentation
func Sqrt32fc(pSrc *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_32fc(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqrt_32fc_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqrt64fc can be found in cuda npp documentation
func Sqrt64fc(pSrc *Float64Complex, pDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_64fc(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqrt_64fc_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqrt32fI can be found in cuda npp documentation
func Sqrt32fI(pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_32f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqrt_32f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqrt64fI can be found in cuda npp documentation
func Sqrt64fI(pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_64f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqrt_64f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqrt32fcI can be found in cuda npp documentation
func Sqrt32fcI(pSrcDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_32fc_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqrt_32fc_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqrt64fcI can be found in cuda npp documentation
func Sqrt64fcI(pSrcDst *Float64Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_64fc_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsSqrt_64fc_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Sqrt8uSfs can be found in cuda npp documentation
func Sqrt8uSfs(pSrc *Uint8, pDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_8u_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_8u_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt16uSfs can be found in cuda npp documentation
func Sqrt16uSfs(pSrc *Uint16, pDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_16u_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_16u_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt16sSfs can be found in cuda npp documentation
func Sqrt16sSfs(pSrc *Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_16s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_16s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt16scSfs can be found in cuda npp documentation
func Sqrt16scSfs(pSrc *Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_16sc_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_16sc_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt64sSfs can be found in cuda npp documentation
func Sqrt64sSfs(pSrc *Int64, pDst *Int64, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_64s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_64s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt32s16sSfs can be found in cuda npp documentation
func Sqrt32s16sSfs(pSrc *Int32, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_32s16s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_32s16s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt64s16sSfs can be found in cuda npp documentation
func Sqrt64s16sSfs(pSrc *Int64, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_64s16s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_64s16s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt8uISfs can be found in cuda npp documentation
func Sqrt8uISfs(pSrcDst *Uint8, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_8u_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_8u_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt16uISfs can be found in cuda npp documentation
func Sqrt16uISfs(pSrcDst *Uint16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_16u_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_16u_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt16sISfs can be found in cuda npp documentation
func Sqrt16sISfs(pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_16s_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_16s_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt16scISfs can be found in cuda npp documentation
func Sqrt16scISfs(pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_16sc_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_16sc_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt64sISfs can be found in cuda npp documentation
func Sqrt64sISfs(pSrcDst *Int64, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_64s_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_64s_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Cubrt32f can be found in cuda npp documentation
func Cubrt32f(pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsCubrt_32f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsCubrt_32f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Cubrt32s16sSfs can be found in cuda npp documentation
func Cubrt32s16sSfs(pSrc *Int32, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsCubrt_32s16s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsCubrt_32s16s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Exp32f can be found in cuda npp documentation
func Exp32f(pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_32f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsExp_32f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Exp64f can be found in cuda npp documentation
func Exp64f(pSrc *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_64f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsExp_64f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Exp32f64f can be found in cuda npp documentation
func Exp32f64f(pSrc *Float32, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_32f64f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsExp_32f64f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Exp32fI can be found in cuda npp documentation
func Exp32fI(pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_32f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsExp_32f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Exp64fI can be found in cuda npp documentation
func Exp64fI(pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_64f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsExp_64f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Exp16sSfs can be found in cuda npp documentation
func Exp16sSfs(pSrc *Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_16s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsExp_16s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Exp32sSfs can be found in cuda npp documentation
func Exp32sSfs(pSrc *Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_32s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsExp_32s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Exp64sSfs can be found in cuda npp documentation
func Exp64sSfs(pSrc *Int64, pDst *Int64, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_64s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsExp_64s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Exp16sISfs can be found in cuda npp documentation
func Exp16sISfs(pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_16s_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsExp_16s_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Exp32sISfs can be found in cuda npp documentation
func Exp32sISfs(pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_32s_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsExp_32s_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Exp64sISfs can be found in cuda npp documentation
func Exp64sISfs(pSrcDst *Int64, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsExp_64s_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsExp_64s_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Ln32f can be found in cuda npp documentation
func Ln32f(pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLn_32f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLn_32f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Ln64f can be found in cuda npp documentation
func Ln64f(pSrc *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLn_64f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLn_64f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Ln64f32f can be found in cuda npp documentation
func Ln64f32f(pSrc *Float64, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLn_64f32f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLn_64f32f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Ln32fI can be found in cuda npp documentation
func Ln32fI(pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLn_32f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLn_32f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Ln64fI can be found in cuda npp documentation
func Ln64fI(pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLn_64f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLn_64f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Ln16sSfs can be found in cuda npp documentation
func Ln16sSfs(pSrc *Int16, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLn_16s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsLn_16s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Ln32sSfs can be found in cuda npp documentation
func Ln32sSfs(pSrc *Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLn_32s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsLn_32s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Ln32s16sSfs can be found in cuda npp documentation
func Ln32s16sSfs(pSrc *Int32, pDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLn_32s16s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsLn_32s16s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Ln16sISfs can be found in cuda npp documentation
func Ln16sISfs(pSrcDst *Int16, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLn_16s_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsLn_16s_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Ln32sISfs can be found in cuda npp documentation
func Ln32sISfs(pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLn_32s_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsLn_32s_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//TenLogTen32sSfs can be found in cuda npp documentation
func TenLogTen32sSfs(pSrc *Int32, pDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.npps10Log10_32s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.npps10Log10_32s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//TenLogTen32sISfs can be found in cuda npp documentation
func TenLogTen32sISfs(pSrcDst *Int32, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.npps10Log10_32s_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.npps10Log10_32s_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//SumLnGetBufferSize32f can be found in cuda npp documentation
func SumLnGetBufferSize32f(nLength int32, ctx *StreamContext) (hpBufferSize int32, err error) {
	if ctx == nil {
		err = status(C.nppsSumLnGetBufferSize_32f((C.int)(nLength), (*C.int)(&hpBufferSize))).ToError()
		return hpBufferSize, err
	}
	err = status(C.nppsSumLnGetBufferSize_32f_Ctx((C.int)(nLength), (*C.int)(&hpBufferSize), ctx.c())).ToError()
	return hpBufferSize, err

}

//SumLn32f can be found in cuda npp documentation
func SumLn32f(pSrc *Float32, nLength int32, pDst *Float32, pDeviceBuffer *Uint8, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSumLn_32f(pSrc.cptr(), (C.int)(nLength), pDst.cptr(), pDeviceBuffer.cptr())).ToError()
	}
	return status(C.nppsSumLn_32f_Ctx(pSrc.cptr(), (C.int)(nLength), pDst.cptr(), pDeviceBuffer.cptr(), ctx.c())).ToError()
}

//SumLnGetBufferSize64f can be found in cuda npp documentation
func SumLnGetBufferSize64f(nLength int32, ctx *StreamContext) (hpBufferSize int32, err error) {
	if ctx == nil {
		err = status(C.nppsSumLnGetBufferSize_64f((C.int)(nLength), (*C.int)(&hpBufferSize))).ToError()
		return hpBufferSize, err
	}
	err = status(C.nppsSumLnGetBufferSize_64f_Ctx((C.int)(nLength), (*C.int)(&hpBufferSize), ctx.c())).ToError()
	return hpBufferSize, err

}

//SumLn64f can be found in cuda npp documentation
func SumLn64f(pSrc *Float64, nLength int32, pDst *Float64, pDeviceBuffer *Uint8, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSumLn_64f(pSrc.cptr(), (C.int)(nLength), pDst.cptr(), pDeviceBuffer.cptr())).ToError()
	}
	return status(C.nppsSumLn_64f_Ctx(pSrc.cptr(), (C.int)(nLength), pDst.cptr(), pDeviceBuffer.cptr(), ctx.c())).ToError()
}

//SumLnGetBufferSize32f64f can be found in cuda npp documentation
func SumLnGetBufferSize32f64f(nLength int32, ctx *StreamContext) (hpBufferSize int32, err error) {
	if ctx == nil {
		err = status(C.nppsSumLnGetBufferSize_32f64f((C.int)(nLength), (*C.int)(&hpBufferSize))).ToError()
		return hpBufferSize, err
	}
	err = status(C.nppsSumLnGetBufferSize_32f64f_Ctx((C.int)(nLength), (*C.int)(&hpBufferSize), ctx.c())).ToError()
	return hpBufferSize, err
}

//SumLn32f64f can be found in cuda npp documentation
func SumLn32f64f(pSrc *Float32, nLength int32, pDst *Float64, pDeviceBuffer *Uint8, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSumLn_32f64f(pSrc.cptr(), (C.int)(nLength), pDst.cptr(), pDeviceBuffer.cptr())).ToError()
	}
	return status(C.nppsSumLn_32f64f_Ctx(pSrc.cptr(), (C.int)(nLength), pDst.cptr(), pDeviceBuffer.cptr(), ctx.c())).ToError()
}

//SumLnGetBufferSize16s32f can be found in cuda npp documentation
func SumLnGetBufferSize16s32f(nLength int32, ctx *StreamContext) (hpBufferSize int32, err error) {
	if ctx == nil {
		err = status(C.nppsSumLnGetBufferSize_16s32f((C.int)(nLength), (*C.int)(&hpBufferSize))).ToError()
		return hpBufferSize, err
	}
	err = status(C.nppsSumLnGetBufferSize_16s32f_Ctx((C.int)(nLength), (*C.int)(&hpBufferSize), ctx.c())).ToError()
	return hpBufferSize, err
}

//SumLn16s32f can be found in cuda npp documentation
func SumLn16s32f(pSrc *Int16, nLength int32, pDst *Float32, pDeviceBuffer *Uint8, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSumLn_16s32f(pSrc.cptr(), (C.int)(nLength), pDst.cptr(), pDeviceBuffer.cptr())).ToError()
	}
	return status(C.nppsSumLn_16s32f_Ctx(pSrc.cptr(), (C.int)(nLength), pDst.cptr(), pDeviceBuffer.cptr(), ctx.c())).ToError()
}

//Arctan32f can be found in cuda npp documentation
func Arctan32f(pSrc *Float32, pDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsArctan_32f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsArctan_32f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Arctan64f can be found in cuda npp documentation
func Arctan64f(pSrc *Float64, pDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsArctan_64f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsArctan_64f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Arctan32fI can be found in cuda npp documentation
func Arctan32fI(pSrcDst *Float32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsArctan_32f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsArctan_32f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Arctan64fI can be found in cuda npp documentation
func Arctan64fI(pSrcDst *Float64, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsArctan_64f_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsArctan_64f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Normalize32f can be found in cuda npp documentation
func Normalize32f(pSrc *Float32, pDst *Float32, nLength int32, vSub, vDiv float32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNormalize_32f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.Npp32f)(vSub), (C.Npp32f)(vDiv))).ToError()
	}
	return status(C.nppsNormalize_32f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.Npp32f)(vSub), (C.Npp32f)(vDiv), ctx.c())).ToError()
}

//Normalize32fc can be found in cuda npp documentation
func Normalize32fc(pSrc *Float32Complex, pDst *Float32Complex, nLength int32, vSub Float32Complex, vDiv float32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNormalize_32fc(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.Npp32f)(vDiv))).ToError()
	}
	return status(C.nppsNormalize_32fc_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.Npp32f)(vDiv), ctx.c())).ToError()
}

//Normalize64f can be found in cuda npp documentation
func Normalize64f(pSrc *Float64, pDst *Float64, nLength int32, vSub, vDiv float64, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNormalize_64f(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.Npp64f)(vSub), (C.Npp64f)(vDiv))).ToError()
	}
	return status(C.nppsNormalize_64f_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.Npp64f)(vSub), (C.Npp64f)(vDiv), ctx.c())).ToError()
}

//Normalize64fc can be found in cuda npp documentation
func Normalize64fc(pSrc *Float64Complex, pDst *Float64Complex, nLength int32, vSub Float64Complex, vDiv float64, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNormalize_64fc(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.Npp64f)(vDiv))).ToError()
	}
	return status(C.nppsNormalize_64fc_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.Npp64f)(vDiv), ctx.c())).ToError()
}

//Normalize16sSfs can be found in cuda npp documentation
func Normalize16sSfs(pSrc *Int16, pDst *Int16, nLength int32, vSub Int16, vDiv int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNormalize_16s_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.int)(vDiv), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsNormalize_16s_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.int)(vDiv), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Normalize16scSfs can be found in cuda npp documentation
func Normalize16scSfs(pSrc *Int16Complex, pDst *Int16Complex, nLength int32, vSub Int16Complex, vDiv int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNormalize_16sc_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.int)(vDiv), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsNormalize_16sc_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.int)(vDiv), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Cauchy32fI can be found in cuda npp documentation
func Cauchy32fI(pSrcDst *Float32, nLength int32, nParam float32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsCauchy_32f_I(pSrcDst.cptr(), (C.int)(nLength), (C.Npp32f)(nParam))).ToError()
	}
	return status(C.nppsCauchy_32f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.Npp32f)(nParam), ctx.c())).ToError()
}

//CauchyD32fI can be found in cuda npp documentation
func CauchyD32fI(pSrcDst *Float32, nLength int32, nParam float32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsCauchyD_32f_I(pSrcDst.cptr(), (C.int)(nLength), (C.Npp32f)(nParam))).ToError()
	}
	return status(C.nppsCauchyD_32f_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.Npp32f)(nParam), ctx.c())).ToError()
}

//CauchyDD232fI can be found in cuda npp documentation
func CauchyDD232fI(pSrcDst *Float32, pD2FVal *Float32, nLength int32, nParam float32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsCauchyDD2_32f_I(pSrcDst.cptr(), pD2FVal.cptr(), (C.int)(nLength), (C.Npp32f)(nParam))).ToError()
	}
	return status(C.nppsCauchyDD2_32f_I_Ctx(pSrcDst.cptr(), pD2FVal.cptr(), (C.int)(nLength), (C.Npp32f)(nParam), ctx.c())).ToError()
}

//AndC8u can be found in cuda npp documentation
func AndC8u(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAndC_8u(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAndC_8u_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AndC16u can be found in cuda npp documentation
func AndC16u(pSrc *Uint16, nValue Uint16, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAndC_16u(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAndC_16u_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AndC32u can be found in cuda npp documentation
func AndC32u(pSrc *Uint32, nValue Uint32, pDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAndC_32u(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAndC_32u_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AndC8uI can be found in cuda npp documentation
func AndC8uI(nValue Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAndC_8u_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAndC_8u_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AndC16uI can be found in cuda npp documentation
func AndC16uI(nValue Uint16, pSrcDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAndC_16u_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAndC_16u_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//AndC32uI can be found in cuda npp documentation
func AndC32uI(nValue Uint32, pSrcDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAndC_32u_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAndC_32u_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//And8u can be found in cuda npp documentation
func And8u(pSrc1, pSrc2 *Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAnd_8u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAnd_8u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//And16u can be found in cuda npp documentation
func And16u(pSrc1, pSrc2 *Uint16, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAnd_16u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAnd_16u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//And32u can be found in cuda npp documentation
func And32u(pSrc1, pSrc2 *Uint32, pDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAnd_32u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAnd_32u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//And8uI can be found in cuda npp documentation
func And8uI(pSrc *Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAnd_8u_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAnd_8u_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//And16uI can be found in cuda npp documentation
func And16uI(pSrc *Uint16, pSrcDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAnd_16u_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAnd_16u_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//And32uI can be found in cuda npp documentation
func And32uI(pSrc *Uint32, pSrcDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsAnd_32u_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsAnd_32u_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//OrC8u can be found in cuda npp documentation
func OrC8u(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOrC_8u(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOrC_8u_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//OrC16u can be found in cuda npp documentation
func OrC16u(pSrc *Uint16, nValue Uint16, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOrC_16u(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOrC_16u_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//OrC32u can be found in cuda npp documentation
func OrC32u(pSrc *Uint32, nValue Uint32, pDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOrC_32u(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOrC_32u_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//OrC8uI can be found in cuda npp documentation
func OrC8uI(nValue Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOrC_8u_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOrC_8u_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//OrC16uI can be found in cuda npp documentation
func OrC16uI(nValue Uint16, pSrcDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOrC_16u_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOrC_16u_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//OrC32uI can be found in cuda npp documentation
func OrC32uI(nValue Uint32, pSrcDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOrC_32u_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOrC_32u_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Or8u can be found in cuda npp documentation
func Or8u(pSrc1, pSrc2 *Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOr_8u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOr_8u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Or16u can be found in cuda npp documentation
func Or16u(pSrc1, pSrc2 *Uint16, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOr_16u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOr_16u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Or32u can be found in cuda npp documentation
func Or32u(pSrc1, pSrc2 *Uint32, pDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOr_32u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOr_32u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Or8uI can be found in cuda npp documentation
func Or8uI(pSrc *Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOr_8u_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOr_8u_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Or16uI can be found in cuda npp documentation
func Or16uI(pSrc *Uint16, pSrcDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOr_16u_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOr_16u_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Or32uI can be found in cuda npp documentation
func Or32uI(pSrc *Uint32, pSrcDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsOr_32u_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsOr_32u_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//XorC8u can be found in cuda npp documentation
func XorC8u(pSrc *Uint8, nValue Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXorC_8u(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXorC_8u_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//XorC16u can be found in cuda npp documentation
func XorC16u(pSrc *Uint16, nValue Uint16, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXorC_16u(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXorC_16u_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//XorC32u can be found in cuda npp documentation
func XorC32u(pSrc *Uint32, nValue Uint32, pDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXorC_32u(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXorC_32u_Ctx(pSrc.cptr(), nValue.c(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//XorC8uI can be found in cuda npp documentation
func XorC8uI(nValue Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXorC_8u_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXorC_8u_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//XorC16uI can be found in cuda npp documentation
func XorC16uI(nValue Uint16, pSrcDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXorC_16u_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXorC_16u_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//XorC32uI can be found in cuda npp documentation
func XorC32uI(nValue Uint32, pSrcDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXorC_32u_I(nValue.c(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXorC_32u_I_Ctx(nValue.c(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Xor8u can be found in cuda npp documentation
func Xor8u(pSrc1, pSrc2 *Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXor_8u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXor_8u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Xor16u can be found in cuda npp documentation
func Xor16u(pSrc1, pSrc2 *Uint16, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXor_16u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXor_16u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Xor32u can be found in cuda npp documentation
func Xor32u(pSrc1, pSrc2 *Uint32, pDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXor_32u(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXor_32u_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Xor8uI can be found in cuda npp documentation
func Xor8uI(pSrc *Uint8, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXor_8u_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXor_8u_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Xor16uI can be found in cuda npp documentation
func Xor16uI(pSrc *Uint16, pSrcDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXor_16u_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXor_16u_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Xor32uI can be found in cuda npp documentation
func Xor32uI(pSrc *Uint32, pSrcDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsXor_32u_I(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsXor_32u_I_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Not8u can be found in cuda npp documentation
func Not8u(pSrc *Uint8, pDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNot_8u(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsNot_8u_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Not16u can be found in cuda npp documentation
func Not16u(pSrc *Uint16, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNot_16u(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsNot_16u_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Not32u can be found in cuda npp documentation
func Not32u(pSrc *Uint32, pDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNot_32u(pSrc.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsNot_32u_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Not8uI can be found in cuda npp documentation
func Not8uI(pSrcDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNot_8u_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsNot_8u_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Not16uI can be found in cuda npp documentation
func Not16uI(pSrcDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNot_16u_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsNot_16u_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//Not32uI can be found in cuda npp documentation
func Not32uI(pSrcDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNot_32u_I(pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsNot_32u_I_Ctx(pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//LShiftC8u can be found in cuda npp documentation
func LShiftC8u(pSrc *Uint8, nValue int32, pDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLShiftC_8u(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLShiftC_8u_Ctx(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//LShiftC16u can be found in cuda npp documentation
func LShiftC16u(pSrc *Uint16, nValue int32, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLShiftC_16u(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLShiftC_16u_Ctx(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//LShiftC16s can be found in cuda npp documentation
func LShiftC16s(pSrc *Int16, nValue int32, pDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLShiftC_16s(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLShiftC_16s_Ctx(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//LShiftC32u can be found in cuda npp documentation
func LShiftC32u(pSrc *Uint32, nValue int32, pDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLShiftC_32u(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLShiftC_32u_Ctx(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//LShiftC32s can be found in cuda npp documentation
func LShiftC32s(pSrc *Int32, nValue int32, pDst *Int32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLShiftC_32s(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLShiftC_32s_Ctx(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//LShiftC8uI can be found in cuda npp documentation
func LShiftC8uI(nValue int32, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLShiftC_8u_I((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLShiftC_8u_I_Ctx((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//LShiftC16uI can be found in cuda npp documentation
func LShiftC16uI(nValue int32, pSrcDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLShiftC_16u_I((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLShiftC_16u_I_Ctx((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//LShiftC16sI can be found in cuda npp documentation
func LShiftC16sI(nValue int32, pSrcDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLShiftC_16s_I((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLShiftC_16s_I_Ctx((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//LShiftC32uI can be found in cuda npp documentation
func LShiftC32uI(nValue int32, pSrcDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLShiftC_32u_I((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLShiftC_32u_I_Ctx((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//LShiftC32sI can be found in cuda npp documentation
func LShiftC32sI(nValue int32, pSrcDst *Int32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsLShiftC_32s_I((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsLShiftC_32s_I_Ctx((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//RShiftC8u can be found in cuda npp documentation
func RShiftC8u(pSrc *Uint8, nValue int32, pDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsRShiftC_8u(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsRShiftC_8u_Ctx(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//RShiftC16u can be found in cuda npp documentation
func RShiftC16u(pSrc *Uint16, nValue int32, pDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsRShiftC_16u(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsRShiftC_16u_Ctx(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//RShiftC16s can be found in cuda npp documentation
func RShiftC16s(pSrc *Int16, nValue int32, pDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsRShiftC_16s(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsRShiftC_16s_Ctx(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//RShiftC32u can be found in cuda npp documentation
func RShiftC32u(pSrc *Uint32, nValue int32, pDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsRShiftC_32u(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsRShiftC_32u_Ctx(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//RShiftC32s can be found in cuda npp documentation
func RShiftC32s(pSrc *Int32, nValue int32, pDst *Int32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsRShiftC_32s(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsRShiftC_32s_Ctx(pSrc.cptr(), (C.int)(nValue), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//RShiftC8uI can be found in cuda npp documentation
func RShiftC8uI(nValue int32, pSrcDst *Uint8, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsRShiftC_8u_I((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsRShiftC_8u_I_Ctx((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//RShiftC16uI can be found in cuda npp documentation
func RShiftC16uI(nValue int32, pSrcDst *Uint16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsRShiftC_16u_I((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsRShiftC_16u_I_Ctx((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//RShiftC16sI can be found in cuda npp documentation
func RShiftC16sI(nValue int32, pSrcDst *Int16, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsRShiftC_16s_I((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsRShiftC_16s_I_Ctx((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//RShiftC32uI can be found in cuda npp documentation
func RShiftC32uI(nValue int32, pSrcDst *Uint32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsRShiftC_32u_I((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsRShiftC_32u_I_Ctx((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}

//RShiftC32sI can be found in cuda npp documentation
func RShiftC32sI(nValue int32, pSrcDst *Int32, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsRShiftC_32s_I((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsRShiftC_32s_I_Ctx((C.int)(nValue), pSrcDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
}
