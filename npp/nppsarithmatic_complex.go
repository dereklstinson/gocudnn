package npp

/*
#include<npps_arithmetic_and_logical_operations.h>
*/
import "C"

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

//Mul32f32fc can be found in cuda npp documentation
func Mul32f32fc(pSrc1 *Float32, pSrc2 *Float32Complex, pDst *Float32Complex, nLength int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32f32fc(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength))).ToError()
	}
	return status(C.nppsMul_32f32fc_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), ctx.c())).ToError()
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

//Mul32s32scSfs can be found in cuda npp documentation
func Mul32s32scSfs(pSrc1 *Int32, pSrc2 *Int32Complex, pDst *Int32Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsMul_32s32sc_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsMul_32s32sc_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
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

//Div16scSfs can be found in cuda npp documentation
func Div16scSfs(pSrc1, pSrc2 *Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_16sc_Sfs(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_16sc_Sfs_Ctx(pSrc1.cptr(), pSrc2.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
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

//Div16scISfs can be found in cuda npp documentation
func Div16scISfs(pSrc *Int16Complex, pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsDiv_16sc_ISfs(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsDiv_16sc_ISfs_Ctx(pSrc.cptr(), pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
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

//Sqr16scSfs can be found in cuda npp documentation
func Sqr16scSfs(pSrc *Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_16sc_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqr_16sc_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqr16scISfs can be found in cuda npp documentation
func Sqr16scISfs(pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqr_16sc_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqr_16sc_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
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

//Sqrt16scSfs can be found in cuda npp documentation
func Sqrt16scSfs(pSrc *Int16Complex, pDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_16sc_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_16sc_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Sqrt16scISfs can be found in cuda npp documentation
func Sqrt16scISfs(pSrcDst *Int16Complex, nLength int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsSqrt_16sc_ISfs(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsSqrt_16sc_ISfs_Ctx(pSrcDst.cptr(), (C.int)(nLength), (C.int)(nScaleFactor), ctx.c())).ToError()
}

//Normalize32fc can be found in cuda npp documentation
func Normalize32fc(pSrc *Float32Complex, pDst *Float32Complex, nLength int32, vSub Float32Complex, vDiv float32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNormalize_32fc(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.Npp32f)(vDiv))).ToError()
	}
	return status(C.nppsNormalize_32fc_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.Npp32f)(vDiv), ctx.c())).ToError()
}

//Normalize64fc can be found in cuda npp documentation
func Normalize64fc(pSrc *Float64Complex, pDst *Float64Complex, nLength int32, vSub Float64Complex, vDiv float64, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNormalize_64fc(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.Npp64f)(vDiv))).ToError()
	}
	return status(C.nppsNormalize_64fc_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.Npp64f)(vDiv), ctx.c())).ToError()
}

//Normalize16scSfs can be found in cuda npp documentation
func Normalize16scSfs(pSrc *Int16Complex, pDst *Int16Complex, nLength int32, vSub Int16Complex, vDiv int32, nScaleFactor int32, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppsNormalize_16sc_Sfs(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.int)(vDiv), (C.int)(nScaleFactor))).ToError()
	}
	return status(C.nppsNormalize_16sc_Sfs_Ctx(pSrc.cptr(), pDst.cptr(), (C.int)(nLength), vSub.c(), (C.int)(vDiv), (C.int)(nScaleFactor), ctx.c())).ToError()
}
