package npp

/*
#include <npps_conversion_functions.h>
#include <nppdefs.h>
*/
import "C"

//Convert8s16s is in cuda npp documentation
func Convert8s16s(src *Int8, dst *Int16, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_8s16s(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_8s16s_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert8s32f is in cuda npp documentation
func Convert8s32f(src *Int8, dst *Float32, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_8s32f(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_8s32f_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert8u32f is in cuda npp documentation
func Convert8u32f(src *Uint8, dst *Float32, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_8u32f(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_8u32f_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert16s8sSfs is in cuda npp documentation
func Convert16s8sSfs(src *Int16, dst *Int8, length int32, rmode RoundMode, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_16s8s_Sfs(src.cptr(), dst.cptr(), (C.Npp32u)(length), rmode.c(), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_16s8s_Sfs_Ctx(src.cptr(), dst.cptr(), (C.Npp32u)(length), rmode.c(), (C.int)(scalefactor), s.c())).ToError()
}

//Convert16s32s is in cuda npp documentation
func Convert16s32s(src *Int16, dst *Int32, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_16s32s(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_16s32s_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert16s32f is in cuda npp documentation
func Convert16s32f(src *Int16, dst *Float32, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_16s32f(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_16s32f_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert16u32f is in cuda npp documentation
func Convert16u32f(src *Uint16, dst *Float32, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_16u32f(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_16u32f_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert32s16s is in cuda npp documentation
func Convert32s16s(src *Int32, dst *Int16, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32s16s(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_32s16s_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert32s32f is in cuda npp documentation
func Convert32s32f(src *Int32, dst *Float32, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32s32f(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_32s32f_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert32s64f is in cuda npp documentation
func Convert32s64f(src *Int32, dst *Float64, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32s64f(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_32s64f_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert32f64f is in cuda npp documentation
func Convert32f64f(src *Float32, dst *Float64, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32f64f(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_32f64f_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert64s64f is in cuda npp documentation
func Convert64s64f(src *Int64, dst *Float64, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_64s64f(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_64s64f_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert64f32f is in cuda npp documentation
func Convert64f32f(src *Float64, dst *Float32, length int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_64f32f(src.cptr(), dst.cptr(), (C.int)(length))).ToError()
	}
	return status(C.nppsConvert_64f32f_Ctx(src.cptr(), dst.cptr(), (C.int)(length), s.c())).ToError()
}

//Convert16s32fSfs is in cuda npp documentation
func Convert16s32fSfs(src *Int16, dst *Float32, length int32, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_16s32f_Sfs(src.cptr(), dst.cptr(), (C.int)(length), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_16s32f_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), (C.int)(scalefactor), s.c())).ToError()
}

//Convert16s64fSfs is in cuda npp documentation
func Convert16s64fSfs(src *Int16, dst *Float64, length int32, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_16s64f_Sfs(src.cptr(), dst.cptr(), (C.int)(length), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_16s64f_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), (C.int)(scalefactor), s.c())).ToError()
}

//Convert32s16sSfs is in cuda npp documentation
func Convert32s16sSfs(src *Int32, dst *Int16, length int32, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32s16s_Sfs(src.cptr(), dst.cptr(), (C.int)(length), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_32s16s_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), (C.int)(scalefactor), s.c())).ToError()
}

//Convert32s32fSfs is in cuda npp documentation
func Convert32s32fSfs(src *Int32, dst *Float32, length int32, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32s32f_Sfs(src.cptr(), dst.cptr(), (C.int)(length), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_32s32f_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), (C.int)(scalefactor), s.c())).ToError()
}

//Convert32s64fSfs is in cuda npp documentation
func Convert32s64fSfs(src *Int32, dst *Float64, length int32, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32s64f_Sfs(src.cptr(), dst.cptr(), (C.int)(length), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_32s64f_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), (C.int)(scalefactor), s.c())).ToError()
}

//Convert32f8sSfs is in cuda npp documentation
func Convert32f8sSfs(src *Float32, dst *Int8, length int32, rmode RoundMode, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32f8s_Sfs(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_32f8s_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor), s.c())).ToError()
}

//Convert32f8uSfs is in cuda npp documentation
func Convert32f8uSfs(src *Float32, dst *Uint8, length int32, rmode RoundMode, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32f8u_Sfs(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_32f8u_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor), s.c())).ToError()
}

//Convert32f16sSfs is in cuda npp documentation
func Convert32f16sSfs(src *Float32, dst *Int16, length int32, rmode RoundMode, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32f16s_Sfs(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_32f16s_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor), s.c())).ToError()
}

//Convert32f16uSfs is in cuda npp documentation
func Convert32f16uSfs(src *Float32, dst *Uint16, length int32, rmode RoundMode, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32f16u_Sfs(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_32f16u_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor), s.c())).ToError()
}

//Convert32f32sSfs is in cuda npp documentation
func Convert32f32sSfs(src *Float32, dst *Int32, length int32, rmode RoundMode, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_32f32s_Sfs(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_32f32s_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor), s.c())).ToError()
}

//Convert64s32sSfs is in cuda npp documentation
func Convert64s32sSfs(src *Int64, dst *Int32, length int32, rmode RoundMode, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_64s32s_Sfs(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_64s32s_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor), s.c())).ToError()
}

//Convert64f16sSfs is in cuda npp documentation
func Convert64f16sSfs(src *Float64, dst *Int16, length int32, rmode RoundMode, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_64f16s_Sfs(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_64f16s_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor), s.c())).ToError()
}

//Convert64f32sSfs is in cuda npp documentation
func Convert64f32sSfs(src *Float64, dst *Int32, length int32, rmode RoundMode, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_64f32s_Sfs(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_64f32s_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor), s.c())).ToError()
}

//Convert64f64sSfs is in cuda npp documentation
func Convert64f64sSfs(src *Float64, dst *Int64, length int32, rmode RoundMode, scalefactor int32, s *StreamContext) error {
	if s == nil {
		return status(C.nppsConvert_64f64s_Sfs(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor))).ToError()
	}
	return status(C.nppsConvert_64f64s_Sfs_Ctx(src.cptr(), dst.cptr(), (C.int)(length), rmode.c(), (C.int)(scalefactor), s.c())).ToError()
}
