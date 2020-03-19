package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//RNNClipMode is a flag for the clipmode for an RNN
type RNNClipMode C.cudnnRNNClipMode_t

func (r RNNClipMode) c() C.cudnnRNNClipMode_t {
	return C.cudnnRNNClipMode_t(r)
}

//None sets r to and returns RNNClipMode(C.CUDNN_RNN_CLIP_NONE)
func (r *RNNClipMode) None() RNNClipMode { *r = RNNClipMode(C.CUDNN_RNN_CLIP_NONE); return *r }

//MinMax sets r to and returns RNNClipMode(C.CUDNN_RNN_CLIP_MINMAX)
func (r *RNNClipMode) MinMax() RNNClipMode { *r = RNNClipMode(C.CUDNN_RNN_CLIP_MINMAX); return *r }

func (r RNNClipMode) String() string {
	var x string
	f := r
	switch r {
	case f.MinMax():
		x = "MinMax"
	case f.None():
		x = "None"
	default:
		x = "Unsupported Flag"
	}
	return "RNNClipMode" + x
}

//SetClip sets the clip mode into descriptor
func (r *RNND) SetClip(h *Handle, mode RNNClipMode, nanprop NANProp, lclip, rclip float64) error {
	return Status(C.cudnnRNNSetClip(h.x, r.descriptor, mode.c(), nanprop.c(), C.double(lclip), C.double(rclip))).error("SetClip")
}

//GetClip returns the clip settings for the descriptor
func (r *RNND) GetClip(h *Handle) (mode RNNClipMode, nanprop NANProp, lclip, rclip float64, err error) {
	var (
		m   C.cudnnRNNClipMode_t
		nan C.cudnnNanPropagation_t
		lt  C.double
		rt  C.double
	)
	err = Status(C.cudnnRNNGetClip(h.x, r.descriptor, &m, &nan, &lt, &rt)).error("SetClip")
	mode = RNNClipMode(m)
	nanprop = NANProp(nan)
	lclip = float64(lt)
	rclip = float64(rt)
	return mode, nanprop, lclip, rclip, err

}
