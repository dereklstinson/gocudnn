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

//None returns the none flag  for clip mode
func (r RNNClipMode) None() RNNClipMode {
	return RNNClipMode(C.CUDNN_RNN_CLIP_NONE)
}

//MinMax returns the minmaxflag for clip mode
func (r RNNClipMode) MinMax() RNNClipMode {
	return RNNClipMode(C.CUDNN_RNN_CLIP_MINMAX)
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
