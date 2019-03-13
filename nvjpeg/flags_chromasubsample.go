package nvjpeg

/*
#include <nvjpeg.h>

*/
import "C"

/*
ChromaSubsampling returned by getImageInfo identifies image chroma subsampling stored inside JPEG input stream.
In the case of NVJPEG_CSS_GRAY only 1 luminance channel is encoded in JPEG input stream.
Otherwise both chroma planes are present.

Initial release support: 4:4:4, 4:2:0, 4:2:2, Grayscale.

Flags can be changed by using methods. switches
*/
type ChromaSubsampling C.nvjpegChromaSubsampling_t

func (ch ChromaSubsampling) c() C.nvjpegChromaSubsampling_t {
	return C.nvjpegChromaSubsampling_t(ch)
}

//String returns a human readable string of the flag.  Mostly useful for debugging.
func (ch ChromaSubsampling) String() string {
	var flg ChromaSubsampling
	switch ch {
	case flg.CSS444():
		return "NVJPEG_CSS_444"
	case flg.CSS422():
		return "NVJPEG_CSS_422"
	case flg.CSS420():
		return "NVJPEG_CSS_420"
	case flg.CSS411():
		return "NVJPEG_CSS_411"
	case flg.CSS410():
		return "NVJPEG_CSS_410"
	case flg.CSSGRAY():
		return "NVJPEG_CSS_GRAY"
	case flg.CSSUNKNOWN():
		return "NVJPEG_CSS_UNKNOWN"
	}
	return "error"
}

//CSS444 - 4:4:4 changes the underlying value of type to ChromaSubsampling(C.NVJPEG_CSS_444) and returns it
func (ch *ChromaSubsampling) CSS444() ChromaSubsampling {
	*ch = ChromaSubsampling(C.NVJPEG_CSS_444)
	return *ch
}

//CSS422 - 4:4:2 changes the underlying value of type to ChromaSubsampling(C.NVJPEG_CSS_422) and returns it
func (ch *ChromaSubsampling) CSS422() ChromaSubsampling {
	*ch = ChromaSubsampling(C.NVJPEG_CSS_422)
	return *ch
}

//CSS420 - 4:4:0 changes the underlying value of type to ChromaSubsampling(C.NVJPEG_CSS_420) and returns it
func (ch *ChromaSubsampling) CSS420() ChromaSubsampling {
	*ch = ChromaSubsampling(C.NVJPEG_CSS_420)
	return *ch
}

//CSS411 - 4:1:1 changes the underlying value of type to ChromaSubsampling(C.NVJPEG_CSS_411) and returns it
func (ch *ChromaSubsampling) CSS411() ChromaSubsampling {
	*ch = ChromaSubsampling(C.NVJPEG_CSS_411)
	return *ch
}

//CSS410 - 4:1:0 changes the underlying value of type to ChromaSubsampling(C.NVJPEG_CSS_420) and returns it
func (ch *ChromaSubsampling) CSS410() ChromaSubsampling {
	*ch = ChromaSubsampling(C.NVJPEG_CSS_410)
	return *ch
}

//CSSGRAY - grayscale changes the underlying value of type to ChromaSubsampling(C.NVJPEG_CSS_GRAY) and returns it
func (ch *ChromaSubsampling) CSSGRAY() ChromaSubsampling {
	*ch = ChromaSubsampling(C.NVJPEG_CSS_GRAY)
	return *ch
}

//CSSUNKNOWN - changes the underlying value of type to ChromaSubsampling(C.NVJPEG_CSS_UNKNOWN) and returns it
func (ch *ChromaSubsampling) CSSUNKNOWN() ChromaSubsampling {
	*ch = ChromaSubsampling(C.NVJPEG_CSS_UNKNOWN)
	return *ch
}
