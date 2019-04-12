package npp

/*
//#include <nppi_computer_vision.h>
#include <nppdefs.h>
#include <nppi_filtering_functions.h>
*/
import "C"
import "errors"

//FilterCannyBorderGetBufferSize can be found in npp documentation.
func FilterCannyBorderGetBufferSize(oROI Size) (int32, error) {
	var size C.int
	err := status(C.nppiFilterCannyBorderGetBufferSize(oROI.c(), &size)).ToError()
	return (int32)(size), err
}

//FilterCannyBorder8uC1R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func FilterCannyBorder8uC1R(
	pSrc *Uint8,
	nSrcStep int32,
	oSrcSize Size,
	oSrcOffset Point,
	pDst *Uint8,
	nDstStep int32,
	oROI Size,
	efilterType DifferentialKernel,
	emSize MaskSize,
	nLowThreshold Int16,
	nHighThreshold Int16,
	eNorm Norm,
	eBorderType BorderType,
	pDeviceBuffer *Uint8,
	ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppiFilterCannyBorder_8u_C1R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), pDst.cptr(), (C.int)(nDstStep), oROI.c(), efilterType.c(), emSize.c(), nLowThreshold.c(), nHighThreshold.c(), eNorm.c(), eBorderType.c(), pDeviceBuffer.cptr())).ToError()
	}

	return status(C.nppiFilterCannyBorder_8u_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), pDst.cptr(), (C.int)(nDstStep), oROI.c(), efilterType.c(), emSize.c(), nLowThreshold.c(), nHighThreshold.c(), eNorm.c(), eBorderType.c(), pDeviceBuffer.cptr(), ctx.s)).ToError()
}

//FilterHarrisCornersBorderGetBufferSize can be found in npp documentation.
func FilterHarrisCornersBorderGetBufferSize(oROI Size) (size int32, err error) {
	var hpBufferSize (C.int)
	err = status(C.nppiFilterHarrisCornersBorderGetBufferSize(oROI.c(), &hpBufferSize)).ToError()
	return (int32)(hpBufferSize), err
}

//FilterHarrisCornersBorder8u32fC1R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func FilterHarrisCornersBorder8u32fC1R(
	pSrc *Uint8,
	nSrcStep int32,
	oSrcSize Size,
	oSrcOffset Point,
	pDst *Float32,
	nDstStep int32,
	oROI Size,
	efilterType DifferentialKernel,
	emSize MaskSize,
	eAvgWindowSize MaskSize,
	nK Float32,
	nScale Float32,
	eBorderType BorderType,
	pDeviceBuffer *Uint8,
	ctx *StreamContext) error {
	if ctx == nil {

		return status(C.nppiFilterHarrisCornersBorder_8u32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), pDst.cptr(), (C.int)(nDstStep), oROI.c(), efilterType.c(), emSize.c(), eAvgWindowSize.c(), nK.c(), nScale.c(), eBorderType.c(), pDeviceBuffer.cptr())).ToError()
	}
	return status(C.nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), pDst.cptr(), (C.int)(nDstStep), oROI.c(), efilterType.c(), emSize.c(), eAvgWindowSize.c(), nK.c(), nScale.c(), eBorderType.c(), pDeviceBuffer.cptr(), ctx.s)).ToError()
}

//FilterHoughLineGetBufferSize can be found in npp documentation.
func FilterHoughLineGetBufferSize(
	oROI Size,
	nDelta PolarPoint,
	nMaxLineCount int32) (size int32, err error) {
	var hpBufferSize (C.int)
	err = status(C.nppiFilterHoughLineGetBufferSize(oROI.c(), nDelta.c(), (C.int)(nMaxLineCount), &hpBufferSize)).ToError()
	return (int32)(hpBufferSize), err
}

//FilterHoughLine8u32fC1R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func FilterHoughLine8u32fC1R(
	pSrc *Uint8,
	nSrcStep int32,
	oROI Size,
	nDelta PolarPoint,
	nThreshold int32,
	nMaxLineCount int32,
	pDeviceBuffer *Uint8,
	ctx *StreamContext) (pDeviceLines []PolarPoint, err error) {

	pDeviceLines = make([]PolarPoint, nMaxLineCount)
	var pDeviceLineCount C.int
	if ctx != nil {
		err = status(C.nppiFilterHoughLine_8u32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oROI.c(), nDelta.c(), (C.int)(nThreshold), pDeviceLines[0].cptr(), (C.int)(nMaxLineCount), &pDeviceLineCount, pDeviceBuffer.cptr(), ctx.s)).ToError()

	} else {

		err = status(C.nppiFilterHoughLine_8u32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oROI.c(), nDelta.c(), (C.int)(nThreshold), pDeviceLines[0].cptr(), (C.int)(nMaxLineCount), &pDeviceLineCount, pDeviceBuffer.cptr())).ToError()
	}
	return pDeviceLines[:pDeviceLineCount], err
}

//FilterHoughLineRegion8u32fC1R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func FilterHoughLineRegion8u32fC1R(
	pSrc *Uint8,
	nSrcStep int32,
	oROI Size,
	nDelta PolarPoint,
	nThreshold int32,
	oDstROI []PolarPoint,
	nMaxLineCount int32,
	pDeviceBuffer *Uint8,
	ctx *StreamContext) (pDeviceLines []PolarPoint, err error) {

	var pDeviceLineCount C.int
	pDeviceLines = make([]PolarPoint, nMaxLineCount)
	if len(oDstROI) != 2 {
		return nil, errors.New("length of oDstROI needs to be 2")
	}
	if ctx != nil {
		err = status(C.nppiFilterHoughLineRegion_8u32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oROI.c(), nDelta.c(), (C.int)(nThreshold), pDeviceLines[0].cptr(), oDstROI[0].cptr(), (C.int)(nMaxLineCount), &pDeviceLineCount, pDeviceBuffer.cptr(), ctx.s)).ToError()
	} else {
		err = status(C.nppiFilterHoughLineRegion_8u32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oROI.c(), nDelta.c(), (C.int)(nThreshold), pDeviceLines[0].cptr(), oDstROI[0].cptr(), (C.int)(nMaxLineCount), &pDeviceLineCount, pDeviceBuffer.cptr())).ToError()
	}
	return pDeviceLines[:pDeviceLineCount], err
}

//HistogramOfGradientsBorderGetBufferSize can be found in npp documentation.
func HistogramOfGradientsBorderGetBufferSize(ocfg HOGConfig, hpLocations []Point, oROI Size) (hpBufferSize int32, err error) {
	nLocations := (C.int)(len(hpLocations))
	err = status(C.nppiHistogramOfGradientsBorderGetBufferSize(ocfg.c(), hpLocations[0].cptr(), nLocations, oROI.c(), (*C.int)(&hpBufferSize))).ToError()
	return hpBufferSize, err
}

//HistogramOfGradientsBorderGetDescriptorsSize can be found in npp documentation.
func HistogramOfGradientsBorderGetDescriptorsSize(ocfg HOGConfig, numhpLocations int32) (hpDescriptorsSize int32, err error) {
	err = status(C.nppiHistogramOfGradientsBorderGetDescriptorsSize(ocfg.c(), (C.int)(numhpLocations), (*C.int)(&hpDescriptorsSize))).ToError()
	return hpDescriptorsSize, err
}

//HistogramOfGradientsBorder8u32fC1R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func HistogramOfGradientsBorder8u32fC1R(
	pSrc *Uint8,
	nSrcStep int32,
	oSrcSize Size,
	oSrcOffset Point,
	hpLocations []Point,
	pDstWindowDescriptorBuffer *Float32,
	oROI Size,
	ocfg HOGConfig,
	pScratchBuffer *Uint8,
	eBorderType BorderType,
	ctx *StreamContext) error {

	nLocations := (C.int)(len(hpLocations))
	if ctx != nil {
		return status(C.nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c(), ctx.s)).ToError()
	}
	return status(C.nppiHistogramOfGradientsBorder_8u32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c())).ToError()
}

//HistogramOfGradientsBorder8u32fC3R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func HistogramOfGradientsBorder8u32fC3R(
	pSrc *Uint8,
	nSrcStep int32,
	oSrcSize Size,
	oSrcOffset Point,
	hpLocations []Point,
	pDstWindowDescriptorBuffer *Float32,
	oROI Size,
	ocfg HOGConfig,
	pScratchBuffer *Uint8,
	eBorderType BorderType,
	ctx *StreamContext) error {

	nLocations := (C.int)(len(hpLocations))
	if ctx != nil {
		return status(C.nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c(), ctx.s)).ToError()
	}
	return status(C.nppiHistogramOfGradientsBorder_8u32f_C3R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c())).ToError()
}

//HistogramOfGradientsBorder16u32fC1R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func HistogramOfGradientsBorder16u32fC1R(
	pSrc *Uint16,
	nSrcStep int32,
	oSrcSize Size,
	oSrcOffset Point,
	hpLocations []Point,
	pDstWindowDescriptorBuffer *Float32,
	oROI Size,
	ocfg HOGConfig,
	pScratchBuffer *Uint8,
	eBorderType BorderType,
	ctx *StreamContext) error {

	nLocations := (C.int)(len(hpLocations))
	if ctx != nil {
		return status(C.nppiHistogramOfGradientsBorder_16u32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c(), ctx.s)).ToError()
	}
	return status(C.nppiHistogramOfGradientsBorder_16u32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c())).ToError()
}

//HistogramOfGradientsBorder16u32fC3R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func HistogramOfGradientsBorder16u32fC3R(
	pSrc *Uint16,
	nSrcStep int32,
	oSrcSize Size,
	oSrcOffset Point,
	hpLocations []Point,
	pDstWindowDescriptorBuffer *Float32,
	oROI Size,
	ocfg HOGConfig,
	pScratchBuffer *Uint8,
	eBorderType BorderType,
	ctx *StreamContext) error {

	nLocations := (C.int)(len(hpLocations))
	if ctx != nil {
		return status(C.nppiHistogramOfGradientsBorder_16u32f_C3R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c(), ctx.s)).ToError()
	}
	return status(C.nppiHistogramOfGradientsBorder_16u32f_C3R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c())).ToError()
}

//HistogramOfGradientsBorder16s32fC1R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func HistogramOfGradientsBorder16s32fC1R(
	pSrc *Int16,
	nSrcStep int32,
	oSrcSize Size,
	oSrcOffset Point,
	hpLocations []Point,
	pDstWindowDescriptorBuffer *Float32,
	oROI Size,
	ocfg HOGConfig,
	pScratchBuffer *Uint8,
	eBorderType BorderType,
	ctx *StreamContext) error {

	nLocations := (C.int)(len(hpLocations))
	if ctx != nil {
		return status(C.nppiHistogramOfGradientsBorder_16s32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c(), ctx.s)).ToError()
	}
	return status(C.nppiHistogramOfGradientsBorder_16s32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c())).ToError()
}

//HistogramOfGradientsBorder16s32fC3R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func HistogramOfGradientsBorder16s32fC3R(
	pSrc *Int16,
	nSrcStep int32,
	oSrcSize Size,
	oSrcOffset Point,
	hpLocations []Point,
	pDstWindowDescriptorBuffer *Float32,
	oROI Size,
	ocfg HOGConfig,
	pScratchBuffer *Uint8,
	eBorderType BorderType,
	ctx *StreamContext) error {

	nLocations := (C.int)(len(hpLocations))
	if ctx != nil {
		return status(C.nppiHistogramOfGradientsBorder_16s32f_C3R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c(), ctx.s)).ToError()
	}
	return status(C.nppiHistogramOfGradientsBorder_16s32f_C3R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c())).ToError()
}

//HistogramOfGradientsBorder32fC1R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func HistogramOfGradientsBorder32fC1R(
	pSrc *Float32,
	nSrcStep int32,
	oSrcSize Size,
	oSrcOffset Point,
	hpLocations []Point,
	pDstWindowDescriptorBuffer *Float32,
	oROI Size,
	ocfg HOGConfig,
	pScratchBuffer *Uint8,
	eBorderType BorderType,
	ctx *StreamContext) error {

	nLocations := (C.int)(len(hpLocations))
	if ctx != nil {
		return status(C.nppiHistogramOfGradientsBorder_32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c(), ctx.s)).ToError()
	}
	return status(C.nppiHistogramOfGradientsBorder_32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c())).ToError()
}

//HistogramOfGradientsBorder32fC3R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func HistogramOfGradientsBorder32fC3R(
	pSrc *Float32,
	nSrcStep int32,
	oSrcSize Size,
	oSrcOffset Point,
	hpLocations []Point,
	pDstWindowDescriptorBuffer *Float32,
	oROI Size,
	ocfg HOGConfig,
	pScratchBuffer *Uint8,
	eBorderType BorderType,
	ctx *StreamContext) error {

	nLocations := (C.int)(len(hpLocations))
	if ctx != nil {
		return status(C.nppiHistogramOfGradientsBorder_32f_C3R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c(), ctx.s)).ToError()
	}
	return status(C.nppiHistogramOfGradientsBorder_32f_C3R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), hpLocations[0].cptr(), nLocations, pDstWindowDescriptorBuffer.cptr(), oROI.c(), ocfg.c(), pScratchBuffer.cptr(), eBorderType.c())).ToError()
}

//LabelMarkersGetBufferSize8uC1R can be found in npp documentation.
func LabelMarkersGetBufferSize8uC1R(oROI Size) (hpBufferSize int32, err error) {
	err = status(C.nppiLabelMarkersGetBufferSize_8u_C1R(oROI.c(), (*C.int)(&hpBufferSize))).ToError()
	return hpBufferSize, err
}

//LabelMarkersGetBufferSize8u32uC1R can be found in npp documentation.
func LabelMarkersGetBufferSize8u32uC1R(oROI Size) (hpBufferSize int32, err error) {
	err = status(C.nppiLabelMarkersGetBufferSize_8u32u_C1R(oROI.c(), (*C.int)(&hpBufferSize))).ToError()
	return hpBufferSize, err
}

//LabelMarkersGetBufferSize16uC1R can be found in npp documentation.
func LabelMarkersGetBufferSize16uC1R(oROI Size) (hpBufferSize int32, err error) {
	err = status(C.nppiLabelMarkersGetBufferSize_16u_C1R(oROI.c(), (*C.int)(&hpBufferSize))).ToError()
	return hpBufferSize, err
}

//LabelMarkers8uC1IR can be found in npp documentation. If ctx is nil it will run the non ctx function.
func LabelMarkers8uC1IR(
	pSrcDst *Uint8,
	nSrcDstStep int32,
	oROI Size,
	nMinVal Uint8,
	eNorm Norm,
	pBuffer *Uint8,
	ctx *StreamContext) (pNumber int32, err error) {
	if ctx != nil {
		err = status(C.nppiLabelMarkers_8u_C1IR_Ctx(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), nMinVal.c(), eNorm.c(), (*C.int)(&pNumber), pBuffer.cptr(), ctx.s)).ToError()
	} else {
		err = status(C.nppiLabelMarkers_8u_C1IR(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), nMinVal.c(), eNorm.c(), (*C.int)(&pNumber), pBuffer.cptr())).ToError()
	}
	return pNumber, err
}

//LabelMarkers8u32uC1R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func LabelMarkers8u32uC1R(
	pSrc *Uint8,
	nSrcStep int32,
	pDst *Uint32,
	nDstStep int32,
	oROI Size,
	nMinVal Uint8,
	eNorm Norm,
	pBuffer *Uint8,
	ctx *StreamContext) (pNumber int32, err error) {
	if ctx != nil {
		err = status(C.nppiLabelMarkers_8u32u_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), pDst.cptr(), (C.int)(nDstStep), oROI.c(), nMinVal.c(), eNorm.c(), (*C.int)(&pNumber), pBuffer.cptr(), ctx.s)).ToError()
	} else {

		err = status(C.nppiLabelMarkers_8u32u_C1R(pSrc.cptr(), (C.int)(nSrcStep), pDst.cptr(), (C.int)(nDstStep), oROI.c(), nMinVal.c(), eNorm.c(), (*C.int)(&pNumber), pBuffer.cptr())).ToError()
	}
	return pNumber, err
}

//LabelMarkers16uC1IR can be found in npp documentation. If ctx is nil it will run the non ctx function.
func LabelMarkers16uC1IR(
	pSrcDst *Uint16,
	nSrcDstStep int32,
	oROI Size,
	nMinVal Uint16,
	eNorm Norm,
	pBuffer *Uint8,
	ctx *StreamContext) (pNumber int32, err error) {
	if ctx != nil {
		err = status(C.nppiLabelMarkers_16u_C1IR_Ctx(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), nMinVal.c(), eNorm.c(), (*C.int)(&pNumber), pBuffer.cptr(), ctx.s)).ToError()
	} else {

		err = status(C.nppiLabelMarkers_16u_C1IR(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), nMinVal.c(), eNorm.c(), (*C.int)(&pNumber), pBuffer.cptr())).ToError()
	}
	return pNumber, err
}

//CompressMarkerLabelsGetBufferSize8uC1R can be found in npp documentation.
func CompressMarkerLabelsGetBufferSize8uC1R(nStartingNumber int32) (hpBufferSize int32, err error) {
	err = status(C.nppiCompressMarkerLabelsGetBufferSize_8u_C1R((C.int)(nStartingNumber), (*C.int)(&hpBufferSize))).ToError()
	return hpBufferSize, err
}

//CompressMarkerLabelsGetBufferSize32u8uC1R can be found in npp documentation.
func CompressMarkerLabelsGetBufferSize32u8uC1R(nStartingNumber int32) (hpBufferSize int32, err error) {
	err = status(C.nppiCompressMarkerLabelsGetBufferSize_32u8u_C1R((C.int)(nStartingNumber), (*C.int)(&hpBufferSize))).ToError()
	return hpBufferSize, err
}

//CompressMarkerLabelsGetBufferSize16uC1R can be found in npp documentation.
func CompressMarkerLabelsGetBufferSize16uC1R(nStartingNumber int32) (hpBufferSize int32, err error) {
	err = status(C.nppiCompressMarkerLabelsGetBufferSize_16u_C1R((C.int)(nStartingNumber), (*C.int)(&hpBufferSize))).ToError()
	return hpBufferSize, err
}

//CompressMarkerLabelsGetBufferSize32uC1R can be found in npp documentation.
func CompressMarkerLabelsGetBufferSize32uC1R(nStartingNumber int32) (hpBufferSize int32, err error) {
	err = status(C.nppiCompressMarkerLabelsGetBufferSize_32u_C1R((C.int)(nStartingNumber), (*C.int)(&hpBufferSize))).ToError()
	return hpBufferSize, err
}

//CompressMarkerLabels8uC1IR can be found in npp documentation. If ctx is nil it will run the non ctx function.
func CompressMarkerLabels8uC1IR(
	pSrcDst *Uint8,
	nSrcDstStep int32,
	oROI Size,
	nStartingNumber int32,
	pBuffer *Uint8,
	ctx *StreamContext) (pNewNumber int32, err error) {
	if ctx != nil {
		err = status(C.nppiCompressMarkerLabels_8u_C1IR_Ctx(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), (C.int)(nStartingNumber), (*C.int)(&pNewNumber), pBuffer.cptr(), ctx.s)).ToError()
	} else {
		err = status(C.nppiCompressMarkerLabels_8u_C1IR(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), (C.int)(nStartingNumber), (*C.int)(&pNewNumber), pBuffer.cptr())).ToError()
	}
	return pNewNumber, err
}

//CompressMarkerLabels32u8uC1R can be found in npp documentation. If ctx is nil it will run the non ctx function.
func CompressMarkerLabels32u8uC1R(
	pSrc *Uint32,
	nSrcStep int32,
	pDst *Uint8,
	nDstStep int32,
	oROI Size,
	nStartingNumber int32,
	pBuffer *Uint8,
	ctx *StreamContext) (pNewNumber int32, err error) {
	if ctx != nil {
		err = status(C.nppiCompressMarkerLabels_32u8u_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), pDst.cptr(), (C.int)(nDstStep), oROI.c(), (C.int)(nStartingNumber), (*C.int)(&pNewNumber), pBuffer.cptr(), ctx.s)).ToError()
	} else {
		err = status(C.nppiCompressMarkerLabels_32u8u_C1R(pSrc.cptr(), (C.int)(nSrcStep), pDst.cptr(), (C.int)(nDstStep), oROI.c(), (C.int)(nStartingNumber), (*C.int)(&pNewNumber), pBuffer.cptr())).ToError()
	}
	return pNewNumber, err
}

//CompressMarkerLabels16uC1IR can be found in npp documentation. If ctx is nil it will run the non ctx function.
func CompressMarkerLabels16uC1IR(
	pSrcDst *Uint16,
	nSrcDstStep int32,
	oROI Size,
	nStartingNumber int32,
	pBuffer *Uint8,
	ctx *StreamContext) (pNewNumber int32, err error) {

	if ctx != nil {
		err = status(C.nppiCompressMarkerLabels_16u_C1IR_Ctx(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), (C.int)(nStartingNumber), (*C.int)(&pNewNumber), pBuffer.cptr(), ctx.s)).ToError()
	} else {
		err = status(C.nppiCompressMarkerLabels_16u_C1IR(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), (C.int)(nStartingNumber), (*C.int)(&pNewNumber), pBuffer.cptr())).ToError()
	}
	return pNewNumber, err
}

//CompressMarkerLabels32uC1IR can be found in npp documentation. If ctx is nil it will run the non ctx function.
func CompressMarkerLabels32uC1IR(
	pSrcDst *Uint32,
	nSrcDstStep int32,
	oROI Size,
	nStartingNumber int32,
	pBuffer *Uint8,
	ctx *StreamContext) (pNewNumber int32, err error) {
	if ctx != nil {
		err = status(C.nppiCompressMarkerLabels_32u_C1IR_Ctx(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), (C.int)(nStartingNumber), (*C.int)(&pNewNumber), pBuffer.cptr(), ctx.s)).ToError()
	} else {
		err = status(C.nppiCompressMarkerLabels_32u_C1IR(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), (C.int)(nStartingNumber), (*C.int)(&pNewNumber), pBuffer.cptr())).ToError()
	}
	return pNewNumber, err
}

//BoundSegments8uC1IR can be found in npp documentation. If ctx is nil it will run the non ctx function.
func BoundSegments8uC1IR(
	pSrcDst *Uint8,
	nSrcDstStep int32,
	oROI Size,
	nBorderVal Uint8,
	ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppiBoundSegments_8u_C1IR_Ctx(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), nBorderVal.c(), ctx.s)).ToError()
	}
	return status(C.nppiBoundSegments_8u_C1IR(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), nBorderVal.c())).ToError()
}

//BoundSegments16uC1IR can be found in npp documentation. If ctx is nil it will run the non ctx function.
func BoundSegments16uC1IR(
	pSrcDst *Uint16,
	nSrcDstStep int32,
	oROI Size,
	nBorderVal Uint16,
	ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppiBoundSegments_16u_C1IR_Ctx(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), nBorderVal.c(), ctx.s)).ToError()
	}
	return status(C.nppiBoundSegments_16u_C1IR(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), nBorderVal.c())).ToError()
}

//BoundSegments32uC1IR can be found in npp documentation. If ctx is nil it will run the non ctx function.
func BoundSegments32uC1IR(
	pSrcDst *Uint32,
	nSrcDstStep int32,
	oROI Size,
	nBorderVal Uint32,
	ctx *StreamContext) error {
	if ctx != nil {
		return status(C.nppiBoundSegments_32u_C1IR_Ctx(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), nBorderVal.c(), ctx.s)).ToError()
	}

	return status(C.nppiBoundSegments_32u_C1IR(pSrcDst.cptr(), (C.int)(nSrcDstStep), oROI.c(), nBorderVal.c())).ToError()
}
