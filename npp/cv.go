package npp

/*
#include <nppi_computer_vision.h>
#include <nppdefs.h>
#include <nppi_filtering_functions.h>
*/
import "C"

/*
func FilterCannyBorderGetBufferSize(oROI Size) (int32, error) {
	var size C.int
	err := status(C.nppiFilterCannyBorderGetBufferSize(oROI.c(), &size)).ToError()
	return (int32)(size), err
}

func FilterCannyBorder_8u_C1R(pSrc *Uint8, nSrcStep int32, oSrcSize Size, oSrcOffset Point,
	pDst *Uint8, nDstStep int32, oROI Size, efilterType DifferentialKernel,
	emSize MaskSize, nLowThreshold Int16, nHighThreshold Int16, eNorm Norm,
	eBorderType BorderType, pDeviceBuffer *Uint8, ctx *StreamContext) error {
	if ctx == nil {
		return status(C.nppiFilterCannyBorder_8u_C1R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), pDst.cptr(), (C.int)(nDstStep), oROI.c(), efilterType.c(), emSize.c(), nLowThreshold.c(), nHighThreshold.c(), eNorm.c(), eBorderType.c(), pDeviceBuffer.cptr())).ToError()
	}

	return status(C.nppiFilterCannyBorder_8u_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), pDst.cptr(), (C.int)(nDstStep), oROI.c(), efilterType.c(), emSize.c(), nLowThreshold.c(), nHighThreshold.c(), eNorm.c(), eBorderType.c(), pDeviceBuffer.cptr(), ctx.s)).ToError()
}

func FilterHarrisCornersBorderGetBufferSize(oROI Size) (size int32, err error) {
	var hpBufferSize (C.int)
	err = status(C.nppiFilterHarrisCornersBorderGetBufferSize(oROI.c(), &hpBufferSize)).ToError()
	return (int32)(hpBufferSize), err
}
func FilterHarrisCornersBorder_8u32f_C1R(pSrc *Uint8, nSrcStep int32, oSrcSize Size, oSrcOffset Point, pDst *Float32, nDstStep int32, oROI Size, efilterType DifferentialKernel, emSize MaskSize, eAvgWindowSize MaskSize, nK, nScale Float32, eBorderType BorderType, pDeviceBuffer *Uint8, ctx *StreamContext) error {
	if ctx == nil {

		return status(C.nppiFilterHarrisCornersBorder_8u32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), pDst.cptr(), (C.int)(nDstStep), oROI.c(), efilterType.c(), emSize.c(), eAvgWindowSize.c(), nk.c(), nScale.c(), eBorderType.c(), pDeviceBuffer.cptr())).ToError()
	}
	return status(C.nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), pDst.cptr(), (C.int)(nDstStep), oROI.c(), efilterType.c(), emSize.c(), eAvgWindowSize.c(), nK.c(), nScale.c(), eBorderType.c(), pDeviceBuffer.cptr(), ctx.s)).ToError()
}

func FilterHoughLineGetBufferSize(oROI Size, nDelta PointPolar, nMaxLineCount int32) (size int32, err error) {
	var hpBufferSize (C.int)
	err = status(C.nppiFilterHoughLineGetBufferSize(oROI.c(), nDelta.c(), (C.int)(nMaxLineCount), &hpBufferSize)).ToError()
	return (int32)(hpBufferSize), err
}

func FilterHoughLine_8u32f_C1R(pSrc *Uint8, nSrcStep int32, oROI Size, nDelta PointPolar, nThreshold int32, pDeviceLines *PointPolar, nMaxLineCount int32, int *pDeviceLineCount, pDeviceBuffer *Uint8, ctx *StreamContext) (linecount int32, err error) {
	var pDeviceLineCount C.int
	if ctx != nil {
		err = status(C.nppiFilterHoughLine_8u32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oROI.c(), nDelta.c(), (C.int)(nThreshold), pDeviceLines.cptr(), (C.int)(nMaxLineCount), &pDeviceLineCount, pDeviceBuffer.cptr(), ctx.s)).ToError()

	} else {

		err = status(C.nppiFilterHoughLine_8u32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oROI.c(), nDelta.c(), (C.int)(nThreshold), pDeviceLines.cptr(), (C.int)(nMaxLineCount), pDeviceLineCount, pDeviceBuffer.cptr())).ToError()
	}
	return (int32)(pDeviceLineCount), err
}

func FilterHoughLineRegion_8u32f_C1R(pSrc *Uint8, nSrcStep int32, oROI Size, nDelta PointPolar, nThreshold int32, pDeviceLines *PointPolar, oDstROI [2]PointPolar, nMaxLineCount int32, int *pDeviceLineCount, pDeviceBuffer *Uint8, ctx *StreamContext) (linecount int32, err error) {
	var pDeviceLineCount C.int
	//NppPointPolar oDstROI[2]
	if ctx != nil {
		err = status(C.nppiFilterHoughLineRegion_8u32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oROI.c(), nDelta.c(), (C.int)(nThreshold), pDeviceLines.cptr(), oDstROI[0].c(), (C.int)(nMaxLineCount), &pDeviceLineCount, pDeviceBuffer.cptr(), ctx.s)).ToError()
	} else {
		err = status(C.nppiFilterHoughLineRegion_8u32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oROI.c(), nDelta.c(), (C.int)(nThreshold), pDeviceLines.cptr(), oDstROI[0].c(), (C.int)(nMaxLineCount), &pDeviceLineCount, pDeviceBuffer.cptr())).ToError()
	}
	return (int32)(pDeviceLineCount), err
}

/*
func  HistogramOfGradientsBorderGetBufferSize(NppiHOGConfig oHOGConfig, NppiPoint * hpLocations, int nLocations, oROI Size, int * hpBufferSize)error{
return  status(C.nppiHistogramOfGradientsBorderGetBufferSize(NppiHOGConfig oHOGConfig, NppiPoint * hpLocations, int nLocations, oROI.c(), int * hpBufferSize)).ToError()}

func  HistogramOfGradientsBorderGetDescriptorsSize(NppiHOGConfig oHOGConfig, int nLocations, int * hpDescriptorsSize)error{
return  status(C.nppiHistogramOfGradientsBorderGetDescriptorsSize(NppiHOGConfig oHOGConfig, int nLocations, int * hpDescriptorsSize)).ToError()}

func  HistogramOfGradientsBorder_8u32f_C1R(pSrc *Uint8, nSrcStep int32, oSrcSize Size, oSrcOffset Point, NppiPoint * hpLocations, int nLocations, pDst *Float32WindowDescriptorBuffer, oROI Size, NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType BorderType, ctx *StreamContext)error{
	if ctx!=nil{
return  status(C.nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c(), ctx.s)).ToError()
}
return  status(C.nppiHistogramOfGradientsBorder_8u32f_C1R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c())).ToError()}

func  HistogramOfGradientsBorder_8u32f_C3R(pSrc *Uint8, nSrcStep int32, oSrcSize Size, oSrcOffset Point, NppiPoint * hpLocations, int nLocations, pDst *Float32WindowDescriptorBuffer, oROI Size, NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType BorderType, ctx *StreamContext)error{
if ctx!=nil{return  status(C.nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c(), ctx.s)).ToError()}
return  status(C.nppiHistogramOfGradientsBorder_8u32f_C3R(pSrc.cptr(), (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c())).ToError()}

func  HistogramOfGradientsBorder_16u32f_C1R(Npp16u * pSrc, nSrcStep int32, oSrcSize Size, oSrcOffset Point, NppiPoint * hpLocations, int nLocations, pDst *Float32WindowDescriptorBuffer, oROI Size, NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType BorderType, ctx *StreamContext)error{
if ctx!=nil{return  status(C.nppiHistogramOfGradientsBorder_16u32f_C1R_Ctx(Npp16u * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c(), ctx.s)).ToError()}
return  status(C.nppiHistogramOfGradientsBorder_16u32f_C1R(Npp16u * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c())).ToError()}

func  HistogramOfGradientsBorder_16u32f_C3R(Npp16u * pSrc, nSrcStep int32, oSrcSize Size, oSrcOffset Point, NppiPoint * hpLocations, int nLocations, pDst *Float32WindowDescriptorBuffer, oROI Size, NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType BorderType, ctx *StreamContext)error{
if ctx!=nil{return  status(C.nppiHistogramOfGradientsBorder_16u32f_C3R_Ctx(Npp16u * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c(), ctx.s)).ToError()}
return  status(C.nppiHistogramOfGradientsBorder_16u32f_C3R(Npp16u * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c())).ToError()}

func  HistogramOfGradientsBorder_16s32f_C1R(Npp16s * pSrc, nSrcStep int32, oSrcSize Size, oSrcOffset Point, NppiPoint * hpLocations, int nLocations, pDst *Float32WindowDescriptorBuffer, oROI Size, NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType BorderType, ctx *StreamContext)error{
if ctx!=nil{return  status(C.nppiHistogramOfGradientsBorder_16s32f_C1R_Ctx(Npp16s * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c(), ctx.s)).ToError()}
return  status(C.nppiHistogramOfGradientsBorder_16s32f_C1R(Npp16s * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c())).ToError()}

func  HistogramOfGradientsBorder_16s32f_C3R(Npp16s * pSrc, nSrcStep int32, oSrcSize Size, oSrcOffset Point, NppiPoint * hpLocations, int nLocations, pDst *Float32WindowDescriptorBuffer, oROI Size, NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType BorderType, ctx *StreamContext)error{
if ctx!=nil{return  status(C.nppiHistogramOfGradientsBorder_16s32f_C3R_Ctx(Npp16s * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c(), ctx.s)).ToError()}
return  status(C.nppiHistogramOfGradientsBorder_16s32f_C3R(Npp16s * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c())).ToError()}

func  HistogramOfGradientsBorder_32f_C1R(Npp32f * pSrc, nSrcStep int32, oSrcSize Size, oSrcOffset Point, NppiPoint * hpLocations, int nLocations, pDst *Float32WindowDescriptorBuffer, oROI Size, NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType BorderType, ctx *StreamContext)error{
if ctx !=nilreturn  status(C.nppiHistogramOfGradientsBorder_32f_C1R_Ctx(Npp32f * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c(), ctx.s)).ToError()}
return  status(C.nppiHistogramOfGradientsBorder_32f_C1R(Npp32f * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer,oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c())).ToError()}

func  HistogramOfGradientsBorder_32f_C3R(Npp32f * pSrc, nSrcStep int32, oSrcSize Size, oSrcOffset Point, NppiPoint * hpLocations, int nLocations, pDst *Float32WindowDescriptorBuffer, oROI Size, NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType BorderType, ctx *StreamContext)error{
if ctx !=nil{return  status(C.nppiHistogramOfGradientsBorder_32f_C3R_Ctx(Npp32f * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c(), ctx.s)).ToError()}
return  status(C.nppiHistogramOfGradientsBorder_32f_C3R(Npp32f * pSrc, (C.int)(nSrcStep), oSrcSize.c(), oSrcOffset.c(), NppiPoint * hpLocations, int nLocations, pDst.cptr()WindowDescriptorBuffer, oROI.c(), NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, eBorderType.c())).ToError()}
func  LabelMarkersGetBufferSize_8u_C1R(oROI Size, int * hpBufferSize)error{
return  status(C.nppiLabelMarkersGetBufferSize_8u_C1R(oROI.c(), int * hpBufferSize)).ToError()}
func  LabelMarkersGetBufferSize_8u32u_C1R(oROI Size, int * hpBufferSize)error{
return  status(C.nppiLabelMarkersGetBufferSize_8u32u_C1R(oROI.c(), int * hpBufferSize)).ToError()}
func  LabelMarkersGetBufferSize_16u_C1R(oROI Size, int * hpBufferSize)error{
return  status(C.nppiLabelMarkersGetBufferSize_16u_C1R(oROI.c(), int * hpBufferSize)).ToError()}

func  LabelMarkers_8u_C1IR(pSrc *Uint8Dst, int nSrcDstStep, oROI Size, Npp8u nMinVal, eNorm Norm, int * pNumber, Npp8u * pBuffer, ctx *StreamContext)error{
if ctx !=nil{return  status(C.nppiLabelMarkers_8u_C1IR_Ctx(pSrc.cptr()Dst, int nSrcDstStep, oROI.c(), Npp8u nMinVal, eNorm.c(), int * pNumber, Npp8u * pBuffer, ctx.s)).ToError()}

return  status(C.nppiLabelMarkers_8u_C1IR(pSrc.cptr()Dst, int nSrcDstStep, oROI.c(), Npp8u nMinVal, eNorm.c(), int * pNumber, Npp8u * pBuffer)).ToError()}
func  LabelMarkers_8u32u_C1R(pSrc *Uint8, nSrcStep int32, Npp32u * pDst, nDstStep int32, oROI Size, Npp8u nMinVal, eNorm Norm, int * pNumber, Npp8u * pBuffer, ctx *StreamContext)error{
	if ctx !=nil{return  status(C.nppiLabelMarkers_8u32u_C1R_Ctx(pSrc.cptr(), (C.int)(nSrcStep), Npp32u * pDst, (C.int)(nDstStep), oROI.c(), Npp8u nMinVal, eNorm.c(), int * pNumber, Npp8u * pBuffer, ctx.s)).ToError()}

return  status(C.nppiLabelMarkers_8u32u_C1R(pSrc.cptr(), (C.int)(nSrcStep), Npp32u * pDst, (C.int)(nDstStep), oROI.c(), Npp8u nMinVal, eNorm.c(), int * pNumber, Npp8u * pBuffer)).ToError()}
func  LabelMarkers_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, oROI Size, Npp16u nMinVal, eNorm Norm, int * pNumber, Npp8u * pBuffer, ctx *StreamContext)error{
	if ctx !=nil{return  status(C.nppiLabelMarkers_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, oROI.c(), Npp16u nMinVal, eNorm.c(), int * pNumber, Npp8u * pBuffer, ctx.s)).ToError()}

return  status(C.nppiLabelMarkers_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, oROI.c(), Npp16u nMinVal, eNorm.c(), int * pNumber, Npp8u * pBuffer)).ToError()}

func  CompressMarkerLabelsGetBufferSize_8u_C1R(int nStartingNumber, int * hpBufferSize)error{
return  status(C.nppiCompressMarkerLabelsGetBufferSize_8u_C1R(int nStartingNumber, int * hpBufferSize)).ToError()}

func  CompressMarkerLabelsGetBufferSize_32u8u_C1R(int nStartingNumber, int * hpBufferSize)error{
return  status(C.nppiCompressMarkerLabelsGetBufferSize_32u8u_C1R(int nStartingNumber, int * hpBufferSize)).ToError()}

func  CompressMarkerLabelsGetBufferSize_16u_C1R(int nStartingNumber, int * hpBufferSize)error{
return  status(C.nppiCompressMarkerLabelsGetBufferSize_16u_C1R(int nStartingNumber, int * hpBufferSize)).ToError()}

func  CompressMarkerLabelsGetBufferSize_32u_C1R(int nStartingNumber, int * hpBufferSize)error{
return  status(C.nppiCompressMarkerLabelsGetBufferSize_32u_C1R(int nStartingNumber, int * hpBufferSize)).ToError()}

func  CompressMarkerLabels_8u_C1IR(pSrc *Uint8Dst, int nSrcDstStep, oROI Size, int nStartingNumber, int * pNewNumber, Npp8u * pBuffer, ctx *StreamContext)error{
	if ctx !=nil{return  status(C.nppiCompressMarkerLabels_8u_C1IR_Ctx(pSrc.cptr()Dst, int nSrcDstStep, oROI.c(), int nStartingNumber, int * pNewNumber, Npp8u * pBuffer, ctx.s)).ToError()}
return  status(C.nppiCompressMarkerLabels_8u_C1IR(pSrc.cptr()Dst, int nSrcDstStep, oROI.c(), int nStartingNumber, int * pNewNumber, Npp8u * pBuffer)).ToError()}

func  CompressMarkerLabels_32u8u_C1R(Npp32u * pSrc, nSrcStep int32, pDst *Uint8, nDstStep int32, oROI Size, int nStartingNumber, int * pNewNumber, Npp8u * pBuffer, ctx *StreamContext)error{
	if ctx !=nil{return  status(C.nppiCompressMarkerLabels_32u8u_C1R_Ctx(Npp32u * pSrc, (C.int)(nSrcStep), pDst.cptr(), (C.int)(nDstStep), oROI.c(), int nStartingNumber, int * pNewNumber, Npp8u * pBuffer, ctx.s)).ToError()}
return  status(C.nppiCompressMarkerLabels_32u8u_C1R(Npp32u * pSrc, (C.int)(nSrcStep), pDst.cptr(), (C.int)(nDstStep), oROI.c(), int nStartingNumber, int * pNewNumber, Npp8u * pBuffer)).ToError()}


func  CompressMarkerLabels_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, oROI Size, int nStartingNumber, int * pNewNumber, Npp8u * pBuffer, ctx *StreamContext)error{
	if ctx !=nil{return  status(C.nppiCompressMarkerLabels_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, oROI.c(), int nStartingNumber, int * pNewNumber, Npp8u * pBuffer, ctx.s)).ToError()}
return  status(C.nppiCompressMarkerLabels_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, oROI.c(), int nStartingNumber, int * pNewNumber, Npp8u * pBuffer)).ToError()}


func  CompressMarkerLabels_32u_C1IR(Npp32u * pSrcDst, int nSrcDstStep, oROI Size, int nStartingNumber, int * pNewNumber, Npp8u * pBuffer, ctx *StreamContext)error{
	if ctx !=nil{return  status(C.nppiCompressMarkerLabels_32u_C1IR_Ctx(Npp32u * pSrcDst, int nSrcDstStep, oROI.c(), int nStartingNumber, int * pNewNumber, Npp8u * pBuffer, ctx.s)).ToError()}
return  status(C.nppiCompressMarkerLabels_32u_C1IR(Npp32u * pSrcDst, int nSrcDstStep, oROI.c(), int nStartingNumber, int * pNewNumber, Npp8u * pBuffer)).ToError()}


func  BoundSegments_8u_C1IR(pSrc *Uint8Dst, int nSrcDstStep, oROI Size, Npp8u nBorderVal, ctx *StreamContext)error{
	if ctx !=nil{return  status(C.nppiBoundSegments_8u_C1IR_Ctx(pSrc.cptr()Dst, int nSrcDstStep, oROI.c(), Npp8u nBorderVal, ctx.s)).ToError()}
return  status(C.nppiBoundSegments_8u_C1IR(pSrc.cptr()Dst, int nSrcDstStep, oROI.c(), Npp8u nBorderVal)).ToError()}

func  BoundSegments_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, oROI Size, Npp16u nBorderVal, ctx *StreamContext)error{
	if ctx !=nil{return  status(C.nppiBoundSegments_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, oROI.c(), Npp16u nBorderVal, ctx.s)).ToError()}
return  status(C.nppiBoundSegments_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, oROI.c(), Npp16u nBorderVal)).ToError()}

func  BoundSegments_32u_C1IR(Npp32u * pSrcDst, int nSrcDstStep, oROI Size, Npp32u nBorderVal, ctx *StreamContext)error{
	if ctx !=nil{return  status(C.nppiBoundSegments_32u_C1IR_Ctx(Npp32u * pSrcDst, int nSrcDstStep, oROI.c(), Npp32u nBorderVal, ctx.s)).ToError()}

return  status(C.nppiBoundSegments_32u_C1IR(Npp32u * pSrcDst, int nSrcDstStep, oROI.c(), Npp32u nBorderVal)).ToError()}
*/
