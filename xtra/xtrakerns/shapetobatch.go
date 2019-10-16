package xtrakerns

//ShapetoBatch4DNCHWFP16 is like ShapetoBatch4DNCHW
func ShapetoBatch4DNCHWFP16() Kernel {
	return Kernel{
		Name: `ShapetoBatch4DNCHWFP16`,
		Code: `
		extern "C" __global__ void ShapetoBatch4DNCHWFP16(
			const int xThreads,
			const int yThreads,
			const int zThreads,
			const int hSize,
			const int wSize,
			const int num_original_batches,
			const int BatchVolume,
			const int OriginalVol,
			const int N1,
			const int N2,
			const int hstride,
			const int wstride,
			__half *shape,
			__half *batch,
			const int h_over_scan,
			const int w_over_scan,
			const bool S2B)
		{
			int batch0 = N2 * xThreads * yThreads * zThreads;
			int batch1 = xThreads * yThreads * zThreads;
			int batch2 = xThreads * yThreads;
			int batch3 = yThreads;
			for (int b = 0;b<num_original_batches;b++)
			{
				const int ShapeOffset = OriginalVol*b;
				const int BatchOffset=BatchVolume*b;
			for (int i = 0; i < N1; i++)
			{
				for (int j = 0; j < N2; j++)
				{
					CUDA_GRID_AXIS_LOOP(xIdx, xThreads, x)
					{
						CUDA_GRID_AXIS_LOOP(yIdx, yThreads, y)
						{
							CUDA_GRID_AXIS_LOOP(zIdx, zThreads, z)
							{
		
								int oh = (hstride * i) + yIdx;
								int ow = (wstride * j) + zIdx;
		
								if (S2B )
								{
									if (oh < hSize && ow < wSize)
									{
										batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] =
											shape[ShapeOffset + (xIdx * wSize * hSize) + (oh * wSize) + ow];
									}
									else
									{
										if (h_over_scan>0 && ow<wSize){
											batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0;
										}
										if (w_over_scan>0 && oh<hSize){
											batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0; 
										}
									   
									}
								}
								else
								{
									shape[ShapeOffset + (xIdx * wSize * hSize) + (oh * wSize) + ow] +=
										batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx];
								}
							}
						}
					}
				}
			}
		}
		}`,
	}
}

//ShapetoBatch4DNCHW Does a stride shape to batch. Make sure values on receiving end are set to zero when s2b is 0
func ShapetoBatch4DNCHW() Kernel {
	return Kernel{
		Name: `ShapetoBatch4DNCHW`,
		Code: `
		extern "C" __global__ void ShapetoBatch4DNCHW(
			const int xThreads,
			const int yThreads,
			const int zThreads,
			const int hSize,
			const int wSize,
			const int num_original_batches,
			const int BatchVolume,
			const int OriginalVol,
			const int N1,
			const int N2,
			const int hstride,
			const int wstride,
			float *shape,
			float *batch,
			const int h_over_scan,
			const int w_over_scan,
			const bool S2B)
		{
			int batch0 = N2 * xThreads * yThreads * zThreads;
			int batch1 = xThreads * yThreads * zThreads;
			int batch2 = xThreads * yThreads;
			int batch3 = yThreads;
			for (int b = 0;b<num_original_batches;b++)
			{
				const int ShapeOffset = OriginalVol*b;
				const int BatchOffset=BatchVolume*b;
			for (int i = 0; i < N1; i++)
			{
				for (int j = 0; j < N2; j++)
				{
					CUDA_GRID_AXIS_LOOP(xIdx, xThreads, x)
					{
						CUDA_GRID_AXIS_LOOP(yIdx, yThreads, y)
						{
							CUDA_GRID_AXIS_LOOP(zIdx, zThreads, z)
							{
		
								int oh = (hstride * i) + yIdx;
								int ow = (wstride * j) + zIdx;
		
								if (S2B )
								{
									if (oh < hSize && ow < wSize)
									{
										batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] =
											shape[ShapeOffset + (xIdx * wSize * hSize) + (oh * wSize) + ow];
									}
									else
									{
										if (h_over_scan>0 && ow<wSize){
											batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0;
										}
										if (w_over_scan>0 && oh<hSize){
											batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0; 
										}
									   
									}
								}
								else
								{
									shape[ShapeOffset + (xIdx * wSize * hSize) + (oh * wSize) + ow] +=
										batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx];
								}
							}
						}
					}
				}
			}
		}
		}
		`,
	}
}

//ShapeToBatch4DNHWC Does a stride shape to batch.
//Make sure values on receiving end are set to zero when s2b is 0
func ShapeToBatch4DNHWC() Kernel {
	return Kernel{
		Name: `ShapetoBatch4DNHWC`,
		Code: `extern "C" __global__ void ShapetoBatch4DNHWC(
			const int xThreads,
			const int yThreads,
			const int zThreads,
			const int hSize,
			const int wSize,
			const int num_original_batches,
			const int BatchVolume,
			const int OriginalVol,
			const int N1,
			const int N2,
			const int hstride,
			const int wstride,
			float *shape,
			float *batch,
			const int h_over_scan,
			const int w_over_scan,
			const bool S2B)
		{
			int batch0 = N2 * xThreads * yThreads * zThreads;
			int batch1 = xThreads * yThreads * zThreads;
			int batch2 = yThreads * zThreads;
			int batch3 = zThreads;
			for (int b = 0;b<num_original_batches;b++)
			{
				const int ShapeOffset = OriginalVol*b;
				const int BatchOffset=BatchVolume*b;
			for (int i = 0; i < N1; i++)
			{
				for (int j = 0; j < N2; j++)
				{
					CUDA_GRID_AXIS_LOOP(xIdx, xThreads, x)
					{
						CUDA_GRID_AXIS_LOOP(yIdx, yThreads, y)
						{
							CUDA_GRID_AXIS_LOOP(zIdx, zThreads, z)
							{
		
								int oh = (hstride * i) + xIdx;
								int ow = (wstride * j) + yIdx;
		
								if (S2B)
								{
									if (oh < hSize && ow < wSize)
									{
										batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] =
											shape[ShapeOffset + (oh * hSize * zThreads) + (ow * zThreads) + zIdx];
									}
									else
									{
										if (h_over_scan>0 && ow<wSize){
										batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0;
										}
										if (w_over_scan>0 && oh<hSize){
											batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0;
										}
									}
								}
								else
								{
									shape[ShapeOffset + (oh * hSize * zThreads) + (ow * zThreads) + zIdx] +=
										batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx];
								}
							}
						}
					}
				}
			}
		}
		}`,
	}
}

//ShapeToBatch4DNHWCFP16 Does a stride shape to batch.
// Make sure values on receiving end are set to zero when s2b is 0
func ShapeToBatch4DNHWCFP16() Kernel {
	return Kernel{
		Name: `ShapetoBatch4DNHWCFP16`,
		Code: `extern "C" __global__ void ShapetoBatch4DNHWCFP16(
		const int xThreads,
		const int yThreads,
		const int zThreads,
		const int hSize,
		const int wSize,
		const int num_original_batches,
		const int BatchVolume,
		const int OriginalVol,
		const int N1,
		const int N2,
		const int hstride,
		const int wstride,
		__half *shape,
		__half *batch,
		const int h_over_scan,
		const int w_over_scan,
		const bool S2B)
	{
		int batch0 = N2 * xThreads * yThreads * zThreads;
		int batch1 = xThreads * yThreads * zThreads;
		int batch2 = yThreads * zThreads;
		int batch3 = zThreads;
		for (int b = 0;b<num_original_batches;b++)
		{
			const int ShapeOffset = OriginalVol*b;
			const int BatchOffset=BatchVolume*b;
		for (int i = 0; i < N1; i++)
		{
			for (int j = 0; j < N2; j++)
			{
				CUDA_GRID_AXIS_LOOP(xIdx, xThreads, x)
				{
					CUDA_GRID_AXIS_LOOP(yIdx, yThreads, y)
					{
						CUDA_GRID_AXIS_LOOP(zIdx, zThreads, z)
						{
	
							int oh = (hstride * i) + xIdx;
							int ow = (wstride * j) + yIdx;
	
							if (S2B)
							{
								if (oh < hSize && ow < wSize)
								{
									batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] =
										shape[ShapeOffset + (oh * hSize * zThreads) + (ow * zThreads) + zIdx];
								}
								else
								{
									if (h_over_scan>0 && ow<wSize){
									batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0;
									}
									if (w_over_scan>0 && oh<hSize){
										batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx] = 0;
									}
								}
							}
							else
							{
								shape[ShapeOffset + (oh * hSize * zThreads) + (ow * zThreads) + zIdx] +=
									batch[BatchOffset + (i * batch0) + (j * batch1) + (xIdx * batch2) + (yIdx * batch3) + zIdx];
							}
						}
					}
				}
			}
		}
	}
	}
	`,
	}
}
