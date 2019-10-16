package xtrakerns

//NearestNeighborNHWC is a nearest neighbor resize function
func NearestNeighborNHWC() Kernel {
	return Kernel{
		Name: `NearestNeighborNHWC`,
		Code: `
		extern "C" __global__ void NearestNeighborNHWC(
			const int aligncorners,
			const int threads,
			const float *src,
			const int src_height,
			const int src_width,
			const int channels,
			const int dest_height,
			const int dest_width,
			const float height_scale,
			const float width_scale,
			float *dest)
		{
			CUDA_GRID_LOOP_X(i, threads)
			{
				int n = i;
				int c = n % channels;
				n /= channels;
				int dest_x = n % dest_width;
				n /= dest_width;
				int dest_y = n % dest_height;
				n /= dest_height;
				const float *src_data_n = &src[n * channels * src_height * src_width];
				const int src_y = fminf((aligncorners) ? (roundf(dest_y * height_scale))
													   : (floorf(dest_y * height_scale)),
										src_height - 1);
		
				const int src_x = fminf((aligncorners) ? (roundf(dest_x * width_scale))
													   : (floorf(dest_x * width_scale)),
										src_width - 1);
				const int idx = (src_y * src_width + src_x) * channels + c;
				dest[i] = src_data_n[idx];
			}
		}`,
	}
}

//NearestNeighborNHWCFP16 is a nearest neighbor resize function
func NearestNeighborNHWCFP16() Kernel {
	return Kernel{
		Name: `NearestNeighborNHWCFP16`,
		Code: `extern "C" __global__ void NearestNeighborNHWCFP16(
			const int aligncorners,
			const int threads,
			const __half *src,
			const int src_height,
			const int src_width,
			const int channels,
			const int dest_height,
			const int dest_width,
			const float height_scale,
			const float width_scale,
			__half *dest)
		{
			
			CUDA_GRID_LOOP_X(i, threads)
			{
				int n = i;
				int c = n % channels;
				n /= channels;
				int dest_x = n % dest_width;
				n /= dest_width;
				int dest_y = n % dest_height;
				n /= dest_height;
				const __half *src_data_n = &src[n * channels * src_height * src_width];
				const int src_y = fminf((aligncorners) ? (roundf(dest_y * height_scale))
													   : (floorf(dest_y * height_scale)),
										src_height - 1);
		
				const int src_x = fminf((aligncorners) ? (roundf(dest_x * width_scale))
													   : (floorf(dest_x * width_scale)),
										src_width - 1);                 
				const int idx = (src_y * src_width + src_x) * channels + c;
				dest[i] = src_data_n[idx];
			}
		}`,
	}
}

//NearestNeighborNCHW is a nearest neighbor resize function
func NearestNeighborNCHW() Kernel {
	return Kernel{
		Name: `NearestNeighborNCHW`,
		Code: `extern "C" __global__ void NearestNeighborNCHW(
			const int aligncorners,
			const int threads,
			const float *src,
			const int src_height,
			const int src_width,
			const int channels,
			const int dest_height,
			const int dest_width,
			const float height_scale,
			const float width_scale,
			float *dest)
		{
			CUDA_GRID_LOOP_X(i, threads)
			{
				int n = i;
				int dest_x = n % dest_width;
				n /= dest_width;
				int dest_y = n % dest_height;
				n /= dest_height;
				int c = n % channels;
				n /= channels;
				const float *src_data_n = &src[n * channels * src_height * src_width];
				const int src_y = fminf((aligncorners) ? (roundf(dest_y * height_scale))
													   : (floorf(dest_y * height_scale)),
										src_height - 1);
		
				const int src_x = fminf((aligncorners) ? (roundf(dest_x * width_scale))
													   : (floorf(dest_x * width_scale)),
										src_width - 1);
				const int idx = (c * src_height * src_width) + (src_y * src_width) + src_x;
				dest[i] = src_data_n[idx];
			}}`,
	}
}

//NearestNeighborNCHWFP16 is a nearest neighbor resize function
func NearestNeighborNCHWFP16() Kernel {
	return Kernel{
		Name: `NearestNeighborNCHWFP16`,
		Code: `extern "C" __global__ void NearestNeighborNCHWFP16(
			const int aligncorners,
			const int threads,
			const __half *src,
			const int src_height,
			const int src_width,
			const int channels,
			const int dest_height,
			const int dest_width,
			const float height_scale,
			const float width_scale,
			__half *dest)
		{
			CUDA_GRID_LOOP_X(i, threads)
			{
				int n = i;
				int dest_x = n % dest_width;
				n /= dest_width;
				int dest_y = n % dest_height;
				n /= dest_height;
				int c = n % channels;
				n /= channels;
				const __half *src_data_n = &src[n * channels * src_height * src_width];
				const int src_y = fminf((aligncorners) ? (roundf(dest_y * height_scale))
													   : (floorf(dest_y * height_scale)),
										src_height - 1);
		
				const int src_x = fminf((aligncorners) ? (roundf(dest_x * width_scale))
													   : (floorf(dest_x * width_scale)),
										src_width - 1);
				const int idx = (c * src_height * src_width) + (src_y * src_width) + src_x;
				dest[i] = src_data_n[idx];
			}
		}`,
	}
}

//NearestNeighborNCHWBack is a nearest neighbor resize function
func NearestNeighborNCHWBack() Kernel {
	return Kernel{
		Name: `NearestNeighborNCHWBack`,
		Code: `extern "C" __global__ void NearestNeighborNCHWBack(
			const int aligncorners,
			const int threads,
			float *src,
			const int src_height,
			const int src_width,
			const int channels,
			const int dest_height,
			const int dest_width,
			const float height_scale,
			const float width_scale,
			float *dest)
		{
			CUDA_GRID_LOOP_X(i, threads)
			{
				int n = i;
				int src_x = n % src_width;
				n /= src_width;
				int src_y = n % src_height;
				n /= src_height;
				int c = n % channels;
				n /= channels;
				float *src_data_n = &src[n * channels * src_height * src_width];
				const int dest_y = fminf((aligncorners) ? (roundf(src_y * height_scale))
														: (floorf(src_y * height_scale)),
										 dest_height - 1);
		
				const int dest_x = fminf((aligncorners) ? (roundf(src_x * width_scale))
														: (floorf(src_x * width_scale)),
										 dest_width - 1);
				const int idx = (c * dest_width * dest_height) + (dest_y * dest_width) + dest_x;
				atomicAdd(&src_data_n[idx], dest[i]);
			}
		}`,
	}
}

//NearestNeighborNCHWBackFP16 is a nearest neighbor resize function
func NearestNeighborNCHWBackFP16() Kernel {
	return Kernel{
		Name: `NearestNeighborNCHWBackFP16`,
		Code: `extern "C" __global__ void NearestNeighborNCHWBackFP16(
			const int aligncorners,
			const int threads,
			__half *src,
			const int src_height,
			const int src_width,
			const int channels,
			const int dest_height,
			const int dest_width,
			const float height_scale,
			const float width_scale,
			__half *dest)
		{
			CUDA_GRID_LOOP_X(i, threads)
			{
				int n = i;
				int src_x = n % src_width;
				n /= src_width;
				int src_y = n % src_height;
				n /= src_height;
				int c = n % channels;
				n /= channels;
				__half *src_data_n = &src[n * channels * src_height * src_width];
				const int dest_y = fminf((aligncorners) ? (roundf(src_y * height_scale))
														: (floorf(src_y * height_scale)),
										 dest_height - 1);
		
				const int dest_x = fminf((aligncorners) ? (roundf(src_x * width_scale))
														: (floorf(src_x * width_scale)),
										 dest_width - 1);
				const int idx = (c * dest_width * dest_height) + (dest_y * dest_width) + dest_x;
				atomicAdd(&src_data_n[idx], dest[i]);
			}
		}`,
	}
}

//NearestNeighborNHWCBack is a nearest neighbor resize function
func NearestNeighborNHWCBack() Kernel {
	return Kernel{
		Name: `NearestNeighborNHWCBack`,
		Code: `extern "C" __global__ void NearestNeighborNHWCBack(
			const int aligncorners,
			const int threads,
			float *src,
			const int src_height,
			const int src_width,
			const int channels,
			const int dest_height,
			const int dest_width,
			const float height_scale,
			const float width_scale,
			float *dest)
		{
			CUDA_GRID_LOOP_X(i, threads)
			{
				int n = i;
				int c = n % channels;
				n /= channels;
				int src_x = n % src_width;
				n /= src_width;
				int src_y = n % src_height;
				n /= src_height;
				float *src_data_n = &src[n * channels * src_height * src_width];
				const int dest_y = fminf((aligncorners) ? (roundf(src_y * height_scale))
														: (floorf(src_y * height_scale)),
										 dest_height - 1);
		
				const int dest_x = fminf((aligncorners) ? (roundf(src_x * width_scale))
														: (floorf(src_x * width_scale)),
										 dest_width - 1);
				const int idx = (dest_y * dest_width + dest_x) * channels + c;
				atomicAdd(&src_data_n[idx], dest[i]);
			}
		}`,
	}
}

//NearestNeighborNHWCBackFP16 is a nearest neighbor resize function
func NearestNeighborNHWCBackFP16() Kernel {
	return Kernel{
		Name: `NearestNeighborNHWCBackFP16`,
		Code: `extern "C" __global__ void NearestNeighborNHWCBackFP16(
			const int aligncorners,
			const int threads,
			__half *src,
			const int src_height,
			const int src_width,
			const int channels,
			const int dest_height,
			const int dest_width,
			const float height_scale,
			const float width_scale,
			__half *dest)
		{
			CUDA_GRID_LOOP_X(i, threads)
			{
				int n = i;
				int c = n % channels;
				n /= channels;
				int src_x = n % src_width;
				n /= src_width;
				int src_y = n % src_height;
				n /= src_height;
				__half *src_data_n = &src[n * channels * src_height * src_width];
				const int dest_y = fminf((aligncorners) ? (roundf(src_y * height_scale))
														: (floorf(src_y * height_scale)),
										 dest_height - 1);
		
				const int dest_x = fminf((aligncorners) ? (roundf(src_x * width_scale))
														: (floorf(src_x * width_scale)),
										 dest_width - 1);
				const int idx = (dest_y * dest_width + dest_x) * channels + c;
		
				atomicAdd(&src_data_n[idx], dest[i]);
			}
		}`,
	}
}
