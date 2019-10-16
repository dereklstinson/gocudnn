package xtrakerns

//MakePlanarImageBatchesUint8 - for this to work all the each batch should have the same amount of channels and all the channels
//need to be the same size
func MakePlanarImageBatchesUint8() Kernel {
	return Kernel{
		Name: `MakePlanarImageBatchesUint8`,
		Code: `
		extern "C" __global__ void MakePlanarImageBatchesUint8(const int XThreads, //Should be channel size
														 const int Batches,
														 const int channelsperbatch,
														 const float *Srcs, //all the channels for everything.
														 float *dest)
		{
			const int batchsize = XThreads*channelsperbatch;
			for (int i = 0;i<Batches;i++)
			{
				for (int j = 0;j<channelsperbatch;j++)
				{
					CUDA_GRID_LOOP_X(xIdx, XThreads)
					{
					   dest[(i*batchsize)+(j*XThreads)+xIdx]=Srcs[(j*XThreads)+xIdx];
					}
				}
			
			}
		}`,
	}
}
