run:

	nvcc --gpu-architecture=compute_75 --gpu-code=compute_75 --ptx gocudnnxtra.cu


gpuarch75:
	nvcc --gpu-architecture=compute_75 --gpu-code=compute_75 --ptx gocudnnxtra.cu -o gocudnnxtra75.ptx

gpuarch61:
	nvcc --gpu-architecture=compute_61 --gpu-code=compute_61 --ptx gocudnnxtra.cu -o gocudnnxtra61.ptx

gpuarch75fp16:
	nvcc --gpu-architecture=compute_75 --gpu-code=compute_75 --ptx gocudnnxtrafp16.cu -o gocudnnxtra75fp16.ptx

gpuarch61fp16:
	nvcc --gpu-architecture=compute_61 --gpu-code=compute_61 --ptx gocudnnxtrafp16.cu -o gocudnnxtra61fp16.ptx


gpuarch61both:
	nvcc --gpu-architecture=compute_61 --gpu-code=compute_61 --ptx gocudnnextraboth.cu -o gocudnnxtra61both.ptx

gpuarch75both:
	nvcc --gpu-architecture=compute_75 --gpu-code=compute_75 --ptx gocudnnextraboth.cu -o gocudnnxtra75both.ptx