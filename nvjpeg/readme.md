#nvjpeg - go bindings for nvjpeg.h

nvjpeg contains bindings for nvjpeg. 

note: CreateEx creates a special handler with a pinned allocator and device allocator. It uses callback functions. Make one if you want, but it will have to be part of the nvjpeg package since go is a pain when it comes to C types and librarys.

I will probably make an nvjpegutil to create wrappers that can be used with golang io package. It seems that stream and handle can be placed into JpegState struct and maybe even EncoderParams. and with that I can make a an interface that will do a reader and writer.  
