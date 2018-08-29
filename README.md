# Efusion_OpenCL

This project is the OpenCL version of ElasticFusion.

## How to build it

### In Ubuntu

What things you need:
* Ubuntu 16.04
* CMake
* OpenCL 1.2 (CUDA >= 8.0)
* OpenGL
* OpenNI2
* Eigen
* Pangolin
* SuiteSparse
* zlib
* libjpeg

### Build

Add OpencCL library (when using CUDA):

add the path into /Efusion_OpenCL/Core/src/CMakeLists.txt

  line 16: include_directories('/usr/include/CL')
  
  line 79: target_link_libraries(efusion /usr/lib/x86_64-linux-gnu/libOpenCL.so)
           or
           target_link_libraries(efusion /usr/local/cuda-8.0/lib64/libOpenCL.so)

Then, use build.sh to build Efusion
```
chmod a+x build.sh
./build.sh
```

