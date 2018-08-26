#ifndef OCLKERNEL_H_
#define OCLKERNEL_H_

#ifdef MAC  
#include <OpenCL/cl.h>  
#else  
#include <CL/cl.h>  
#endif  

#include <string>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>
 
 
#include "../Cuda/types.h"

class ocl;

class OclKernel
{
public:
	OclKernel(ocl *easycl, std::string sourceFilename, std::string kernelName, std::string source, cl_program program, cl_kernel kernel);
	~OclKernel(); 
	void Run2D(size_t global_worksize[2], size_t local_worksize[2], bool fast_read = false);
	void Run1D(size_t global_worksize, size_t local_worksize, bool fast_read = false);
 
	void Output(cl_mem devicearray);
	template<typename T>
	void Input(T value); 

	ocl* cl;
	cl_kernel kernel;
	std::string buildLog;
	std::string sourceFilename; // just for info really
	std::string kernelName; // this too
	cl_program program;
	cl_int error;
	std::string source;
	int nextArg;
 
}; 
 
#endif
