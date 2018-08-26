#include "OclKernel.h" 
#include "ocl.h"
#include <string>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <assert.h>

using namespace std;

template void OclKernel::Input(float value); 
template void OclKernel::Input(int value); 
template void OclKernel::Input(cl_float3 value); 
template void OclKernel::Input(cl_mem value); 

OclKernel::OclKernel(ocl *easycl, std::string sourceFilename, std::string kernelName, std::string source, cl_program program, cl_kernel kernel)
{
	this->sourceFilename = sourceFilename;
	this->kernelName = kernelName;
	this->source = source;
	this->cl = easycl;
	nextArg = 0;
	error = CL_SUCCESS;
	this->program = program;
	this->kernel = kernel;

}

OclKernel::~OclKernel()
{
	clReleaseProgram(this->program);	
	clReleaseKernel(this->kernel);
	
}


void OclKernel::Run2D(size_t global_worksize[2], size_t local_worksize[2], bool fast_read)
{
	int ND = 1;
	/*size_t global_ws = global_worksize;
	size_t local_ws = local_worksize;*/
	cl_command_queue *queue = cl->queue;
	cl_event *kernelFinishedEvent = new cl_event();
	error = clEnqueueNDRangeKernel(*(queue), kernel, ND, NULL, local_worksize, local_worksize, 0, NULL, kernelFinishedEvent);
	if (error != 0) {
		cout << "kernel failed to run, saving to failedkernel.cl" << endl;
		ofstream f;
		f.open("failedkernel.cl", ios_base::out);
		f << source << endl;
		f.close();
		switch (error) {
		case -4:
			throw std::runtime_error("Memory object allocation failure, code -4");
			break;
		case -5:
			throw std::runtime_error("Out of resources, code -5");
			break;
		case -11:
			throw std::runtime_error("Program build failure, code -11");
			break;
		case -46:
			throw std::runtime_error("Invalid kernel name, code -46");
			break;
		case -52:
			throw std::runtime_error("Invalid kernel args, code -52");
			break;
		case -54:
			throw std::runtime_error("Invalid work group size, code -54");
			break;
		default:
			throw std::runtime_error("Something went wrong, code " + toString(error));
		}
	}
	cl->checkError(error);
	clFinish(*queue);
	//// mark wrappers dirty:
	//for (int i = 0; i < (int)wrappersToDirty.size(); i++) {
	//	wrappersToDirty[i]->markDeviceDirty();
	//}	 
	//wrappersToDirty.clear();
	nextArg = 0;
}
 
void OclKernel::Run1D(size_t global_worksize, size_t local_worksize, bool fast_read)
{
	int ND = 1;
	/*size_t global_ws = global_worksize;
	size_t local_ws = local_worksize;*/
	cl_command_queue *queue = cl->queue;
	cl_event *kernelFinishedEvent = new cl_event();
	error = clEnqueueNDRangeKernel(*(queue), kernel, ND, NULL, &local_worksize, &local_worksize, 0, NULL, kernelFinishedEvent);
	if (error != 0) {
		cout << "kernel failed to run, saving to failedkernel.cl" << endl;
		ofstream f;
		f.open("failedkernel.cl", ios_base::out);
		f << source << endl;
		f.close();
		switch (error) {
		case -4:
			throw std::runtime_error("Memory object allocation failure, code -4");
			break;
		case -5:
			throw std::runtime_error("Out of resources, code -5");
			break;
		case -11:
			throw std::runtime_error("Program build failure, code -11");
			break;
		case -46:
			throw std::runtime_error("Invalid kernel name, code -46");
			break;
		case -52:
			throw std::runtime_error("Invalid kernel args, code -52");
			break;
		case -54:
			throw std::runtime_error("Invalid work group size, code -54");
			break;
		default:
			throw std::runtime_error("Something went wrong, code " + toString(error));
		}
	}
	delete kernelFinishedEvent; kernelFinishedEvent = NULL;
	cl->checkError(error); 
	clFinish(*queue);
	nextArg = 0;
} 


void OclKernel::Output(cl_mem devicearray)
{
	error = clSetKernelArg(kernel, nextArg, sizeof(cl_mem), &devicearray);
	cl->checkError(error);
	nextArg++;

}

template<typename T>
void OclKernel::Input(T value)
{
	error = clSetKernelArg(kernel, nextArg, sizeof(T), &value);
	cl->checkError(error);
	nextArg++;
}
  
