#include "ocl.h"
#include <stdio.h>
#include <exception>
#include <stdexcept>
#include <string>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include "OclKernel.h" 

using namespace std;
  
ocl* ocl::instance = NULL;

ocl::~ocl()
{
	clReleaseDevice(device);
	if (context)
	{
		clReleaseContext(*context);
		delete context;
		context = NULL;
	}
	if (queue)
	{
		clReleaseCommandQueue(*queue);
		delete queue;
		queue = NULL;
	}	
}

void ocl::InitOcl(int gpuIndex)
{
	error = 0;

	queue = 0;
	context = 0;

	// Platform
	cl_uint num_platforms;
	error = clGetPlatformIDs(1, &platform_id, &num_platforms);
	//        std::cout << "num platforms: " << num_platforms << std::endl;
	//        assert (num_platforms == 1);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error getting OpenCL platforms ids: " + errorMessage(error));
	}
	if (num_platforms == 0) {
		throw std::runtime_error("Error: no OpenCL platforms available");
	}

	cl_uint num_devices;
	error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 0, 0, &num_devices);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error getting OpenCL device ids: " + errorMessage(error));
	}
	//      std::cout << "num devices: " << num_devices << std::endl;
	cl_device_id *device_ids = new cl_device_id[num_devices];
	error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, num_devices, device_ids, &num_devices);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error getting OpenCL device ids: " + errorMessage(error));
	}

	if (gpuIndex >= static_cast<int>(num_devices)) {
		throw std::runtime_error("requested gpuindex " + toString(gpuIndex) + " goes beyond number of available device " + toString(num_devices));
	}
	device = device_ids[gpuIndex];
	delete[] device_ids;

	// Context
	context = new cl_context();
	*context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error creating OpenCL context, OpenCL errorcode: " + errorMessage(error));
	}
	// Command-queue
	queue = new cl_command_queue;
	*queue = clCreateCommandQueue(*context, device, 0, &error);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error creating OpenCL command queue, OpenCL errorcode: " + errorMessage(error));
	}
}

void ocl::checkError(cl_int error)
{
	if (error != CL_SUCCESS) {
		std::string message = toString(error);
		switch (error) {
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			message = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
			break;
		case CL_INVALID_ARG_SIZE:
			message = "CL_INVALID_ARG_SIZE";
			break;
		case CL_INVALID_BUFFER_SIZE:
			message = "CL_INVALID_BUFFER_SIZE";
			break;
		}
		cout << "opencl execution error, code " << error << " " << message << endl;
		throw std::runtime_error(std::string("OpenCL error, code: ") + message);
	}
}


OclKernel* ocl::buildKernel(std::string kernelfilepath, std::string kernelname, std::string options, bool quiet)
{
	std::string path = kernelfilepath.c_str();
	std::string source = getFileContents(path);

	size_t src_size = 0;
	const char *source_char = source.c_str();
	src_size = strlen(source_char);
	//    cl_program program = new cl_program();
	cl_program program = clCreateProgramWithSource(*context, 1, &source_char, &src_size, &error);
	checkError(error);

	//    error = clBuildProgram(program, 1, &device, "-cl-opt-disable", NULL, NULL);
	//    std::cout << "options: [" << options.c_str() << "]" << std::endl;
	error = clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL);

	char* build_log;
	size_t log_size;
	error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	checkError(error);
	build_log = new char[log_size + 1];
	error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	checkError(error);
	build_log[log_size] = '\0';
	std::string buildLogMessage = "";
	if (log_size > 2) {
		buildLogMessage = kernelfilepath + " build log: " + "\n" + build_log;
		if (!quiet) {
			cout << buildLogMessage << endl;
		}
	}
	delete[] build_log;
	checkError(error);

	cl_kernel kernel = clCreateKernel(program, kernelname.c_str(), &error);
	if (error != CL_SUCCESS) {
		std::string exceptionMessage = "";
		switch (error) {
		case -46:
			exceptionMessage = "Invalid kernel name, code -46, kernel " + kernelname + "\n" + buildLogMessage;
			break;
		default:
			exceptionMessage = "Something went wrong with clCreateKernel, OpenCL error code " + toString(error) + "\n" + buildLogMessage;
			break;
		}
		if (quiet) {
			cout << buildLogMessage << std::endl;
		}
		cout << "kernel build error:\n" << exceptionMessage << endl;
		cout << "storing failed kernel into: easycl-failedkernel.cl" << endl;
		exceptionMessage += "storing failed kernel into: easycl-failedkernel.cl\n";

		ofstream f;
		f.open("failedkernel.cl", ios_base::out);
		f << source << endl;
		f.close();
		throw std::runtime_error(exceptionMessage);
	}
	checkError(error);
	//    clReleaseProgram(program);
	OclKernel *newkernel = new OclKernel(this, kernelfilepath, kernelname, source, program, kernel);
	newkernel->buildLog = buildLogMessage;
	return newkernel;
}
 
 void ocl::Synchronize()
 {
	 clFinish(*this->queue);
 }
