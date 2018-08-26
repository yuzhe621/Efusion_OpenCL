#ifndef OCL_H_
#define OCL_H_

#ifdef MAC 
#include <OpenCL/cl.h>  
#else  
#include <CL/cl.h>  
#endif  

#include <string>
#include <string.h>
#include <fstream>
#include <sstream>
#include <stdio.h>

class OclKernel;
template<typename T> class OclWarper;
class ocl
{ 
private:
	ocl()   //构造函数是私有的
	{
		InitOcl(0);
	}
	static ocl *instance;
public:
	static ocl * GetInstance()
	{
		if (instance == NULL)  //判断是否第一次调用
			instance = new ocl();
		return instance;
	}
	~ocl();

	void InitOcl(int gpuIndex);
	OclKernel* buildKernel(std::string kernelfilepath, std::string kernelname, std::string options, bool quiet = false);
	void Synchronize();
	static void checkError(cl_int error);
 
	cl_int error;  // easier than constantly declaring it in each method...
	cl_platform_id platform_id;
	cl_device_id device;
	cl_context *context;
	cl_command_queue *queue;
};
 
 
static std::string errorMessage(cl_int error)
{
	return NULL;
}

static std::string toString(int value)
{
	return NULL;
}

static std::string getFileContents(std::string filename) {
	std::ifstream t(filename.c_str());
	std::stringstream buffer;
	buffer << t.rdbuf();
	return buffer.str();
}

#endif
