 /*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#include "device_memory.hpp"
#include "../convenience.h"

//#include "cuda_runtime_api.h"
#include "assert.h"

//////////////////////////    XADD    ///////////////////////////////

#ifdef __GNUC__
    
    #if __GNUC__*10 + __GNUC_MINOR__ >= 42

        #if !defined WIN32 && (defined __i486__ || defined __i586__ || defined __i686__ || defined __MMX__ || defined __SSE__  || defined __ppc__)
            #define CV_XADD __sync_fetch_and_add
        #else
            #include <ext/atomicity.h>
            #define CV_XADD __gnu_cxx::__exchange_and_add
        #endif
    #else
        #include <bits/atomicity.h>
        #if __GNUC__*10 + __GNUC_MINOR__ >= 34
            #define CV_XADD __gnu_cxx::__exchange_and_add
        #else
            #define CV_XADD __exchange_and_add
        #endif
  #endif
    
#elif defined WIN32 || defined _WIN32
    #include <intrin.h>
    #define CV_XADD(addr,delta) _InterlockedExchangeAdd((long volatile*)(addr), (delta))
#else

    template<typename _Tp> static inline _Tp CV_XADD(_Tp* addr, _Tp delta)
    { int tmp = *addr; *addr += delta; return tmp; }
    
#endif

////////////////////////    DeviceArray    /////////////////////////////
    
DeviceMemory::DeviceMemory() : data_(0), sizeBytes_(0), refcount_(0),mem_cl(0){}
DeviceMemory::DeviceMemory(void *ptr_arg, size_t sizeBytes_arg) : data_(ptr_arg), sizeBytes_(sizeBytes_arg), refcount_(0),mem_cl(0){}
DeviceMemory::DeviceMemory(size_t sizeBtes_arg)  : data_(0), sizeBytes_(0), refcount_(0),mem_cl(0) { create(sizeBtes_arg); }
DeviceMemory::~DeviceMemory() { release(); }

DeviceMemory::DeviceMemory(const DeviceMemory& other_arg)
    : data_(other_arg.data_), sizeBytes_(other_arg.sizeBytes_), refcount_(other_arg.refcount_),mem_cl(other_arg.mem_cl)
{
    if( refcount_ )
        CV_XADD(refcount_, 1);
}

DeviceMemory& DeviceMemory::operator = (const DeviceMemory& other_arg)
{
    if( this != &other_arg )
    {
        if( other_arg.refcount_ )
            CV_XADD(other_arg.refcount_, 1);
        release();
        mem_cl = other_arg.mem_cl;
        data_      = other_arg.data_;
        sizeBytes_ = other_arg.sizeBytes_;                
        refcount_  = other_arg.refcount_;
    }
    return *this;
}

void DeviceMemory::create(size_t sizeBytes_arg)
{
    if (sizeBytes_arg == sizeBytes_)
        return;
            
    if( sizeBytes_arg > 0)
    {        
        if( mem_cl )
            release();

        sizeBytes_ = sizeBytes_arg;
		if (!cl)
			cl = ocl::GetInstance();
		cl_int error;
		mem_cl = clCreateBuffer(*(cl->context), CL_MEM_READ_WRITE, sizeBytes_, 0, &error);
        //cudaSafeCall( cudaMalloc(&data_, sizeBytes_) );        

        refcount_ = new int;
        *refcount_ = 1;
    }
}

void DeviceMemory::copyTo(DeviceMemory& other) const
{
    if (empty())
        other.release();
    else
    {    
        other.create(sizeBytes_);    
        //cudaSafeCall( cudaMemcpy(other.data_, data_, sizeBytes_, cudaMemcpyDeviceToDevice) );
        //cudaSafeCall( cudaDeviceSynchronize() );
		ocl* cl = ocl::GetInstance();
		cl_int error = clEnqueueCopyBuffer(*(cl->queue),mem_cl, other.mem_cl, 0, 0, sizeBytes_,NULL, NULL, NULL);
		cl->checkError(error);
    }
}

void DeviceMemory::release()
{
    		 
    if( refcount_ && CV_XADD(refcount_, -1) == 1 )
    {
        delete refcount_;
        //cudaSafeCall( cudaFree(data_) );
	clReleaseMemObject(mem_cl);
		
    }
    data_ = 0;
    sizeBytes_ = 0;
    refcount_ = 0;
}

void DeviceMemory::upload(const void *host_ptr_arg, size_t sizeBytes_arg)
{
    create(sizeBytes_arg);
    //cudaSafeCall( cudaMemcpy(data_, host_ptr_arg, sizeBytes_, cudaMemcpyHostToDevice) );
    //cudaSafeCall( cudaDeviceSynchronize() ); 

	if (!cl)
		cl = ocl::GetInstance();
	cl_int error = clEnqueueWriteBuffer(*(cl->queue), mem_cl, CL_TRUE, 0, sizeBytes_, host_ptr_arg, 0, NULL, NULL);
	cl->checkError(error);
}

void DeviceMemory::download(void *host_ptr_arg) const
{    
    //cudaSafeCall( cudaMemcpy(host_ptr_arg, data_, sizeBytes_, cudaMemcpyDeviceToHost) );
    //cudaSafeCall( cudaDeviceSynchronize() );
	if (mem_cl == NULL)
		return;
	ocl* cl_ = ocl::GetInstance();
	cl_event event = NULL;
	cl_int error = clEnqueueReadBuffer(*(cl_->queue), mem_cl, CL_TRUE, 0, sizeBytes_, host_ptr_arg, 0, NULL, &event);
	cl_->checkError(error);

	cl_int err = clWaitForEvents(1, &event);
	clReleaseEvent(event);
	if (err != CL_SUCCESS) {
		throw std::runtime_error("wait for event on copytohost failed with " + toString(err));
	}
	cl->Synchronize();
}          

void DeviceMemory::swap(DeviceMemory& other_arg)
{
    std::swap(data_, other_arg.data_);
    std::swap(sizeBytes_, other_arg.sizeBytes_);
    std::swap(refcount_, other_arg.refcount_);
    std::swap(mem_cl,other_arg.mem_cl);
}

bool DeviceMemory::empty() const { return !mem_cl; }
size_t DeviceMemory::sizeBytes() const { return sizeBytes_; }


////////////////////////    DeviceArray2D    /////////////////////////////

DeviceMemory2D::DeviceMemory2D() : data_(0), step_(0), colsBytes_(0), rows_(0), refcount_(0),mem_cl(0) {}

DeviceMemory2D::DeviceMemory2D(int rows_arg, int colsBytes_arg)
    : data_(0), step_(0), colsBytes_(0), rows_(0), refcount_(0),mem_cl(0)
{ 
    create(rows_arg, colsBytes_arg); 
}

DeviceMemory2D::DeviceMemory2D(int rows_arg, int colsBytes_arg, void *data_arg, size_t step_arg)
    :  data_(data_arg), step_(step_arg), colsBytes_(colsBytes_arg), rows_(rows_arg), refcount_(0),mem_cl(0) {}

DeviceMemory2D::~DeviceMemory2D() { release(); }


DeviceMemory2D::DeviceMemory2D(const DeviceMemory2D& other_arg) :
    data_(other_arg.data_), step_(other_arg.step_), colsBytes_(other_arg.colsBytes_), rows_(other_arg.rows_), refcount_(other_arg.refcount_),mem_cl(other_arg.mem_cl)
{
    if( refcount_ )
        CV_XADD(refcount_, 1);
}

DeviceMemory2D& DeviceMemory2D::operator = (const DeviceMemory2D& other_arg)
{
    if( this != &other_arg )
    {
        if( other_arg.refcount_ )
            CV_XADD(other_arg.refcount_, 1);
        release();
        
        colsBytes_ = other_arg.colsBytes_;
        rows_ = other_arg.rows_;
        data_ = other_arg.data_;
        step_ = other_arg.step_;
        mem_cl = other_arg.mem_cl;
        refcount_ = other_arg.refcount_;
    }
    return *this;
}

void DeviceMemory2D::create(int rows_arg, int colsBytes_arg)
{
    if (colsBytes_ == colsBytes_arg && rows_ == rows_arg)
        return;
            
    if( rows_arg > 0 && colsBytes_arg > 0)
    {        
        if( mem_cl )
            release();
              
        colsBytes_ = colsBytes_arg;
        rows_ = rows_arg;
                        
        //cudaSafeCall( cudaMallocPitch( (void**)&data_, &step_, colsBytes_, rows_) );   
		if (!cl)
			cl = ocl::GetInstance();
		cl_int error;
		mem_cl = clCreateBuffer(*(cl->context), CL_MEM_READ_WRITE, colsBytes_*rows_, 0, &error);

        refcount_ = new int;
        *refcount_ = 1;
    }
}

void DeviceMemory2D::release()
{
    
    if( refcount_ && CV_XADD(refcount_, -1) == 1 )
    {
        delete refcount_;
        //cudaSafeCall( cudaFree(data_) );
	clReleaseMemObject(mem_cl);				 
    }

    colsBytes_ = 0;
    rows_ = 0;    
    data_ = 0;    
    step_ = 0;
    refcount_ = 0;
}

void DeviceMemory2D::copyTo(DeviceMemory2D& other) const
{
    if (empty())
        other.release();
    else
    {
        other.create(rows_, colsBytes_);    
        //cudaSafeCall( cudaMemcpy2D(other.data_, other.step_, data_, step_, colsBytes_, rows_, cudaMemcpyDeviceToDevice) );
        //cudaSafeCall( cudaDeviceSynchronize() );

		int size = colsBytes_*rows_;
		cl_int error = clEnqueueCopyBuffer(*(cl->queue), mem_cl, other.mem_cl, 0, 0, size, NULL, NULL, NULL);
		cl->checkError(error);
		cl->Synchronize();
    }
}

void DeviceMemory2D::upload(const void *host_ptr_arg, size_t host_step_arg, int rows_arg, int colsBytes_arg)
{
    create(rows_arg, colsBytes_arg);
    //cudaSafeCall( cudaMemcpy2D(data_, step_, host_ptr_arg, host_step_arg, colsBytes_, rows_, cudaMemcpyHostToDevice) );     
	if (!cl)
		cl = ocl::GetInstance();
	int size = colsBytes_*rows_; //???? \CA\FD\BEݶ\D4\C6\EB
	cl_int error = clEnqueueWriteBuffer(*(cl->queue), mem_cl, CL_TRUE, 0, size, host_ptr_arg, 0, NULL, NULL);
	cl->checkError(error);
}

void DeviceMemory2D::download(void *host_ptr_arg, size_t host_step_arg) const
{    
	if(!mem_cl)
	   return;
    //cudaSafeCall( cudaMemcpy2D(host_ptr_arg, host_step_arg, data_, step_, colsBytes_, rows_, cudaMemcpyDeviceToHost) );
	int size = colsBytes_*rows_;//???? \CA\FD\BEݶ\D4\C6\EB
	
	cl_event event = NULL;
	cl_int error = clEnqueueReadBuffer(*(cl->queue), mem_cl, CL_TRUE, 0, size, host_ptr_arg, 0, NULL, &event);
	cl->checkError(error);
	cl_int err = clWaitForEvents(1, &event);
	clReleaseEvent(event);
	if (err != CL_SUCCESS) {
		throw std::runtime_error("wait for event on copytohost failed with " + toString(err));
	}
}      

void DeviceMemory2D::swap(DeviceMemory2D& other_arg)
{    
	std::swap(mem_cl, other_arg.mem_cl);

    std::swap(data_, other_arg.data_);
    std::swap(step_, other_arg.step_);

    std::swap(colsBytes_, other_arg.colsBytes_);
    std::swap(rows_, other_arg.rows_);
    std::swap(refcount_, other_arg.refcount_);                 
}

bool DeviceMemory2D::empty() const { return !mem_cl; }
int DeviceMemory2D::colsBytes() const { return colsBytes_; }
int DeviceMemory2D::rows() const { return rows_; }
size_t DeviceMemory2D::step() const { return step_; }
