/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
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

#ifndef CUDA_CUDAFUNCS_H_
#define CUDA_CUDAFUNCS_H_

#ifdef MAC 
#include <OpenCL/cl.h>  
#else  
#include <CL/cl.h>  
#endif  

 

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

#include "containers/device_array.hpp"
#include "types.h"


#include "convenience.h"

#include "../ocl/ocl.h"
#include "../ocl/OclKernel.h"
  
static void pyrDown(DeviceArray2D<unsigned short>& src, DeviceArray2D<unsigned short>& dst)
{	 	 
	dst.create(src.rows() / 2, src.cols() / 2);
	//dim3 block(32, 8);
	//dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));
	const float sigma_color = 30;

	//pyrDownGaussKernel << <grid, block >> >(src, dst, sigma_color);
	//cudaSafeCall(cudaGetLastError()); 	 
	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(dst.cols(), local_worksize[0]), getGridDim(dst.rows(), local_worksize[1]) };
	OclKernel *kernel = src.cl->buildKernel("cudafuncs.cl", "pyrDownGaussKernel", " ");
	kernel->Input(src.getDeviceMemory());
	kernel->Output(dst.getDeviceMemory());
	kernel->Input(src.cols());
	kernel->Input(src.rows());
	kernel->Input(dst.cols());
	kernel->Input(dst.rows());
	kernel->Input(sigma_color);
	kernel->Run2D(global_worksize, local_worksize);	 

	delete kernel;
}


static void createVMap(const CameraModel& intr, const DeviceArray2D<unsigned short> & depth, DeviceArray2D<float>& vmap, const float depthCutoff)
{
    vmap.create (depth.rows () * 3, depth.cols ());

    //dim3 block (32, 8);
    //dim3 grid (1, 1, 1);
    //grid.x = getGridDim (depth.cols (), block.x);
    //grid.y = getGridDim (depth.rows (), block.y);

    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    //computeVmapKernel<<<grid, block>>>(depth, vmap, 1.f / fx, 1.f / fy, cx, cy, depthCutoff);
    //cudaSafeCall (cudaGetLastError ());

	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(depth.cols(), local_worksize[0]),getGridDim(depth.rows(), local_worksize[1]) };
	OclKernel *kernel = depth.cl->buildKernel("cudafuncs.cl", "computeVmapKernel", " ");
	kernel->Input(depth.getDeviceMemory());
	kernel->Output(vmap.getDeviceMemory());
	kernel->Input(depth.cols());
	kernel->Input(depth.rows());
	kernel->Input(vmap.cols());
	kernel->Input(vmap.rows());
	kernel->Input(1.f / fx);
	kernel->Input(1.f / fy);
	kernel->Input(cx);
	kernel->Input(cy);
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;
}


static void createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap)
{
    nmap.create (vmap.rows (), vmap.cols ());

    int rows = vmap.rows () / 3;
    int cols = vmap.cols ();

    /*dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (cols, block.x);
    grid.y = getGridDim (rows, block.y);

    computeNmapKernel<<<grid, block>>>(rows, cols, vmap, nmap);
    cudaSafeCall (cudaGetLastError ());*/
	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(cols, local_worksize[0]),getGridDim(rows, local_worksize[1]) };
	OclKernel *kernel = vmap.cl->buildKernel("cudafuncs.cl", "computeNmapKernel", " ");
	kernel->Input(rows);
	kernel->Input(cols);
	kernel->Input(vmap.getDeviceMemory());
	kernel->Output(nmap.getDeviceMemory());
	kernel->Input(vmap.cols());
	kernel->Input(vmap.rows());
	kernel->Input(nmap.cols());
	kernel->Input(nmap.rows());	 
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;
}


static void tranformMaps(const DeviceArray2D<float>& vmap_src,
                  const DeviceArray2D<float>& nmap_src,
                  const mat33_& Rmat, const cl_float3& tvec,
                  DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_src.cols();
    int rows = vmap_src.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    nmap_dst.create(rows * 3, cols);

    /*dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    tranformMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());*/
	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(cols, local_worksize[0]),getGridDim(rows, local_worksize[1]) };
	OclKernel *kernel = vmap_src.cl->buildKernel("cudafuncs.cl", "tranformMapsKernel", " ");
	kernel->Input(rows);
	kernel->Input(cols);
	kernel->Input(vmap_src.getDeviceMemory());
	kernel->Output(nmap_src.getDeviceMemory());
	//kernel->Input(&Rmat);
	kernel->Input(tvec);
	kernel->Input(vmap_dst.getDeviceMemory());
	kernel->Output(nmap_dst.getDeviceMemory());
	kernel->Input(vmap_src.cols());
	kernel->Input(vmap_src.rows());
	kernel->Input(nmap_src.cols());
	kernel->Input(nmap_src.rows());
	kernel->Input(vmap_dst.cols());
	kernel->Input(vmap_dst.rows());
	kernel->Input(nmap_dst.cols());
	kernel->Input(nmap_dst.rows());
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;

}

static void copyMaps(const DeviceArray<float>& vmap_src,
              const DeviceArray<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst,
              DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_dst.cols();
    int rows = vmap_dst.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    nmap_dst.create(rows * 3, cols);

    /*dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    copyMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());*/
	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(cols, local_worksize[0]),getGridDim(rows, local_worksize[1]) };
	OclKernel *kernel = vmap_src.cl->buildKernel("cudafuncs.cl", "copyMapsKernel", " ");
	kernel->Input(rows);
	kernel->Input(cols);
	kernel->Input(vmap_src.getDeviceMemory());
	kernel->Output(nmap_src.getDeviceMemory());
	kernel->Input(vmap_dst.getDeviceMemory());
	kernel->Output(nmap_dst.getDeviceMemory());
	kernel->Input(vmap_dst.cols());
	kernel->Input(vmap_dst.rows());
	kernel->Input(nmap_dst.cols());
	kernel->Input(nmap_dst.rows());
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;
}


 
static void resizeMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output, bool normalize)
{
    int in_cols = input.cols ();
    int in_rows = input.rows () / 3;

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    output.create (out_rows * 3, out_cols);

    /*dim3 block (32, 8);
    dim3 grid (getGridDim (out_cols, block.x), getGridDim (out_rows, block.y));
    resizeMapKernel<normalize><< < grid, block>>>(out_rows, out_cols, in_rows, input, output);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());*/

	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(out_cols, local_worksize[0]),getGridDim(out_rows, local_worksize[1]) };
	OclKernel *kernel = input.cl->buildKernel("cudafuncs.cl", "resizeMapKernel", " ");
	kernel->Input(out_rows);
	kernel->Input(out_cols);
	kernel->Input(in_rows);
	kernel->Input(input.getDeviceMemory());
	kernel->Output(output.getDeviceMemory());	 
	kernel->Input(input.cols());
	kernel->Input(input.rows());
	kernel->Input(output.cols());
	kernel->Input(output.rows());
	kernel->Input((int)normalize);
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;

}


static void resizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    resizeMap(input, output, false);
}

static void resizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    resizeMap(input, output, true);
}

static void pyrDownGaussF(const DeviceArray2D<float>& src, DeviceArray2D<float> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    //dim3 block (32, 8);
    //dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    const float gaussKernel[25] = {1, 4, 6, 4, 1,
                    4, 16, 24, 16, 4,
                    6, 24, 36, 24, 6,
                    4, 16, 24, 16, 4,
                    1, 4, 6, 4, 1};

	

	/* float * gauss_cuda; 
	cudaMalloc((void**) &gauss_cuda, sizeof(float) * 25);
    cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);

    pyrDownKernelGaussF<<<grid, block>>>(src, dst, gauss_cuda);
    cudaSafeCall ( cudaGetLastError () );
    cudaFree(gauss_cuda);*/

	DeviceArray<float> gauss_cuda;
	gauss_cuda.upload(gaussKernel, 25);

	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(dst.cols(), local_worksize[0]),getGridDim(dst.rows(), local_worksize[1]) };
	OclKernel *kernel = src.cl->buildKernel("cudafuncs.cl", "pyrDownKernelGaussF", " ");
 
	kernel->Input(src.getDeviceMemory());
	kernel->Output(dst.getDeviceMemory());
	kernel->Input(gauss_cuda.getDeviceMemory());
	kernel->Input(src.cols());
	kernel->Input(src.rows());
	kernel->Input(dst.cols());
	kernel->Input(dst.rows()); 
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;
}

static void pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src, DeviceArray2D<unsigned char> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    //dim3 block (32, 8);
    //dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    const float gaussKernel[25] = {1, 4, 6, 4, 1,
                    4, 16, 24, 16, 4,
                    6, 24, 36, 24, 6,
                    4, 16, 24, 16, 4,
                    1, 4, 6, 4, 1};
	
	/*float * gauss_cuda;
    cudaMalloc((void**) &gauss_cuda, sizeof(float) * 25);
    cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);
    pyrDownKernelIntensityGauss<<<grid, block>>>(src, dst, gauss_cuda);
    cudaSafeCall ( cudaGetLastError () );
    cudaFree(gauss_cuda);*/
	DeviceArray<float> gauss_cuda;
	gauss_cuda.upload(gaussKernel, 25);
	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(dst.cols(), local_worksize[0]),getGridDim(dst.rows(), local_worksize[1]) };
	OclKernel *kernel = src.cl->buildKernel("cudafuncs.cl", "pyrDownKernelIntensityGauss", " "); 
	kernel->Input(src.getDeviceMemory());
	kernel->Output(dst.getDeviceMemory());
	kernel->Output(gauss_cuda.getDeviceMemory());
	kernel->Input(src.cols());
	kernel->Input(src.rows());
	kernel->Input(dst.cols());
	kernel->Input(dst.rows());
	kernel->Input(5);
	kernel->Input(5);
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;
}

static void verticesToDepth(DeviceArray<float>& vmap_src, DeviceArray2D<float> & dst, float cutOff)
{
	/*dim3 block(32, 8);
	dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

	verticesToDepthKernel << <grid, block >> >(vmap_src, dst, cutOff);
	cudaSafeCall(cudaGetLastError());*/
	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(dst.cols(), local_worksize[0]),getGridDim(dst.rows(), local_worksize[1]) };
	OclKernel *kernel = vmap_src.cl->buildKernel("cudafuncs.cl", "verticesToDepthKernel", " ");
	kernel->Input(vmap_src.getDeviceMemory());
	kernel->Output(dst.getDeviceMemory());
	kernel->Input(cutOff); 
	kernel->Input(dst.cols());
	kernel->Input(dst.rows()); 
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;

}

static void computeDerivativeImages(DeviceArray2D<unsigned char>& src, DeviceArray2D<short>& dx, DeviceArray2D<short>& dy)
{
    static bool once = false;

    if(!once)
    {
        float gsx3x3[9] = {0.52201,  0.00000, -0.52201,
                           0.79451, -0.00000, -0.79451,
                           0.52201,  0.00000, -0.52201};

        float gsy3x3[9] = {0.52201, 0.79451, 0.52201,
                           0.00000, 0.00000, 0.00000,
                           -0.52201, -0.79451, -0.52201};

        /*cudaMemcpyToSymbol(gsobel_x3x3, gsx3x3, sizeof(float) * 9);
        cudaMemcpyToSymbol(gsobel_y3x3, gsy3x3, sizeof(float) * 9);

        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());*/
 
        once = true;
    }

    /*dim3 block(32, 8);
    dim3 grid(getGridDim (src.cols (), block.x), getGridDim (src.rows (), block.y));

    applyKernel<<<grid, block>>>(src, dx, dy);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());*/

	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(src.cols(), local_worksize[0]),getGridDim(src.rows(), local_worksize[1]) };
	OclKernel *kernel = src.cl->buildKernel("cudafuncs.cl", "applyKernel", " ");
	kernel->Input(src.getDeviceMemory());
	kernel->Output(dx.getDeviceMemory());
	kernel->Output(dy.getDeviceMemory()); 
	kernel->Input(src.cols());
	kernel->Input(src.rows());
	kernel->Input(dx.cols());
	kernel->Input(dx.rows());
	kernel->Input(dy.cols());
	kernel->Input(dy.rows());
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;
}

static CameraModel CameraModel_level_cpu(CameraModel model, int level)
{
	CameraModel res;
	res.fx = model.fx / level;
	res.fx = model.fy / level;
	res.fx = model.cx / level;
	res.fx = model.cy / level;
	return res;
}


static void projectToPointCloud(const DeviceArray2D<float> & depth,
                         const DeviceArray2D<cl_float3> & cloud,
						CameraModel & intrinsics,
                         const int & level)
{
    //dim3 block (32, 8);
    //dim3 grid (getGridDim (depth.cols (), block.x), getGridDim (depth.rows (), block.y));

    CameraModel intrinsicsLevel = CameraModel_level_cpu(intrinsics,level);

    /*projectPointsKernel<<<grid, block>>>(depth, cloud, 1.0f / intrinsicsLevel.fx, 1.0f / intrinsicsLevel.fy, intrinsicsLevel.cx, intrinsicsLevel.cy);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());*/
	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(depth.cols(), local_worksize[0]),getGridDim(depth.rows(), local_worksize[1]) };
	OclKernel *kernel = depth.cl->buildKernel("cudafuncs.cl", "applyKernel", " ");
	kernel->Input(depth.getDeviceMemory());
	kernel->Output(cloud.getDeviceMemory());	 
	kernel->Input(1.0f / intrinsicsLevel.fx);
	kernel->Input(1.0f / intrinsicsLevel.fy);
	kernel->Input(intrinsicsLevel.cx);
	kernel->Input(intrinsicsLevel.cy);
	kernel->Input(depth.cols());
	kernel->Input(depth.rows());
	kernel->Input(cloud.cols());
	kernel->Input(cloud.rows());
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;

}

static void imageBGRToIntensity(cl_mem& mem/*cudaArray * cuArr*/,int src_w,int src_h, DeviceArray2D<unsigned char> & dst)
{
    /*dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));
    cudaSafeCall(cudaBindTextureToArray(inTex, cuArr));
    bgr2IntensityKernel<<<grid, block>>>(dst);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaUnbindTexture(inTex));*/

	size_t local_worksize[] = { 32, 8 };
	size_t global_worksize[] = { getGridDim(dst.cols(), local_worksize[0]),getGridDim(dst.rows(), local_worksize[1]) };
	OclKernel *kernel = dst.cl->buildKernel("cudafuncs.cl", "bgr2IntensityKernel", " ");
	kernel->Input(mem);
	kernel->Output(dst.getDeviceMemory()); 
	kernel->Input(src_w);
	kernel->Input(src_h);
	kernel->Input(dst.cols());
	kernel->Input(dst.rows());
	kernel->Run2D(global_worksize, local_worksize);
	delete kernel;
};


#endif /* CUDA_CUDAFUNCS_CUH_ */
