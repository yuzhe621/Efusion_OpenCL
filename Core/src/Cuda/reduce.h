#ifndef REDUCE_H_
#define REDUCE_H_

#include "containers/device_array.hpp"
#include "types.h" 
#include "../ocl/OclKernel.h"
#include "types.h"

static void rgbStep(const DeviceArray2D<DataTerm> & corresImg,
	const float & sigma,
	const DeviceArray2D<cl_float3> & cloud,
	const float & fx,
	const float & fy,
	const DeviceArray2D<short> & dIdx,
	const DeviceArray2D<short> & dIdy,
	const float & sobelScale,
	DeviceArray<JtJJtrSE3> & sum,
	DeviceArray<JtJJtrSE3> & out,
	float * matrixA_host,
	float * vectorB_host,
	int threads,
	int blocks)
{
	/*RGBReduction rgb;
	rgb.corresImg = corresImg;
	rgb.cols = corresImg.cols();
	rgb.rows = corresImg.rows();
	rgb.sigma = sigma;
	rgb.cloud = cloud;
	rgb.fx = fx;
	rgb.fy = fy;
	rgb.dIdx = dIdx;
	rgb.dIdy = dIdy;
	rgb.sobelScale = sobelScale;
	rgb.N = rgb.cols * rgb.rows;
	rgb.out = sum;*/

	/*rgbKernel << <blocks, threads >> >(rgb);
	reduceSum << <1, MAX_THREADS >> >(sum, out, blocks);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());*/

	ocl* cl = ocl::GetInstance(); 
	size_t local_worksize = threads;
	size_t global_worksize = blocks;
	OclKernel *kernel = cl->buildKernel("./Cuda/reduce.cl", "rgbKernel", " ");
	kernel->Input(corresImg.getDeviceMemory());
	kernel->Input(sigma);
	kernel->Input(cloud.getDeviceMemory());
	kernel->Input(fx);
	kernel->Input(fy);
	kernel->Input(dIdx.getDeviceMemory());
	kernel->Input(dIdy.getDeviceMemory());
	kernel->Input(sobelScale); 
	kernel->Input(out.getDeviceMemory()); 
	kernel->Run1D(global_worksize, local_worksize);
	delete kernel; kernel = NULL;

	OclKernel *kernel1 = cl->buildKernel("./Cuda/reduce.cl", "reduceSum_kernel", " ");
	kernel1->Input(sum.getDeviceMemory());
	kernel1->Input(out.getDeviceMemory());
	kernel1->Input(blocks);
	local_worksize = MAX_THREADS;
	global_worksize = 1;
	kernel1->Run1D(global_worksize, local_worksize);
	delete kernel1; kernel1 = NULL;
	cl->Synchronize();

	float host_data[32];
	out.download((JtJJtrSE3 *)&host_data[0]);

	int shift = 0;
	for (int i = 0; i < 6; ++i)
	{
		for (int j = i; j < 7; ++j)
		{
			float value = host_data[shift++];
			if (j == 6)
				vectorB_host[i] = value;
			else
				matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
		}
	}
}




static void computeRgbResidual(const float & minScale,
	const DeviceArray2D<short> & dIdx,
	const DeviceArray2D<short> & dIdy,
	const DeviceArray2D<float> & lastDepth,
	const DeviceArray2D<float> & nextDepth,
	const DeviceArray2D<unsigned char> & lastImage,
	const DeviceArray2D<unsigned char> & nextImage,
	DeviceArray2D<DataTerm> & corresImg,
	DeviceArray<cl_int2> & sumResidual,
	const float maxDepthDelta,
	const cl_float3 & kt,
	const mat33_& krkinv,
	int & sigmaSum,
	int & count,
	int threads,
	int blocks)
{
	int cols = nextImage.cols();
	int rows = nextImage.rows();

	/*RGBResidual rgb;
	rgb.minScale = minScale;
	rgb.dIdx = dIdx;
	rgb.dIdy = dIdy;

	rgb.lastDepth = lastDepth;
	rgb.nextDepth = nextDepth;

	rgb.lastImage = lastImage;
	rgb.nextImage = nextImage;

	rgb.corresImg = corresImg;

	rgb.maxDepthDelta = maxDepthDelta;

	rgb.kt = kt;
	rgb.krkinv = krkinv;

	rgb.cols = cols;
	rgb.rows = rows;
	rgb.pitch = dIdx.step();
	rgb.imgPitch = nextImage.step();

	rgb.N = cols * rows;
	rgb.out = sumResidual;*/

	//residualKernel << <blocks, threads >> >(rgb);

	cl_int2 out_host = { 0, 0 };
	//int2 * out;
	/*cudaMalloc(&out, sizeof(int2));
	cudaMemcpy(out, &out_host, sizeof(int2), cudaMemcpyHostToDevice);
	reduceSum << <1, MAX_THREADS >> >(sumResidual, out, blocks);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	cudaMemcpy(&out_host, out, sizeof(int2), cudaMemcpyDeviceToHost);
	cudaFree(out);*/

	DeviceArray<mat33> krkinv0;
	krkinv0.upload(&krkinv, 1);

	DeviceArray<cl_int2> out;
	out.upload(&out_host, 1);

	ocl* cl = ocl::GetInstance();
	size_t local_worksize = threads;
	size_t global_worksize = blocks;
	OclKernel *kernel = cl->buildKernel("./Cuda/reduce.cl", "residualKernel", " ");
	kernel->Input(minScale);
	kernel->Input(dIdx.getDeviceMemory());
	kernel->Input(dIdy.getDeviceMemory());
	kernel->Input(lastDepth.getDeviceMemory());
	kernel->Input(nextDepth.getDeviceMemory());
	kernel->Input(lastImage.getDeviceMemory());
	kernel->Input(nextImage.getDeviceMemory());
	kernel->Input(corresImg.getDeviceMemory());	 
	kernel->Input(maxDepthDelta);
	kernel->Input(kt);
	kernel->Input(krkinv0.getDeviceMemory());
	kernel->Run1D(global_worksize, local_worksize);
	delete kernel; kernel = NULL;

	OclKernel *kernel1 = cl->buildKernel("./Cuda/reduce.cl", "reduceSum_int2_kernel", " ");
	kernel1->Input(sumResidual.getDeviceMemory());
	kernel1->Input(out.getDeviceMemory());
	kernel1->Input(blocks);
	local_worksize = MAX_THREADS;
	global_worksize = 1;
	kernel1->Run1D(global_worksize, local_worksize);
	delete kernel1; kernel1 = NULL;
	cl->Synchronize();

	count = out_host.s[0];
	sigmaSum = out_host.s[1];
}

static void so3Step(/*const*/ DeviceArray2D<unsigned char>& lastImage,
	/*const*/ DeviceArray2D<unsigned char>& nextImage,
	/*const*/ mat33& imageBasis,
	/*const*/ mat33& kinv,
	/*const*/ mat33& krlr,
	DeviceArray<JtJJtrSO3>& sum,
	DeviceArray<JtJJtrSO3>& out,
	float * matrixA_host,
	float * vectorB_host,
	float * residual_host,
	int threads,
	int blocks)
{
	int cols = nextImage.cols();
	int rows = nextImage.rows();

	/*SO3Reduction_ so3;

	so3.lastImage = lastImage;

	so3.nextImage = nextImage;

	so3.imageBasis = imageBasis;
	so3.kinv = kinv;
	so3.krlr = krlr;

	so3.cols = cols;
	so3.rows = rows;

	so3.N = cols * rows;

	so3.out = sum;*/
	  
	/*so3Kernel << <blocks, threads >> >(so3);
	reduceSum << <1, MAX_THREADS >> >(sum, out, blocks);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());*/

	ocl* cl = ocl::GetInstance();

	DeviceArray<mat33> imageBasis0;
	DeviceArray<mat33> kinv0;
	DeviceArray<mat33> krlr0;
	imageBasis0.upload(&imageBasis, 1);
	kinv0.upload(&kinv, 1);
	krlr0.upload(&krlr, 1);

	size_t local_worksize = threads;
	size_t global_worksize = blocks;
	OclKernel *kernel = cl->buildKernel("./Cuda/reduce.cl", "so3Kernel", " ");
	kernel->Input(lastImage.getDeviceMemory());
	kernel->Input(nextImage.getDeviceMemory());
	kernel->Input(imageBasis0.getDeviceMemory());
	kernel->Input(kinv0.getDeviceMemory());
	kernel->Input(krlr0.getDeviceMemory());
	kernel->Input((int)false);
	kernel->Input(cols);
	kernel->Input(rows);
	kernel->Input(cols * rows);
	kernel->Output(out.getDeviceMemory());
	kernel->Run1D(global_worksize, local_worksize);
	delete kernel; kernel = NULL;
	//����kernel ִ�д��󣬵�������error

	OclKernel *kernel1 = cl->buildKernel("./Cuda/reduce.cl", "reduceSum_kernel", " ");
	kernel1->Input(sum.getDeviceMemory());
	kernel1->Input(out.getDeviceMemory());
	kernel1->Input(blocks);
	local_worksize = MAX_THREADS;
	global_worksize = 1;
	kernel1->Run1D(global_worksize, local_worksize);
	delete kernel1; kernel1 = NULL;
	cl->Synchronize();

	float host_data[11];
	out.download((JtJJtrSO3*)&host_data[0]);

	int shift = 0;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = i; j < 4; ++j)
		{
			float value = host_data[shift++];
			if (j == 3)
				vectorB_host[i] = value;
			else
				matrixA_host[j * 3 + i] = matrixA_host[i * 3 + j] = value;
		}
	}

	residual_host[0] = host_data[9];
	residual_host[1] = host_data[10];
}


void icpStep(mat33& Rcurr,
	cl_float3& tcurr,
	DeviceArray2D<float>& vmap_curr,
	DeviceArray2D<float>& nmap_curr,
	mat33& Rprev_inv,
	cl_float3& tprev,
	CameraModel& intr,
	DeviceArray2D<float>& vmap_g_prev,
	DeviceArray2D<float>& nmap_g_prev,
	float distThres,
	float angleThres,
	DeviceArray<JtJJtrSE3> & sum,
	DeviceArray<JtJJtrSE3> & out,
	float * matrixA_host,
	float * vectorB_host,
	float * residual_host,
	int threads,
	int blocks)
{
	int cols = vmap_curr.cols();
	int rows = vmap_curr.rows() / 3;

	/*ICPReduction icp;
	icp.Rcurr = Rcurr;
	icp.tcurr = tcurr;
	icp.vmap_curr = vmap_curr;
	icp.nmap_curr = nmap_curr;
	icp.Rprev_inv = Rprev_inv;
	icp.tprev = tprev;
	icp.intr = intr;
	icp.vmap_g_prev = vmap_g_prev;
	icp.nmap_g_prev = nmap_g_prev;
	icp.distThres = distThres;
	icp.angleThres = angleThres;
	icp.cols = cols;
	icp.rows = rows;
	icp.N = cols * rows;
	icp.out = sum;*/
	/*icpKernel << <blocks, threads >> >(icp);
	reduceSum << <1, MAX_THREADS >> >(sum, out, blocks);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());*/

	ocl* cl = ocl::GetInstance();
	DeviceArray<mat33> Rcurr0;
	DeviceArray<mat33> Rprev_inv0;
	Rcurr0.upload(&Rcurr, 1);
	Rprev_inv0.upload(&Rprev_inv, 1);
	DeviceArray<CameraModel> intr0;
	intr0.upload(&intr, 1);

	size_t local_worksize = threads;
	size_t global_worksize = blocks;
	OclKernel *kernel = cl->buildKernel("./Cuda/reduce.cl", "icpKernel", " ");
	kernel->Input(Rcurr0.getDeviceMemory());
	kernel->Input(tcurr);
	kernel->Input(vmap_curr.getDeviceMemory());
	kernel->Input(nmap_curr.getDeviceMemory());
	kernel->Input(Rprev_inv0.getDeviceMemory());
	kernel->Input(tprev);
	kernel->Input(intr0.getDeviceMemory());
	kernel->Input(vmap_g_prev.getDeviceMemory());
	kernel->Input(nmap_g_prev.getDeviceMemory());
	kernel->Input(distThres);
	kernel->Input(angleThres);
	kernel->Output(out.getDeviceMemory());
	kernel->Input(cols);
	kernel->Input(rows);	
	kernel->Run1D(global_worksize, local_worksize);
	delete kernel; kernel = NULL;
	//����kernel ִ�д��󣬵�������error

	OclKernel *kernel1 = cl->buildKernel("./Cuda/reduce.cl", "reduceSum_kernel", " ");
	kernel1->Input(sum.getDeviceMemory());
	kernel1->Input(out.getDeviceMemory());
	kernel1->Input(blocks);
	local_worksize = MAX_THREADS;
	global_worksize = 1;
	kernel1->Run1D(global_worksize, local_worksize);
	delete kernel1; kernel1 = NULL;
	cl->Synchronize();

	float host_data[32];
	out.download((JtJJtrSE3*)&host_data[0]);

	int shift = 0;
	for (int i = 0; i < 6; ++i)
	{
		for (int j = i; j < 7; ++j)
		{
			float value = host_data[shift++];
			if (j == 6)
				vectorB_host[i] = value;
			else
				matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
		}
	}

	residual_host[0] = host_data[27];
	residual_host[1] = host_data[28];
}
#endif
