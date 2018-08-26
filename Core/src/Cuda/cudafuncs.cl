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
 


float3 normalized(float3 v)
{
	float res;
	return res;
}

__kernel void pyrDownGaussKernel(__global unsigned short* src, __global unsigned short* dst, const int src_width, const int src_height,
	const int dst_width, const int dst_height,const float sigma_color)
{
    int x = get_global_id(0);
    int y = get_global_id(1);	 
	 
   if (x >= dst_width || y >= dst_height)
        return;

    const int D = 5;

    int center = src[2 * y*src_width +2 * x];

    int x_mi = max(0, 2*x - D/2) - 2*x;
    int y_mi = max(0, 2*y - D/2) - 2*y;

    int x_ma = min(src_width, 2*x -D/2+D) - 2*x;
    int y_ma = min(src_height, 2*y -D/2+D) - 2*y;

    float sum = 0;
    float wall = 0;

    float weights[] = {0.375f, 0.25f, 0.0625f} ;

    for(int yi = y_mi; yi < y_ma; ++yi)
        for(int xi = x_mi; xi < x_ma; ++xi)
        {
            int val = src[(2*y + yi)*dst_width+(2*x + xi)];

            if (abs (val - center) < 3 * sigma_color)
            {
                sum += val * weights[abs(xi)] * weights[abs(yi)];
                wall += weights[abs(xi)] * weights[abs(yi)];
            }
        }


    dst[y*dst_width +x] = (sum /wall);
}


__kernel void computeVmapKernel(__global const unsigned short* depth, __global float* vmap,int depth_w,int depth_h,int vmap_w,int vmap_h,
	float fx_inv, float fy_inv, float cx, float cy, float depthCutoff)
{
	int u = get_global_id(0);
	int v = get_global_id(1);
    
    if(u < depth_w && v < depth_h)
    {
        float z = depth[v*depth_w+u] / 1000.f; // load and convert: mm -> meters

        if(z != 0 && z < depthCutoff)
        {
            float vx = z * (u - cx) * fx_inv;
            float vy = z * (v - cy) * fy_inv;
            float vz = z;

            vmap[v*vmap_w+u] = vx;
            vmap[(v + depth_w)*vmap_w+u] = vy;
            vmap[(v + depth_w*2)*vmap_w+u] = vz;
        }
        else
        {
            vmap[v*vmap_w+u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        }
    }
}


__kernel void computeNmapKernel(int rows, int cols, __global const float* vmap, __global float* nmap,int vmap_w,int vmap_h,
								int nmap_w,int nmap_h)
{
	int u = get_global_id(0);
	int v = get_global_id(1);

    if (u >= cols || v >= rows)
        return;

    if (u == cols - 1 || v == rows - 1)
    {
        nmap[v*nmap_w+u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        return;
    }

    float3 v00, v01, v10;
    v00.x = vmap[v*vmap_w +u];
    v01.x = vmap[v*vmap_w+u + 1];
    v10.x = vmap[(v + 1)*vmap_w+u];

    if (!isnan (v00.x) && !isnan (v01.x) && !isnan (v10.x))
    {
        v00.y = vmap[(v + rows)*vmap_w+u];
        v01.y = vmap[(v + rows)*vmap_w+u + 1];
        v10.y = vmap[(v + 1 + rows)*vmap_w+u];

        v00.z = vmap[(v + 2 * rows)*vmap_w+u];
        v01.z = vmap[(v + 2 * rows)*vmap_w+u + 1];
        v10.z = vmap[(v + 1 + 2 * rows)*vmap_w+u];

        float3 r = normalized (cross (v01 - v00, v10 - v00));

        nmap[v*nmap_w +u] = r.x;
        nmap[(v + rows)*nmap_w +u] = r.y;
        nmap[(v + 2 * rows)*nmap_w +u] = r.z;
    }
    else
        nmap[v*nmap_w+u] = /*__int_as_float*/(0x7fffffff); /*CUDART_NAN_F*/
}

__kernel void tranformMapsKernel(int rows, int cols, __global const float* vmap_src, __global const float* nmap_src,
								__global const float3* Rmat, const float3 tvec, __global float* vmap_dst, __global float* nmap_dst,
								int vmap_src_w,int vmap_src_h,int nmap_src_w,int nmap_src_h,
								int vmap_dst_w,int vmap_dst_h,int nmap_dst_w,int nmap_dst_h)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        //vertexes
		float3 vsrc, vdst;
		vsrc.x = __int_as_float(0x7fffffff);
		vsrc.y = __int_as_float(0x7fffffff);
		vsrc.z = __int_as_float(0x7fffffff);
		vdst.x = __int_as_float(0x7fffffff);
		vdst.y = __int_as_float(0x7fffffff);
		vdst.z = __int_as_float(0x7fffffff);

		 
        vsrc.x = vmap_src[y*vmap_src_w+x];

        if (!isnan (vsrc.x))
        {
            vsrc.y = vmap_src[ (y + rows)*vmap_src_w+x];
            vsrc.z = vmap_src[ (y + 2 * rows)*vmap_src_w+x];

            vdst = Rmat * vsrc + tvec;

            vmap_dst[(y + rows)*vmap_dst_w+x] = vdst.y;
            vmap_dst[ (y + 2 * rows)*vmap_dst_w+x] = vdst.z;
        }

        vmap_dst[y*vmap_dst_w+x] = vdst.x;

        //normals
		float3 nsrc, ndst;
		nsrc.x = __int_as_float(0x7fffffff);
		nsrc.y = __int_as_float(0x7fffffff);
		nsrc.z = __int_as_float(0x7fffffff);
		ndst.x = __int_as_float(0x7fffffff);
		ndst.y = __int_as_float(0x7fffffff);
		ndst.z = __int_as_float(0x7fffffff);
		 
        nsrc.x = nmap_src[y*nmap_src_w+x];

        if (!isnan (nsrc.x))
        {
			nsrc.y = nmap_src[(y + rows)*nmap_src_w+x];
            nsrc.z = nmap_src[(y + 2 * rows)*nmap_src_w+x];

            ndst = Rmat * nsrc;

            nmap_dst[(y + rows)*nmap_dst_w+x] = ndst.y;
            nmap_dst[(y + 2 * rows)*nmap_dst_w+x] = ndst.z;
        }

        nmap_dst[y*nmap_dst_w+x] = ndst.x;
    }
}


__kernel void copyMapsKernel(int rows, int cols, __global const float * vmap_src, __global const float * nmap_src,
	__global float* vmap_dst, __global float* nmap_dst,int vmap_dst_w,int vmap_dst_h,int nmap_dst_w,int nmap_dst_h)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        //vertexes
		float3 vsrc, vdst;
		vsrc.x = __int_as_float(0x7fffffff);
		vsrc.y = __int_as_float(0x7fffffff);
		vsrc.z = __int_as_float(0x7fffffff);
		vdst.x = __int_as_float(0x7fffffff);
		vdst.y = __int_as_float(0x7fffffff);
		vdst.z = __int_as_float(0x7fffffff);

        vsrc.x = vmap_src[y * cols * 4 + (x * 4) + 0];
        vsrc.y = vmap_src[y * cols * 4 + (x * 4) + 1];
        vsrc.z = vmap_src[y * cols * 4 + (x * 4) + 2];

        if(!(vsrc.z == 0))
        {
            vdst = vsrc;
        }

        vmap_dst[y*vmap_dst_w+x] = vdst.x;
        vmap_dst[(y + rows)*vmap_dst_w+x] = vdst.y;
        vmap_dst[(y + 2 * rows)*vmap_dst_w+x] = vdst.z;

        //normals
		float3 nsrc, ndst;
		nsrc.x = __int_as_float(0x7fffffff);
		nsrc.y = __int_as_float(0x7fffffff);
		nsrc.z = __int_as_float(0x7fffffff);
		ndst.x = __int_as_float(0x7fffffff);
		ndst.y = __int_as_float(0x7fffffff);
		ndst.z = __int_as_float(0x7fffffff);

        nsrc.x = nmap_src[y * cols * 4 + (x * 4) + 0];
        nsrc.y = nmap_src[y * cols * 4 + (x * 4) + 1];
        nsrc.z = nmap_src[y * cols * 4 + (x * 4) + 2];

        if(!(vsrc.z == 0))
        {
            ndst = nsrc;
        }

        nmap_dst[(y)*nmap_dst_w+x] = ndst.x;
        nmap_dst[(y + rows)*nmap_dst_w+x] = ndst.y;
        nmap_dst[(y + 2 * rows)*nmap_dst_w+x] = ndst.z;
    }
}


__kernel void pyrDownKernelGaussF(__global const float* src, __global float* dst, __global float* gaussKernel,
								int src_w,int src_h,int dst_w,int dst_h)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

    if (x >= dst_w || y >= dst_h)
        return;

    const int D = 5;

    float center = src[(2 * y)*src_w+2 * x];

    int tx = min (2 * x - D / 2 + D, src_w - 1);
    int ty = min (2 * y - D / 2 + D, src_h - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
    {
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            if(!isnan(src[cy*src_w+cx]))
            {
                sum += src[cy*src_w+cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    }
    dst[y*dst_w+x] = (float)(sum / (float)count);
}
 
__kernel void resizeMapKernel(int drows, int dcols, int srows, __global const float* input, __global float* output,
							int in_w,int in_h,int out_w,int out_h, int normalize)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

    if (x >= dcols || y >= drows)
        return;

    const float qnan = __int_as_float(0x7fffffff);

    int xs = x * 2;
    int ys = y * 2;

    float x00 = input[(ys + 0)*in_w+xs + 0];
    float x01 = input[(ys + 0)*in_w+xs + 1];
    float x10 = input[(ys + 1)*in_w+xs + 0];
    float x11 = input[(ys + 1)*in_w+xs + 1];

    if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
    {
        output[y*out_w+x] = qnan;
        return;
    }
    else
    {
        float3 n;

        n.x = (x00 + x01 + x10 + x11) / 4;

        float y00 = input[(ys + srows + 0)*in_w+xs + 0];
        float y01 = input[(ys + srows + 0)*in_w+xs + 1];
        float y10 = input[(ys + srows + 1)*in_w+xs + 0];
        float y11 = input[(ys + srows + 1)*in_w+xs + 1];

        n.y = (y00 + y01 + y10 + y11) / 4;

        float z00 = input[ (ys + 2 * srows + 0)*in_w+xs + 0];
        float z01 = input[ (ys + 2 * srows + 0)*in_w+xs + 1];
        float z10 = input[ (ys + 2 * srows + 1)*in_w+xs + 0];
        float z11 = input[ (ys + 2 * srows + 1)*in_w+xs + 1];

        n.z = (z00 + z01 + z10 + z11) / 4;

        if (normalize)
            n = normalized (n);

        output[ (y        )*out_w+x] = n.x;
        output[ (y + drows)*out_w+x] = n.y;
        output[ (y + 2 * drows)*out_w+x] = n.z;
    }
}



__kernel void pyrDownKernelIntensityGauss(__global const unsigned char* src, __global unsigned char* dst, __global float * gaussKernel,
										int src_w,int src_h,int dst_w,int dst_h,int gaussKernel_w,int gaussKernel_h)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

    if (x >= dst_w || y >= dst_h)
        return;

    const int D = 5;

    int center = src[(2 * y)*src_w+2 * x];

    int tx = min (2 * x - D / 2 + D, src_w - 1);
    int ty = min (2 * y - D / 2 + D, src_w - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            //This might not be right, but it stops incomplete model images from making up colors
            if(src[(cy)*src_w+cx] > 0)
            {
                sum += src[(cy)*src_w+cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    dst[y*dst_w+x] = (sum / (float)count);
}


__kernel void verticesToDepthKernel(__global const float * vmap_src, __global float* dst, float cutOff,
									int dst_w,int dst_h)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

    if (x >= dst_w || y >= dst_h)
        return;

    float z = vmap_src[y *dst_w * 4 + (x * 4) + 2];

    dst[y*dst_w+x] = z > cutOff || z <= 0 ? __int_as_float(0x7fffffff)/*CUDART_NAN_F*/ : z;
}


//
//texture<uchar4, 2, cudaReadModeElementType> inTex;
//
__kernel void bgr2IntensityKernel(__global uchar4* src,__global unsigned char* dst,
									int src_w,int src_h,int dst_w,int dst_h)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

    if (x >= dst_w || y >= dst_h)
        return;

    uchar4 pixel = src[y*src_w +x];

    int value = (float)pixel.x * 0.114f + (float)pixel.y * 0.299f + (float)pixel.z * 0.587f;

    dst[y*dst_w + x] = value;
}


constant float gsobel_x3x3[9] = { 0.52201,  0.00000, -0.52201,
								0.79451, -0.00000, -0.79451,
								0.52201,  0.00000, -0.52201 };
constant float gsobel_y3x3[9] = { 0.52201, 0.79451, 0.52201,
								0.00000, 0.00000, 0.00000,
								-0.52201, -0.79451, -0.52201 };

__kernel void applyKernel(__global const unsigned char* src, __global short* dx, __global short* dy,
						int src_w,int src_h,int dx_w,int dx_y,int dy_w,int dy_h)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

  if(x >= src_w || y >= src_h)
    return;

  float dxVal = 0;
  float dyVal = 0;

  int kernelIndex = 8;
  for(int j = max(y - 1, 0); j <= min(y + 1, src_h - 1); j++)
  {
      for(int i = max(x - 1, 0); i <= min(x + 1, src_w - 1); i++)
      {
          dxVal += (float)src[j*src_w+i] * gsobel_x3x3[kernelIndex];
          dyVal += (float)src[j*src_w+i] * gsobel_y3x3[kernelIndex];
          --kernelIndex;
      }
  }

  dx[(y)*dx_w+x] = dxVal;
  dy[(y)*dy_w+x] = dyVal;
}


__kernel void projectPointsKernel(__global const float* depth,
									__global float3* cloud,
                                    const float invFx,
                                    const float invFy,
                                    const float cx,
                                    const float cy,
									int depth_x,int depth_y,int cloud_w,int cloud_h)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

    if (x >= depth_x || y >= depth_y)
        return;

    float z = depth[y*depth_x+x];

    cloud[y*cloud_w+x].x = (float)((x - cx) * z * invFx);
    cloud[y*cloud_w+x].y = (float)((y - cy) * z * invFy);
    cloud[y*cloud_w+x].z = z;
}


