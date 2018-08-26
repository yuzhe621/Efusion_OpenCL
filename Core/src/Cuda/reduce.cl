typedef struct JtJJtrSO3_
{
	//9 floats for each product (9)
	float aa, ab, ac, ad,
		bb, bc, bd,
		cc, cd;

	//Extra data needed (11)
	float residual, inliers;

	/* __device__ inline void add(const JtJJtrSO3 & a)
	{
	aa += a.aa;
	ab += a.ab;
	ac += a.ac;
	ad += a.ad;

	bb += a.bb;
	bc += a.bc;
	bd += a.bd;

	cc += a.cc;
	cd += a.cd;

	residual += a.residual;
	inliers += a.inliers;
	}*/
}JtJJtrSO3;

typedef struct JtJJtrSE3_
{
	//27 floats for each product (27)
	float aa, ab, ac, ad, ae, af, ag,
		bb, bc, bd, be, bf, bg,
		cc, cd, ce, cf, cg,
		dd, de, df, dg,
		ee, ef, eg,
		ff, fg;

	//Extra data needed (29)
	float residual, inliers;

	/* __device__ inline void add(const JtJJtrSE3 & a)
	{
	aa += a.aa;
	ab += a.ab;
	ac += a.ac;
	ad += a.ad;
	ae += a.ae;
	af += a.af;
	ag += a.ag;

	bb += a.bb;
	bc += a.bc;
	bd += a.bd;
	be += a.be;
	bf += a.bf;
	bg += a.bg;

	cc += a.cc;
	cd += a.cd;
	ce += a.ce;
	cf += a.cf;
	cg += a.cg;

	dd += a.dd;
	de += a.de;
	df += a.df;
	dg += a.dg;

	ee += a.ee;
	ef += a.ef;
	eg += a.eg;

	ff += a.ff;
	fg += a.fg;

	residual += a.residual;
	inliers += a.inliers;
	}*/
}JtJJtrSE3;

typedef struct mat33_
{
	//    mat33() {}
	//
	//#if !defined(__CUDACC__)
	//    mat33(Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & e)
	//    {
	//        memcpy(data, e.data(), sizeof(mat33));
	//    }
	//#endif
	float3 data[3];
}mat33;

typedef struct SO3Reduction_
{
	__global unsigned char* lastImage;
	__global unsigned char* nextImage;

	mat33 imageBasis;
	mat33 kinv;
	mat33 krlr;
	bool gradCheck;

	int cols;
	int rows;
	int N;

	global JtJJtrSO3* out;
}SO3Reduction;

float3 make_float3(const float x, const float y, const float z)
{
	float3 value;
	value.x = x;
	value.y = y;
	value.z = z;
	return value;
}


float3 vector_minus(const float3 a, const float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 vector_add(const float3 a, const float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


float3 cross1(const float3 a, const float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float dot1(const float3 a, const float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

float norm1(const float3 a)
{
	return sqrt(dot1(a, a));
}

float3 normalized1(const float3 a)
{
	const float rn = sqrt(dot(a, a));
	return make_float3(a.x * rn, a.y * rn, a.z * rn);
}

float3 multiply_mat33_float3(mat33 m, float3 a)
{
	return make_float3(dot(m.data[0], a), dot(m.data[1], a), dot(m.data[2], a));
}


float2 getGradient(global unsigned char* img, int x, int y, int img_w, int img_h)
{
	float2 gradient;

	float actu = (float)(img[y*img_w + x]);

	float back = (float)(img[y*img_w + x - 1]);
	float fore = (float)(img[y*img_w + x + 1]);
	gradient.x = ((back + actu) / 2.0f) - ((fore + actu) / 2.0f);

	back = (float)(img[(y - 1)*img_w + x]);
	fore = (float)(img[(y + 1)*img_w + x]);
	gradient.y = ((back + actu) / 2.0f) - ((fore + actu) / 2.0f);

	return gradient;
}

JtJJtrSO3 getProducts(SO3Reduction* so3, int k)
{
	int cols = so3->cols;
	int rows = so3->rows;
	int y = k / cols;
	int x = k - (y * cols);

	bool found_coresp = false;

	mat33 imageBasis = so3->imageBasis;
	float3 unwarpedReferencePoint = { x, y, 1.0f };
	float3 warpedReferencePoint = multiply_mat33_float3(imageBasis, unwarpedReferencePoint);

	int2 warpedReferencePixel = { /*__float2int_rn*/(warpedReferencePoint.x / warpedReferencePoint.z),
		/*__float2int_rn*/(warpedReferencePoint.y / warpedReferencePoint.z) };

	if (warpedReferencePixel.x >= 1 &&
		warpedReferencePixel.x < cols - 1 &&
		warpedReferencePixel.y >= 1 &&
		warpedReferencePixel.y < rows - 1 &&
		x >= 1 &&
		x < cols - 1 &&
		y >= 1 &&
		y < rows - 1)
	{
		found_coresp = true;
	}

	float row[4];
	row[0] = row[1] = row[2] = row[3] = 0.f;

	if (found_coresp)
	{
		global unsigned char* lastImage = so3->lastImage;
		global unsigned char* nextImage = so3->nextImage;
		float2 gradNext = getGradient(nextImage, warpedReferencePixel.x, warpedReferencePixel.y, cols, rows);///???
		float2 gradLast = getGradient(lastImage, x, y, cols, rows);//????

		float gx = (gradNext.x + gradLast.x) / 2.0f;
		float gy = (gradNext.y + gradLast.y) / 2.0f;

		mat33 kinv = so3->kinv;
		float3 point = multiply_mat33_float3(kinv, unwarpedReferencePoint);

		float z2 = point.z * point.z;


		mat33 krlr = so3->krlr;

		float a = krlr.data[0].x;
		float b = krlr.data[0].y;
		float c = krlr.data[0].z;

		float d = krlr.data[1].x;
		float e = krlr.data[1].y;
		float f = krlr.data[1].z;

		float g = krlr.data[2].x;
		float h = krlr.data[2].y;
		float i = krlr.data[2].z;

		//Aren't jacobians great fun
		float3 leftProduct = { ((point.z * (d * gy + a * gx)) - (gy * g * y) - (gx * g * x)) / z2,
			((point.z * (e * gy + b * gx)) - (gy * h * y) - (gx * h * x)) / z2,
			((point.z * (f * gy + c * gx)) - (gy * i * y) - (gx * i * x)) / z2 };

		float3 jacRow = cross1(leftProduct, point);

		row[0] = jacRow.x;
		row[1] = jacRow.y;
		row[2] = jacRow.z;
		row[3] = -((float)(nextImage[warpedReferencePixel.y*cols + warpedReferencePixel.x]) - (float)(lastImage[y*cols + x]));///???
	}

	JtJJtrSO3 values = { row[0] * row[0],
		row[0] * row[1],
		row[0] * row[2],
		row[0] * row[3],
		row[1] * row[1],
		row[1] * row[2],
		row[1] * row[3],
		row[2] * row[2],
		row[2] * row[3],
		row[3] * row[3],
		(float)found_coresp };

	return values;
}



float __shfl_down_float(float val, int offset)//, int width  = 32)
{
	int width = 32;
	/*static __shared__ float shared[MAX_THREADS];
	int lane = threadIdx.x % 32;
	shared[threadIdx.x] = val;
	__syncthreads();
	val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
	__syncthreads();
	return val;*/

	int idx = get_local_id(0);
	__local float shared_mem[512];
	int lane = idx % 32;
	shared_mem[idx] = val;
	barrier(CLK_LOCAL_MEM_FENCE);
	val = (lane + offset < width) ? shared_mem[idx + offset] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	return val;
}
int __shfl_down_int(int val, int offset)//, int width  = 32)
{
	int width = 32;
	/*static __shared__ float shared[MAX_THREADS];
	int lane = threadIdx.x % 32;
	shared[threadIdx.x] = val;
	__syncthreads();
	val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
	__syncthreads();
	return val;*/

	int idx = get_local_id(0);
	__local float shared_mem[512];
	int lane = idx % 32;
	shared_mem[idx] = val;
	barrier(CLK_LOCAL_MEM_FENCE);
	val = (lane + offset < width) ? shared_mem[idx + offset] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	return val;
}


//template<typename T>
// T __ldg(__global const T* ptr)
//{
//	return *ptr;
//}

float __ldg_float(float* ptr)
{
	return *ptr;
}

JtJJtrSE3 warpReduceSum_se3(JtJJtrSE3 val)
{
	int warpSize = 32;
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		val.aa += __shfl_down_float(val.aa, offset);
		val.ab += __shfl_down_float(val.ab, offset);
		val.ac += __shfl_down_float(val.ac, offset);
		val.ad += __shfl_down_float(val.ad, offset);
		val.ae += __shfl_down_float(val.ae, offset);
		val.af += __shfl_down_float(val.af, offset);
		val.ag += __shfl_down_float(val.ag, offset);

		val.bb += __shfl_down_float(val.bb, offset);
		val.bc += __shfl_down_float(val.bc, offset);
		val.bd += __shfl_down_float(val.bd, offset);
		val.be += __shfl_down_float(val.be, offset);
		val.bf += __shfl_down_float(val.bf, offset);
		val.bg += __shfl_down_float(val.bg, offset);

		val.cc += __shfl_down_float(val.cc, offset);
		val.cd += __shfl_down_float(val.cd, offset);
		val.ce += __shfl_down_float(val.ce, offset);
		val.cf += __shfl_down_float(val.cf, offset);
		val.cg += __shfl_down_float(val.cg, offset);

		val.dd += __shfl_down_float(val.dd, offset);
		val.de += __shfl_down_float(val.de, offset);
		val.df += __shfl_down_float(val.df, offset);
		val.dg += __shfl_down_float(val.dg, offset);

		val.ee += __shfl_down_float(val.ee, offset);
		val.ef += __shfl_down_float(val.ef, offset);
		val.eg += __shfl_down_float(val.eg, offset);

		val.ff += __shfl_down_float(val.ff, offset);
		val.fg += __shfl_down_float(val.fg, offset);

		val.residual += __shfl_down_float(val.residual, offset);
		val.inliers += __shfl_down_float(val.inliers, offset);
	}

	return val;
}

JtJJtrSO3 warpReduceSum_so3(JtJJtrSO3 val)
{
	int warpSize = 32;
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		val.aa += __shfl_down_float(val.aa, offset);
		val.ab += __shfl_down_float(val.ab, offset);
		val.ac += __shfl_down_float(val.ac, offset);
		val.ad += __shfl_down_float(val.ad, offset);

		val.bb += __shfl_down_float(val.bb, offset);
		val.bc += __shfl_down_float(val.bc, offset);
		val.bd += __shfl_down_float(val.bd, offset);

		val.cc += __shfl_down_float(val.cc, offset);
		val.cd += __shfl_down_float(val.cd, offset);

		val.residual += __shfl_down_float(val.residual, offset);
		val.inliers += __shfl_down_float(val.inliers, offset);
	}

	return val;
}

JtJJtrSO3 blockReduceSum_so3(JtJJtrSO3 val)
{
	//static __shared__ JtJJtrSO3 shared[32];
	__local JtJJtrSO3 shared_mem[32];
	int idx = get_local_id(0);
	int size_x = get_local_size(0);
	int warpSize = 32;

	int lane = idx % warpSize;
	int wid = idx / warpSize;
	val = warpReduceSum_so3(val);

	//write reduced value to shared memory
	if (lane == 0)
	{
		shared_mem[wid] = val;
	}
	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE);

	JtJJtrSO3 zero = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	//ensure we only grab a value from shared memory if that warp existed
	//val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;
	val = idx < size_x / warpSize ? shared_mem[lane] : zero;

	if (wid == 0)
	{
		val = warpReduceSum_so3(val);
	}

	return val;
}

JtJJtrSE3 blockReduceSum_se3(JtJJtrSE3 val)
{
	//static _shared_ JtJJtrSE3 shared[32];
	__local JtJJtrSE3 shared_mem[32];

	int idx = get_local_id(0);
	int size_x = get_local_size(0);
	int warpSize = 32;
	int lane = idx % warpSize;

	int wid = idx / warpSize;

	val = warpReduceSum_se3(val);

	//write reduced value to shared memory
	if (lane == 0)
	{
		//shared[wid] = val;
		shared_mem[wid] = val;
	}
	//__syncthreads();
	barrier(CLK_GLOBAL_MEM_FENCE);

	const JtJJtrSE3 zero = { 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	//ensure we only grab a value from shared memory if that warp existed
	//val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;
	val = idx<size_x / warpSize ? shared_mem[lane] : zero;

	if (wid == 0)
	{
		val = warpReduceSum_se3(val);
	}

	return val;
}


void JtJJtrSO3_Add(JtJJtrSO3 sum, JtJJtrSO3 value)
{
	sum.aa += value.aa;
	sum.ab += value.ab;
	sum.ac += value.ac;
	sum.ad += value.ad;
	sum.bb += value.bb;
	sum.bc += value.bc;
	sum.bd += value.bd;
	sum.cc += value.cc;
	sum.cd += value.cd;
	sum.residual += value.residual;
	sum.inliers += value.inliers;
}

void JtJJtrSO3_equal(JtJJtrSO3 sum, JtJJtrSO3 value)
{
	sum.aa = value.aa;
	sum.ab = value.ab;
	sum.ac = value.ac;
	sum.ad = value.ad;
	sum.bb = value.bb;
	sum.bc = value.bc;
	sum.bd = value.bd;
	sum.cc = value.cc;
	sum.cd = value.cd;
	sum.residual = value.residual;
	sum.inliers = value.inliers;
}


void so3Kernel_(SO3Reduction* so3)
{
	JtJJtrSO3 sum = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int N = so3->N;
	//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	for (int i = get_global_id(0); i < N; i += get_global_size(0))
	{
		JtJJtrSO3 val = getProducts(so3, i);
		JtJJtrSO3_Add(sum, val);
	}

	sum = blockReduceSum_so3(sum);
	////if (threadIdx.x == 0)
	if (get_local_id(0) == 0)
	{
		//out[blockIdx.x] = sum;z
		//so3->out[get_group_id(0)] = sum;	
		JtJJtrSO3_equal(so3->out[get_group_id(0)], sum);
	}
}

//--------------------------------------
__kernel void so3Kernel(__global unsigned char* lastImage, __global unsigned char* nextImage, __global mat33* imageBasis,
	__global mat33* kinv, __global mat33* krlr, int gradCheck, int cols, int rows, int N, __global JtJJtrSO3* out)
{
	SO3Reduction so3;
	so3.lastImage = lastImage;
	so3.nextImage = nextImage;
	so3.imageBasis = imageBasis[0];
	so3.kinv = kinv[0];
	so3.krlr = krlr[0];
	so3.gradCheck = gradCheck;
	so3.cols = cols;
	so3.rows = rows;
	so3.out = out;
	so3Kernel_(&so3);
}



//-----------------------------
void se3_add(JtJJtrSE3 res, const JtJJtrSE3 a)
{
	res.aa += a.aa;
	res.ab += a.ab;
	res.ac += a.ac;
	res.ad += a.ad;
	res.ae += a.ae;
	res.af += a.af;
	res.ag += a.ag;

	res.bb += a.bb;
	res.bc += a.bc;
	res.bd += a.bd;
	res.be += a.be;
	res.bf += a.bf;
	res.bg += a.bg;

	res.cc += a.cc;
	res.cd += a.cd;
	res.ce += a.ce;
	res.cf += a.cf;
	res.cg += a.cg;

	res.dd += a.dd;
	res.de += a.de;
	res.df += a.df;
	res.dg += a.dg;

	res.ee += a.ee;
	res.ef += a.ef;
	res.eg += a.eg;

	res.ff += a.ff;
	res.fg += a.fg;

	res.residual += a.residual;
	res.inliers += a.inliers;
}

void se3_equal(JtJJtrSE3 res, const JtJJtrSE3 a)
{
	res.aa = a.aa;
	res.ab = a.ab;
	res.ac = a.ac;
	res.ad = a.ad;
	res.ae = a.ae;
	res.af = a.af;
	res.ag = a.ag;

	res.bb = a.bb;
	res.bc = a.bc;
	res.bd = a.bd;
	res.be = a.be;
	res.bf = a.bf;
	res.bg = a.bg;

	res.cc = a.cc;
	res.cd = a.cd;
	res.ce = a.ce;
	res.cf = a.cf;
	res.cg = a.cg;

	res.dd = a.dd;
	res.de = a.de;
	res.df = a.df;
	res.dg = a.dg;

	res.ee = a.ee;
	res.ef = a.ef;
	res.eg = a.eg;

	res.ff = a.ff;
	res.fg = a.fg;

	res.residual = a.residual;
	res.inliers = a.inliers;
}

__kernel void reduceSum_kernel(__global JtJJtrSE3* input, __global JtJJtrSE3 * output, int N)
{
	JtJJtrSE3 sum = { 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	int idx = get_local_id(0);
	int size_x = get_local_size(0);
	int group_id = get_group_id(0);

	///*for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	//{
	//	sum.add(in[i]);
	//}*/
	// 

	//int local_size_ = get_local_size(0);
	int incre = get_global_size(0);
	for (int i = get_group_id(0); i<N; i += incre)
	{
		se3_add(sum, input[i]);
	}

	sum = blockReduceSum_se3(sum);

	///*if (threadIdx.x == 0)	
	//{
	//	out[blockIdx.x] = sum;
	//}*/
	if (idx == 0)
	{
		output[group_id] = sum;
	}
}


//-------------------------------------------------
typedef struct CameraModel_
{
	float fx, fy, cx, cy;
}CameraModel;

CameraModel CameraModel_level(CameraModel model, int level)
{
	CameraModel res;
	res.fx = model.fx / level;
	res.fx = model.fy / level;
	res.fx = model.cx / level;
	res.fx = model.cy / level;
	return res;
}

typedef struct ICPReduction_
{
	mat33 Rcurr;
	float3 tcurr;

	__global float* vmap_curr;
	__global float* nmap_curr;

	mat33 Rprev_inv;
	float3 tprev;

	CameraModel intr;

	__global float* vmap_g_prev;
	__global float* nmap_g_prev;

	float distThres;
	float angleThres;

	int cols;
	int rows;
	int N;

	__global JtJJtrSE3 * out;
}ICPReduction;

bool search(ICPReduction* icp, int x, int y, float3 n, float3 d, float3 s)
{
	__global float* vmap_curr = icp->vmap_curr;
	int cols = icp->cols;
	int rows = icp->rows;
	mat33 Rcurr = icp->Rcurr;
	float3 tcurr = icp->tcurr;
	float3 tprev = icp->tprev;
	mat33 Rprev_inv = icp->Rprev_inv;
	CameraModel intr = icp->intr;
	__global float* vmap_g_prev = icp->vmap_g_prev;
	__global float* nmap_curr = icp->nmap_curr;
	__global float* nmap_g_prev = icp->nmap_g_prev;
	float distThres = icp->distThres;
	float angleThres = icp->angleThres;

	float3 vcurr;
	vcurr.x = vmap_curr[y*cols + x];
	vcurr.y = vmap_curr[(y + rows)*cols + x];
	vcurr.z = vmap_curr[(y + 2 * rows)*cols + x];

	float3 vcurr_g = vector_add(multiply_mat33_float3(Rcurr, vcurr), tcurr);
	float3 vcurr_cp = multiply_mat33_float3(Rprev_inv, vector_minus(vcurr_g, tprev));

	int2 ukr;
	ukr.x = /*__float2int_rn*/(vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);
	ukr.y = /*__float2int_rn*/(vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);

	if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0)
		return false;

	float3 vprev_g;
	/*vprev_g.x = __ldg_float(&vmap_g_prev[(ukr.y)*cols+ukr.x]);
	vprev_g.y = __ldg_float(&vmap_g_prev[(ukr.y + rows)*cols+ukr.x]);
	vprev_g.z = __ldg_float(&vmap_g_prev[(ukr.y + 2 * rows)*cols+ukr.x]);*/
	vprev_g.x = vmap_g_prev[(ukr.y)*cols + ukr.x];
	vprev_g.y = vmap_g_prev[(ukr.y + rows)*cols + ukr.x];
	vprev_g.z = vmap_g_prev[(ukr.y + 2 * rows)*cols + ukr.x];

	float3 ncurr;
	ncurr.x = nmap_curr[(y)*cols + x];
	ncurr.y = nmap_curr[(y + rows)*cols + x];
	ncurr.z = nmap_curr[(y + 2 * rows)*cols + x];

	float3 ncurr_g = multiply_mat33_float3(Rcurr, ncurr);

	float3 nprev_g;
	/*nprev_g.x = __ldg(&nmap_g_prev[(ukr.y)*cols+ukr.x]);
	nprev_g.y = __ldg(&nmap_g_prev[(ukr.y + rows)*cols+ukr.x]);
	nprev_g.z = __ldg(&nmap_g_prev[(ukr.y + 2 * rows)*cols+ukr.x]);*/
	nprev_g.x = nmap_g_prev[(ukr.y)*cols + ukr.x];
	nprev_g.y = nmap_g_prev[(ukr.y + rows)*cols + ukr.x];
	nprev_g.z = nmap_g_prev[(ukr.y + 2 * rows)*cols + ukr.x];

	float dist = norm1(vprev_g - vcurr_g);
	float sine = norm1(cross(ncurr_g, nprev_g));

	n = nprev_g;
	d = vprev_g;
	s = vcurr_g;

	return (sine < angleThres && dist <= distThres && !isnan(ncurr.x) && !isnan(nprev_g.x));
}

JtJJtrSE3 getProducts_IcpReduction(ICPReduction* icp, int i)
{
	int cols = icp->cols;
	mat33 Rcurr = icp->Rcurr;
	float3 tcurr = icp->tcurr;
	float3 tprev = icp->tprev;
	mat33 Rprev_inv = icp->Rprev_inv;
	CameraModel intr = icp->intr;
	__global float* vmap_g_prev = icp->vmap_g_prev;
	__global float* nmap_curr = icp->nmap_curr;

	int y = i / cols;
	int x = i - (y * cols);

	float3 n_cp, d_cp, s_cp;

	bool found_coresp = search(icp, x, y, n_cp, d_cp, s_cp);

	float row[7] = { 0, 0, 0, 0, 0, 0, 0 };

	if (found_coresp)
	{
		s_cp = multiply_mat33_float3(Rprev_inv, vector_minus(s_cp, tprev));
		d_cp = multiply_mat33_float3(Rprev_inv, vector_minus(d_cp, tprev));
		n_cp = multiply_mat33_float3(Rprev_inv, n_cp);

		*(float3*)&row[0] = n_cp;
		*(float3*)&row[3] = cross1(s_cp, n_cp);
		row[6] = dot1(n_cp, s_cp - d_cp);
	}

	JtJJtrSE3 values = { row[0] * row[0],
		row[0] * row[1],
		row[0] * row[2],
		row[0] * row[3],
		row[0] * row[4],
		row[0] * row[5],
		row[0] * row[6],

		row[1] * row[1],
		row[1] * row[2],
		row[1] * row[3],
		row[1] * row[4],
		row[1] * row[5],
		row[1] * row[6],

		row[2] * row[2],
		row[2] * row[3],
		row[2] * row[4],
		row[2] * row[5],
		row[2] * row[6],

		row[3] * row[3],
		row[3] * row[4],
		row[3] * row[5],
		row[3] * row[6],

		row[4] * row[4],
		row[4] * row[5],
		row[4] * row[6],

		row[5] * row[5],
		row[5] * row[6],

		row[6] * row[6],
		found_coresp };

	return values;
}
void icpKernel_(ICPReduction* icp)
{
	JtJJtrSE3 sum = { 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	int N = icp->N;
	//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	for (int i = get_global_id(0); i<N; i += get_global_size(0))
	{
		JtJJtrSE3 val = getProducts_IcpReduction(icp, i);

		//sum.add(val);
		se3_add(sum, val);
	}

	sum = blockReduceSum_se3(sum);

	/*if (threadIdx.x == 0)
	{
	out[blockIdx.x] = sum;
	}*/
	if (get_local_id(0) == 0)
	{
		se3_equal(icp->out[get_group_id(0)], sum);
	}
}

__kernel void icpKernel(__global mat33* Rcurr,
	float3 tcurr,
	__global float* vmap_curr,
	__global float* nmap_curr,
	__global mat33* Rprev_inv,
	float3 tprev,
	__global CameraModel* intr,
	__global float* vmap_g_prev,
	__global float* nmap_g_prev,
	float distThres,
	float angleThres,
	__global JtJJtrSE3* out,
	float cols,
	float rows)
{
	ICPReduction icp;
	icp.Rcurr = Rcurr[0];
	icp.tcurr = tcurr;

	icp.vmap_curr = vmap_curr;
	icp.nmap_curr = nmap_curr;

	icp.Rprev_inv = Rprev_inv[0];
	icp.tprev = tprev;

	icp.intr = intr[0];

	icp.vmap_g_prev = vmap_g_prev;
	icp.nmap_g_prev = nmap_g_prev;

	icp.distThres = distThres;
	icp.angleThres = angleThres;

	icp.cols = cols;
	icp.rows = rows;
	icp.N = cols*rows;

	icp.out = out;
	icpKernel_(&icp);
}



//------------------RGB kernel--------------------------------------------------------------------
typedef struct DataTerm_
{
	short2 zero;
	short2 one;
	float diff;
	bool valid;
}DataTerm;

typedef struct RGBReduction_
{
	__global DataTerm* corresImg;

	float sigma;
	__global float3* cloud;
	float fx;
	float fy;
	__global short* dIdx;
	__global short* dIdy;
	float sobelScale;

	int cols;
	int rows;
	int N;

	__global JtJJtrSE3* out;
}RGBReduction;

JtJJtrSE3 getProducts_rgbReduction(RGBReduction* rgb_reduce, int i)
{
	DataTerm corresp = rgb_reduce->corresImg[i];
	__global float3* cloud = rgb_reduce->cloud;
	bool found_coresp = corresp.valid;
	int cols = rgb_reduce->cols;
	int rows = rgb_reduce->rows;
	__global short* dIdx = rgb_reduce->dIdx;
	__global short* dIdy = rgb_reduce->dIdy;
	float sobelScale = rgb_reduce->sobelScale;
	float fx = rgb_reduce->fx;
	float fy = rgb_reduce->fy;

	float row[7];
	float sigma = rgb_reduce->sigma;
	if (found_coresp)
	{
		float w = sigma + fabs(corresp.diff);

		w = w > FLT_EPSILON ? 1.0f / w : 1.0f;

		//Signals RGB only tracking, so we should only
		if (sigma == -1)
		{
			w = 1;
		}

		row[6] = -w * corresp.diff;

		float3 cloudPoint = { cloud[corresp.zero.y*cols + corresp.zero.x].x,
			cloud[(corresp.zero.y)*cols + corresp.zero.x].y,
			cloud[(corresp.zero.y)*cols + corresp.zero.x].z };

		float invz = 1.0 / cloudPoint.z;
		float dI_dx_val = w * sobelScale * dIdx[(corresp.one.y)*cols + corresp.one.x];
		float dI_dy_val = w * sobelScale * dIdy[(corresp.one.y)*cols + corresp.one.x];
		float v0 = dI_dx_val * fx * invz;
		float v1 = dI_dy_val * fy * invz;
		float v2 = -(v0 * cloudPoint.x + v1 * cloudPoint.y) * invz;

		row[0] = v0;
		row[1] = v1;
		row[2] = v2;
		row[3] = -cloudPoint.z * v1 + cloudPoint.y * v2;
		row[4] = cloudPoint.z * v0 - cloudPoint.x * v2;
		row[5] = -cloudPoint.y * v0 + cloudPoint.x * v1;
	}
	else
	{
		row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;
	}

	JtJJtrSE3 values = { row[0] * row[0],
		row[0] * row[1],
		row[0] * row[2],
		row[0] * row[3],
		row[0] * row[4],
		row[0] * row[5],
		row[0] * row[6],

		row[1] * row[1],
		row[1] * row[2],
		row[1] * row[3],
		row[1] * row[4],
		row[1] * row[5],
		row[1] * row[6],

		row[2] * row[2],
		row[2] * row[3],
		row[2] * row[4],
		row[2] * row[5],
		row[2] * row[6],

		row[3] * row[3],
		row[3] * row[4],
		row[3] * row[5],
		row[3] * row[6],

		row[4] * row[4],
		row[4] * row[5],
		row[4] * row[6],

		row[5] * row[5],
		row[5] * row[6],

		row[6] * row[6],
		found_coresp };

	return values;
}

void rgbKernel_(RGBReduction* rgb_reduce)
{
	JtJJtrSE3 sum = { 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	int N = rgb_reduce->N;
	//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	for (int i = get_global_id(0); i<N; i += get_global_size(0))
	{
		JtJJtrSE3 val = getProducts_rgbReduction(rgb_reduce, i);

		se3_add(sum, val);
	}

	sum = blockReduceSum_se3(sum);

	//if (threadIdx.x == 0)
	if (get_local_id(0) == 0)
	{
		//out[blockIdx.x] = sum;
		se3_equal(rgb_reduce->out[get_group_id(0)], sum);
	}
}


__kernel void rgbKernel(__global DataTerm* corresImg, float sigma, __global float3* cloud, float fx,
	float fy, __global short* dIdx, __global short* dIdy, float sobelScale, __global JtJJtrSE3* out)
{
	RGBReduction rgb;
	rgb.corresImg = corresImg;
	rgb.sigma = sigma;
	rgb.cloud = cloud;
	rgb.fx = fx;
	rgb.fy = fy;
	rgb.dIdx = dIdx;
	rgb.dIdy = dIdy;
	rgb.sobelScale = sobelScale;
	rgb.out = out;


	rgbKernel_(&rgb);
}

//-------------------residual kernel--------------------------

typedef struct RGBResidual
{
	float minScale;

	__global short* dIdx;
	__global short* dIdy;

	__global float* lastDepth;
	__global float* nextDepth;

	__global unsigned char* lastImage;
	__global unsigned char* nextImage;

	//mutable PtrStepSz<DataTerm> corresImg;
	__global DataTerm* corresImg;
	float maxDepthDelta;

	float3 kt;
	mat33 krkinv;

	int cols;
	int rows;
	int N;

	int pitch;
	int imgPitch;

	__global int2* out;
}RGBResidual_;

int2 getProducts_residualKernel(RGBResidual_* rgb, int k)
{
	int cols = rgb->cols;
	int rows = rgb->rows;
	__global unsigned char* lastImage = rgb->lastImage;
	__global unsigned char* nextImage = rgb->nextImage;
	__global DataTerm* corresImg = rgb->corresImg;
	__global float* nextDepth = rgb->nextDepth;
	__global float* lastDepth = rgb->lastDepth;
	mat33 krkinv = rgb->krkinv;
	__global short* dIdx = rgb->dIdx;
	__global short* dIdy = rgb->dIdy;
	int pitch = rgb->pitch;
	float minScale = rgb->minScale;
	float3 kt = rgb->kt;
	float maxDepthDelta = rgb->maxDepthDelta;

	int i = k / cols;
	int j0 = k - (i * cols);

	int2 value = { 0, 0 };

	DataTerm corres;

	corres.valid = false;

	if (i >= 0 && i < rows && j0 >= 0 && j0 < cols)
	{
		if (j0 < cols - 5 && i < rows - 1)
		{
			bool valid = true;

			for (int u = max(i - 2, 0); u < min(i + 2, rows); u++)
			{
				for (int v = max(j0 - 2, 0); v < min(j0 + 2, cols); v++)
				{
					valid = valid && (nextImage[u*cols + v] > 0);
				}
			}

			if (valid)
			{
				/*short * ptr_input_x = (short*)((unsigned char*) dIdx.data + i * pitch);
				short * ptr_input_y = (short*)((unsigned char*) dIdy.data + i * pitch);*/
				__global short * ptr_input_x = (__global short*)((__global unsigned char*) dIdx + i * pitch);
				__global short * ptr_input_y = (__global short*)((__global unsigned char*) dIdy + i * pitch);

				short valx = ptr_input_x[j0];
				short valy = ptr_input_y[j0];
				float mTwo = (valx * valx) + (valy * valy);

				if (mTwo >= minScale)
				{
					int y = i;
					int x = j0;

					float d1 = nextDepth[y*cols + x];

					if (!isnan(d1))
					{
						float transformed_d1 = (float)(d1 * (krkinv.data[2].x * x + krkinv.data[2].y * y + krkinv.data[2].z) + kt.z);
						int u0 = /*__float2int_rn*/((d1 * (krkinv.data[0].x * x + krkinv.data[0].y * y + krkinv.data[0].z) + kt.x) / transformed_d1);
						int v0 = /*__float2int_rn*/((d1 * (krkinv.data[1].x * x + krkinv.data[1].y * y + krkinv.data[1].z) + kt.y) / transformed_d1);

						if (u0 >= 0 && v0 >= 0 && u0 < cols && v0 < rows)
						{
							float d0 = lastDepth[v0*cols + u0];

							if (d0 > 0 && fabs(transformed_d1 - d0) <= maxDepthDelta && lastImage[v0*cols + u0] != 0)
							{
								corres.zero.x = u0;
								corres.zero.y = v0;
								corres.one.x = x;
								corres.one.y = y;
								corres.diff = (float)(nextImage[y*cols + x]) - (float)(lastImage[v0*cols + u0]);
								corres.valid = true;
								value.x = 1;
								value.y = corres.diff * corres.diff;
							}
						}
					}
				}
			}
		}
	}

	corresImg[k] = corres;

	return value;
}

int2 warpReduceSum_int2(int2 val)
{
	int warpSize = 32;
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		val.x += __shfl_down_int(val.x, offset);
		val.y += __shfl_down_int(val.y, offset);
	}

	return val;
}

int2 blockReduceSum_int2(int2 val)
{
	//static __shared__ int2 shared[32];
	__local int2 shared_mem[32];
	int idx = get_local_id(0);
	int warpSize = 32;

	int lane = idx % warpSize;

	int wid = idx / warpSize;

	val = warpReduceSum_int2(val);

	//write reduced value to shared memory
	if (lane == 0)
	{
		//shared[wid] = val;
		shared_mem[wid] = val;
	}
	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE);
	const int2 zero = { 0, 0 };

	//ensure we only grab a value from shared memory if that warp existed
	//val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;
	val = idx < get_local_size(0) / warpSize ? shared_mem[lane] : zero;
	if (wid == 0)
	{
		val = warpReduceSum_int2(val);
	}

	return val;
}

void residualKernel_(RGBResidual_* rgb)
{
	int2 sum = { 0, 0 };
	int N = rgb->N;
	//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	for (int i = get_global_id(0); i<N; i += get_global_size(0))
	{
		int2 val = getProducts_residualKernel(rgb, i);
		sum.x += val.x;
		sum.y += val.y;
	}

	sum = blockReduceSum_int2(sum);

	//if (threadIdx.x == 0)
	if (get_local_id(0) == 0)
	{
		//out[blockIdx.x] = sum;
		rgb->out[get_group_id(0)] = sum;
	}
}

__kernel void residualKernel(float minScale,
	__global short* dIdx,
	__global short* dIdy,
	__global float* lastDepth,
	__global float* nextDepth,
	__global unsigned char* lastImage,
	__global unsigned char* nextImage,
	__global DataTerm* corresImg,
	float maxDepthDelta,
	float3 kt,
	__global mat33* krkinv)
{
	RGBResidual_ rgb;
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
	rgb.krkinv = krkinv[0];
	residualKernel_(&rgb);
}
 


__kernel void reduceSum_int2_kernel(__global int2 * in, __global int2 * out, int N)
{
	int2 sum = { 0, 0 };

	//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	for (int i = get_global_id(0); i<N; i += get_global_size(0))
	{
		sum.x += in[i].x;
		sum.y += in[i].y;
	}

	sum = blockReduceSum_int2(sum);

	/*if (threadIdx.x == 0)
	{
	out[blockIdx.x] = sum;
	}*/
	if (get_global_id(0) == 0)
	{
		out[get_group_id(0)] = sum;
	}
}
