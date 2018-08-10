
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cpu_bitmap.h>
#include <book.h>

#define rnd(x) (x*rand()/RAND_MAX)
#define DIM 1024
#define SPHERES 20
#define INF 2e10f

struct Sphere
{
	/*
	���� �Ӽ���
	*/
	float r, g, b;
	float radius;
	float x, y, z;

	/*
	projective view
	ī�޶��� ��ġ�� +INF
	(ox, oy)�� ��ġ�� �ȼ����� �߻��� ������ ���� �浹�ϴ��� �Ǻ�. �浹�� �� ���̰� ��ȯ 
	*/
	__device__ float hit(float ox, float oy, float *n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if (radius * radius > dx*dx + dy*dy)
		{
			float dz = sqrtf(radius*radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius*radius); //������ �ο��ϱ� ���� scale ��. ���� �߽ɿ��� �־������� �۾���
			return dz + z; //���̰��� ��ȯ
		}
		return -INF;
	}
};

__global__ void kernel(Sphere *s, unsigned char *ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x*gridDim.x;
	
	/*
	z���� ȭ���� ������ ������ ��
	*/
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	/*
	�� ������(�ȼ�)���� ��� ������ �浹 �˻�
	�� ����� ���� �ȼ� ������ ����
	*/
	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for (int i = 0; i < SPHERES; i++)
	{
		float n = 0;
		float t = s[i].hit(ox, oy, &n);
		if (t > maxz)
		{
			float scale = n;
			r = s[i].r*scale;
			g = s[i].g*scale;
			b = s[i].b*scale;
			maxz = t;
		}
	}
	ptr[offset*4 + 0] = (int)(r * 255);
	ptr[offset*4 + 1] = (int)(g * 255);
	ptr[offset*4 + 2] = (int)(b * 255);
	ptr[offset*4 + 3] = 255;
}

Sphere *s;

int main()
{
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;

	/*
	gpu �޸� �Ҵ�
	*/
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere)*SPHERES));

	/*
	�� �����͸� cpu �޸𸮿� ����
	*/
	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere)*SPHERES);
	for (int i = 0; i < SPHERES; i++)
	{
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}

	/*
	gpu �޸𸮷� �� ������ ����
	*/
	HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere)*SPHERES, cudaMemcpyHostToDevice));
	free(temp_s);

	/*
	kernel ����
	*/
	dim3 grids(DIM / 16, DIM / 16);	//16*16��
	dim3 threads(16, 16);			//���� 16*16������
	kernel << <grids, threads >> > (s, dev_bitmap);

	/*
	gpu �޸𸮷κ��� ��Ʈ�� ����
	*/
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
	
	bitmap.display_and_exit();
    
	cudaFree(dev_bitmap);
	cudaFree(s);
	return 0;
}