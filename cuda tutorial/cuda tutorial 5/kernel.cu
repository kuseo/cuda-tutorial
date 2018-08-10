
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cpu_bitmap.h>
#include <book.h>

#define rnd(x) (x*rand()/RAND_MAX)
#define DIM 1024
#define SPHERES 10
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

Sphere *s;

int main()
{
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void**)dev_bitmap, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere)*SPHERES));
    return 0;
}