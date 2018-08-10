
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
	구의 속성값
	*/
	float r, g, b;
	float radius;
	float x, y, z;

	/*
	projective view
	(ox, oy)에 위치한 픽셀에서 발사한 광선이 구와 충돌하는지 판별. 충돌시 그 깊이값 반환 
	*/
	__device__ float hit(float ox, float oy, float *n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if (radius * radius > dx*dx + dy*dy)
		{
			float dz = sqrtf(radius*radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius*radius); //음영을 부여하기 위한 scale 값. 구의 중심에서 멀어질수록 작아짐
			return dz + z; //깊이값을 반환
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