
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cpu_bitmap.h>
#include <book.h>

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

int main()
{
    
    return 0;
}