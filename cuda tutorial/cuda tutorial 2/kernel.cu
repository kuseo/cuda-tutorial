
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <book.h>
#include <stdio.h>
#define N 10
/* grid�� ����� ������ ������ ���� (3����) */

__global__ void add(int * a, int * b, int * c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x; //(������ �ε���) + (�� �ε���) * (�� ���� ������ ��)
	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main()
{
    
    return 0;
}
