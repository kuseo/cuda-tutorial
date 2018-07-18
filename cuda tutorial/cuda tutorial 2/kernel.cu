
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <book.h>
#include <stdio.h>
#define N 10
/* grid는 사용할 블럭들의 범위를 지정 (3차원) */

__global__ void add(int * a, int * b, int * c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x; //(스레드 인덱스) + (블럭 인덱스) * (한 블럭당 스레드 수)
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
