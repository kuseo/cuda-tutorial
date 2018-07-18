/* 128 block with 128 thread for each block */
/* do N add calculations parallelly */

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
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice));

	add << <128, 128 >> > (dev_a, dev_b, dev_c); //128 block, 128 thread
	
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	bool success = true;
	for (int i = 0; i < N; i++)
	{
		if (a[i] + b[i] != c[i])
		{
			printf("Error : %d + %d != %d\n", a[i], b[i], c[i]);
			success = false;
		}
		if (success)
			printf("success\n");

		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_c);
	}
    return 0;
}
