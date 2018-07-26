/* dot product */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <book.h>

#define imin(a,b) (a<b?a:b)
#define sum_squre(x) (x*(x+1)*(2*x+1))/6

const int N = 33 * 1024; //������ ������ ����
const int threadsPerBlock = 256; //�� ���� ������ ��
const int blocksPergrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock); //�� ���� ����

__global__ void dot(float *a, float *b, float *c)
{
	/* 
	�� �� ���� ������ �� ��ŭ�� �����͸� ������ �� �ִ� �����޸� �迭.
	�� ������ �����ϰ� �ϳ��� ������, �� �� ���� ��������� ���� ����.
	*/
	__shared__ float cache[threadsPerBlock]; 

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	/* 
	cache�� ��ϵǱ� �� �ӽ� ���� ����.
	*/
	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid]; //dot product
		tid += blockDim.x * gridDim.x; //add offset
	}

	cache[cacheIndex] = temp; //cache�� ����

	__syncthreads(); //�� �� ������ ����ȭ

	/*
	reduction
	*/
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i]; //2��� �ε��� ���� ¦���� ������
		__syncthreads();
		i /= 2;
	}

	/*
	���������� ù��° ������ �ϳ��� ����� �������� �����.
	*/
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

int main()
{
	float *a, *b, c, *partial_c; //host
	float *dev_a, *dev_b, *dev_partial_c; //device

	/*
	CPU �޸� �Ҵ�
	*/
	a = (float*)malloc(sizeof(float)*N);
	b = (float*)malloc(sizeof(float)*N);
	partial_c = (float*)malloc(sizeof(float)*blocksPergrid);

	/*
	GPU �޸� �Ҵ�
	*/
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPergrid * sizeof(float)));

	/*
	HOST ������ ����
	*/
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = 2*i;
	}

	/*
	host to device
	memory copy
	*/
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_partial_c, partial_c, blocksPergrid * sizeof(float), cudaMemcpyHostToDevice));

	dot << <blocksPergrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

	/*
	device to host
	memory copy
	*/
	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, sizeof(float)*blocksPergrid, cudaMemcpyDeviceToHost));

	c = 0;
	for (int i = 0; i < blocksPergrid; i++)
	{
		c += partial_c[i];
	}

	/*
	Ȯ��
	*/
	printf("GPU value = %g\nTrue value = %g\n", c, 2 * sum_squre((float)(N - 1)));

	/*
	free memory
	*/
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);

	free(a);
	free(b);
	free(partial_c);

	return 0;
}