/* dot product */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <book.h>

#define imin(a,b) (a<b?a:b)

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
	
}