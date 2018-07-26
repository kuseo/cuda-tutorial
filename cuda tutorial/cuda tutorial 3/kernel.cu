/* dot product */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <book.h>

#define imin(a,b) (a<b?a:b)
#define sum_squre(x) (x*(x+1)*(2*x+1))/6

const int N = 33 * 1024; //내적할 벡터의 차원
const int threadsPerBlock = 256; //한 블럭당 스레드 수
const int blocksPergrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock); //블럭 낭비 방지

__global__ void dot(float *a, float *b, float *c)
{
	/* 
	한 블럭 내의 스레드 수 만큼의 데이터를 저장할 수 있는 공유메모리 배열.
	각 블럭마다 고유하게 하나씩 가지며, 한 블럭 내의 스레드들이 서로 공유.
	*/
	__shared__ float cache[threadsPerBlock]; 

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	/* 
	cache에 기록되기 전 임시 변수 선언.
	*/
	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid]; //dot product
		tid += blockDim.x * gridDim.x; //add offset
	}

	cache[cacheIndex] = temp; //cache값 설정

	__syncthreads(); //블럭 내 스레드 동기화

	/*
	reduction
	*/
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i]; //2배수 인덱스 끼리 짝지어 리덕션
		__syncthreads();
		i /= 2;
	}

	/*
	최종적으로 첫번째 스레드 하나만 남기는 리덕션이 수행됨.
	*/
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

int main()
{
	float *a, *b, c, *partial_c; //host
	float *dev_a, *dev_b, *dev_partial_c; //device

	/*
	CPU 메모리 할당
	*/
	a = (float*)malloc(sizeof(float)*N);
	b = (float*)malloc(sizeof(float)*N);
	partial_c = (float*)malloc(sizeof(float)*blocksPergrid);

	/*
	GPU 메모리 할당
	*/
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPergrid * sizeof(float)));

	/*
	HOST 데이터 생성
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
	확인
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