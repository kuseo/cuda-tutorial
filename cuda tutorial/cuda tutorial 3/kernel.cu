/* dot product */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <book.h>

#define imin(a,b) (a<b?a:b)

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
	
}