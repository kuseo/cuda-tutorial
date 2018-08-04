
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <book.h>
#include <cpu_bitmap.h>

#define DIM 1024
#define PI 3.141592

__global__ void kernel(unsigned char *ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * (blockDim.x * gridDim.x);

	__shared__ float shared[16][16];	//�����޸� ����. �� ���� ������� �ϳ��� �׸��� ����.

	/* ������ �����尡 �����޸� ���ۿ� �� ���� ��� */
	/* sin �Լ��� �� ���� */
	const float period = 128.0f;
	shared[threadIdx.x][threadIdx.y] = 255 * (sinf(2.0f * PI * x / period) + 1.0f) * (sinf(2.0f * PI * y / period) + 1.0f) / 4.0f;
	ptr[offset * 4 + 0] = 0;
	ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main()
{
	CPUBitmap bitmap(DIM, DIM);			//1024 * 1024 ��Ʈ
	unsigned char *dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
	dim3 grids(DIM / 16, DIM / 16);		//64 * 64 ��
	dim3 threads(16, 16);				//�� ���� 16 * 16 ������. �� �����尡 �� ��Ʈ�� ���
	kernel << <grids, threads >> > (dev_bitmap);
	HANDLE_ERROR(cudaMemcpy(bitmap.get_bitmap_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();
	cudaFree(dev_bitmap);

    return 0;
}
