
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <book.h>
#include <cpu_bitmap.h>

#define DIM 1024
#define PI 3.141592

__global__ void kernel(unsigned char *ptr)
{
	

}

int main()
{
	CPUBitmap bitmap(DIM, DIM);			//1024 * 1024 비트
	unsigned char *dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
	dim3 grids(DIM / 16, DIM / 16);		//16 * 16 블럭
	dim3 threads(16, 16);				//
	//kernel << <grids, threads >> > (dev_bitmap);
	HANDLE_ERROR(cudaMemcpy(bitmap.get_bitmap_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost)));

	bitmap.display_and_exit();
	cudaFree(dev_bitmap);

    return 0;
}
