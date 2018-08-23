/*
GPU���� ������ �̹��� �����Ͱ� CPU�� ��ġ�� �ʰ�
��ٷ� ���� ��ü�� OpenGL�� �Ѱ����� ����  
*/

#define GL_GLEXT_PROTOTYPES

#include <cmath>
#include <cpu_bitmap.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <book.h>



#define DIM 512

/*
���� ������ �ڵ�
*/
GLuint bufferObj;
cudaGraphicsResource *resource;

__global__ void kernel(uchar4 * ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x / (float)DIM - 0.5f;
	float fy = y / (float)DIM - 0.5f;
	unsigned char green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

	ptr[offset].x = 0;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;
}

void draw()
{
	/*
	bufferObj�� ���� ���۷� ���ε� �����Ƿ� ������ ���ڴ� NULL
	*/
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

void key(unsigned char key, int x, int y)
{
	/*
	ESC Ű�� ���ø����̼� ����
	*/
	switch (key)
	{
	case '27':
		HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);	//���� ���� ���ε� ����
		glDeleteBuffersARB(1, &bufferObj);
		exit(0);
	}
}


int main(int argc, char **argv)
{
	cudaDeviceProp prop;	//cuda device
	int dev;	//cuda device �ĺ���
	
	/*
	cuda device ��� ���� 0���� �ʱ�ȭ�� �� �� ������ 1, �� ������ 0���� ����
	*/
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	

	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	HANDLE_ERROR(cudaGLSetGLDevice(dev));

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("bitmap");
	
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		printf("Error: %s\n", glewGetErrorString(err));
	}

	glGenBuffersARB(1, &bufferObj);	//���� ���� �ڵ� ����
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);	//�ڵ��� �ȼ� ���ۿ� ���ε�

	/*
	OpenGL ����̹����� ���۸� �Ҵ��ϵ��� ��û.
	DIM * DIM ũ���� 32 ��Ʈ ������ ���۸� ������.(handle)
	���۴� ��Ÿ�� �� �������� �����ǹǷ� GL_DYNAMIC_DRAW_ARB �� ������ ������.
	*/
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);
	
	/*
	OpenGL�� CUDA �������� PBO�� ����� ������ CUDA ��Ÿ�ӿ� ���.
	resource�� ���۸� ����Ű�� CUDA ���� �ڵ��� ��ϵ�.
	������ ���� ������ ���� �б� �Ǵ� ���� ���� ������ �ƴ� None.
	*/
	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

	/*
	Ŀ�ο� ���޵� ����̽� �޸��� ���� �ּ� ��û(pointer)
	*/
	uchar4 *devPtr;		//x y z w
	size_t size;
	HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grids, threads >> > (devPtr);

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));

	glutKeyboardFunc(key);
	glutDisplayFunc(draw);
	glutMainLoop();
    return 0;
}
