
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <book.h>
#include <cpu_bitmap.h>
#define DIM 512

/*
���� ������ �ڵ�
*/
GLuint bufferObj;
cudaGraphicsResource *resource;

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

	//glutMainLoop();
    return 0;
}
