
#define GL_GLEXT_PROTOTYPES

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

void draw()
{


	glutSwapBuffers();
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
	
	glGenBuffers(1, &bufferObj);	//���� �ڵ� ����
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);	//�ڵ��� �ȼ� ���ۿ� ���ε�

	/*
	OpenGL ����̹����� ���۸� �Ҵ��ϵ��� ��û.
	���۴� ��Ÿ�� �� �������� �����ǹǷ� GL_DYNAMIC_DRAW_ARB �� ������ ������.
	*/
	glBufferData(GL_PIXEL_PACK_BUFFER_ARB, DIM*DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);
	


	glutDisplayFunc(draw);
	glutMainLoop();
    return 0;
}
