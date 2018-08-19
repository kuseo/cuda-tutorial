
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <book.h>
#include <cpu_bitmap.h>
#define DIM 512

/*
공유 데이터 핸들
*/
GLuint bufferObj;
cudaGraphicsResource *resource;

int main(int argc, char **argv)
{
	cudaDeviceProp prop;	//cuda device
	int dev;	//cuda device 식별값
	
	/*
	cuda device 멤버 값을 0으로 초기화한 후 주 버전을 1, 부 버전을 0으로 설정
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
