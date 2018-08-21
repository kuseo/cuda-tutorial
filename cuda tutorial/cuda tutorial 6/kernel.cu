
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
공유 데이터 핸들
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
	
	glGenBuffers(1, &bufferObj);	//버퍼 핸들 생성
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);	//핸들을 픽셀 버퍼에 바인딩

	/*
	OpenGL 드라이버에게 버퍼를 할당하도록 요청.
	버퍼는 런타임 중 여러차례 수정되므로 GL_DYNAMIC_DRAW_ARB 의 패턴을 따른다.
	*/
	glBufferData(GL_PIXEL_PACK_BUFFER_ARB, DIM*DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);
	


	glutDisplayFunc(draw);
	glutMainLoop();
    return 0;
}
