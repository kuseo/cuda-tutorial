/* dot product */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <book.h>
#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024; //내적할 벡터의 차원
const int threadsPerBlock = 256; //한 블럭당 스레드 수
const int blocksPergrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock); //블럭 낭비 방지

int main()
{

}