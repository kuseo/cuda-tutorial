/* dot product */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <book.h>
#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPergrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

int main()
{

}