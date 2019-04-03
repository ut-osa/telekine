#include<stdio.h>

extern "C" {
__global__
void HelloWorld(){
    int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
    printf("Hello World! thread #%d\n", thid);
}
}
