#include <stdio.h>

__global__ void myKernel() 
{ 
  printf("Hello, world from the device!\n"); 
} 

int main() 
{ 
  myKernel<<<1,10>>>(); 
  cudaDeviceSynchronize();
} 