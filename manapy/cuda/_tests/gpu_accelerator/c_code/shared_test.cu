#include <stdio.h>
#include <stdint.h>
#define KB(byte) (1024LL * byte)
#define MB(byte) (1024LL * KB(byte))
#define SHARED_SIZE MB(2048)

__global__ void myKernel() 
{ 
  __shared__ uint8_t var[SHARED_SIZE];
  
  if (threadIdx.x == 0) {
    var[0] = 11;
    var[SHARED_SIZE - 1] = blockIdx.x;
    printf("--Hello, world from the device! %d - %d #%lld\n", var[SHARED_SIZE - 1], var[0], SHARED_SIZE); 
  }
} 

int main() 
{ 
  myKernel<<<500,1024>>>(); 
  cudaDeviceSynchronize();
  printf("==================\n");
  myKernel<<<1,10>>>(); 
  cudaDeviceSynchronize();
} 