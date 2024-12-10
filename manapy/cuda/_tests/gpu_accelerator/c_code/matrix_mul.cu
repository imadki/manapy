#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>

//M = 1024 N = 1024 K = 1024 => 30ms in gtx950m
__global__ void matrix_mul(
    const float *a,
    const float *b,
    float *c,
    unsigned int M,
    unsigned int K,
    unsigned int N
) 
{
  unsigned int j = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int i = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i < M and j < N)
  {
    float tmp = 0.0;
    for (int k = 0; k < K; k++)
      tmp += a[i * K + k] * b[k * N + j];
    c[i * N + j] = tmp;
  }
}

void verify_result(
  float *a,
  float *b,
  float *c,
  unsigned int M,
  unsigned int K,
  unsigned int N
)
{
  for (int i = 0; i < M; i++) {
    //printf("%lf\n", a[i * N + 0]);
    for (int j = 0; j < N; j++) {
      float tmp = 0.0;
      for (int k = 0; k < K; k++)
        tmp += a[i * K + k] * b[k * N + j];
      float diff = fabs(tmp - c[i * N + j]);
      if (diff >= 0.0001)
      {
        std::cerr << c[i * N + j] << " != " << tmp << " diff=" << diff << " " << i << " " << j << std::endl;
        exit(1);
      }
    }
  }
}

long long get_time()
{
  struct timeval t;

  gettimeofday(&t, NULL);
  return t.tv_sec * 1000000LL + t.tv_usec;
}

int main() {
  constexpr int M = 1024;
  constexpr int K = 1024;
  constexpr int N = 1024;

  float *a = new float[M * K];
  float *b = new float[K * N];
  float *c = new float[M * N];

  float *d_a;
  float *d_b;
  float *d_c;
  

  // Init values
  for (int i = 0; i < M * K; i++) {
    a[i] = (float)i;
  }

  for (int i = 0; i < K * N; i++) {
    b[i] = 1.0;
  }

  // Allocate memory on the device
  cudaMalloc(&d_a, sizeof(float) * M * K);
  cudaMalloc(&d_b, sizeof(float) * K * N);
  cudaMalloc(&d_c, sizeof(float) * M * N);

  // Copy data from the host to the device
  cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

  dim3 number_of_blocks(64, 64);
  dim3 number_of_threads(32, 32);
  
  matrix_mul<<<number_of_blocks, number_of_threads>>>(d_a, d_b, d_c, M, K, N);

  // Copy result from device to host
  cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

  // Check result for errors
  verify_result(a, b, c, M, K, N);

  //Time
  long long start_time;
  int iter = 10;

  start_time = get_time();
  for (int i = 0; i < iter; i++)
  {
    matrix_mul<<<number_of_blocks, number_of_threads>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    //verify_result(a, b, c, M, K, N);
  }
  double time = (get_time() - start_time) / (double)iter;
  printf("time: %.4lf microseconds\n", time);
  printf("time: %.4lf ms\n", time / 1000.0);

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  std::cout << "DONE\n";

  return 0;
}
