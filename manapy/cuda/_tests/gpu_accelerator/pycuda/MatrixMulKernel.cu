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