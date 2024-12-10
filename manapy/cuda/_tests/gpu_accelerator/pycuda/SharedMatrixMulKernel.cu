#define BLOCK_SIZE 32

__global__ void matrix_mul(float *left, float *right, float *res, int dim) {

    int i,j;
    int x, y;
    float temp = 0;

    __shared__ float sa [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sb[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    x = threadIdx.x;
    y = threadIdx.y;


    for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {
        j = tileNUM * BLOCK_SIZE + x;
        i = tileNUM * BLOCK_SIZE + y;

        sa[y][x] = left[row * dim + j];// Coalesced access
        sb[y][x] = right[i * dim + col]; // Coalesced access
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
          temp += sa[y][k] * sb[k][x]; //no shared memory bank conflict

        __syncthreads();
    }

    res[row * dim + col] = temp;
}
