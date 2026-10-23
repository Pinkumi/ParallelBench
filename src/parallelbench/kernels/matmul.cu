// FP32 products accumulated with FP32 fused multiply-add; C-contiguous buffers.
extern "C" __global__ void matmul_naive(
    const float* A, const float* B, float* C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum = fmaf(A[(size_t)row * K + k], B[(size_t)k * N + col], sum);
        C[(size_t)row * N + col] = sum;
    }
}

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
extern "C" __global__ void matmul_tiled(
    const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int start = 0; start < K; start += TILE_SIZE) {
        const int a_col = start + threadIdx.x;
        const int b_row = start + threadIdx.y;
        tile_a[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[(size_t)row * K + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[(size_t)b_row * N + col] : 0.0f;
        // Every thread, including boundary threads, must reach both barriers.
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            sum = fmaf(tile_a[threadIdx.y][k], tile_b[k][threadIdx.x], sum);
        __syncthreads();
    }
    if (row < M && col < N)
        C[(size_t)row * N + col] = sum;
}
