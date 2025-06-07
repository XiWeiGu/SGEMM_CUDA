#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// 向上取整除法
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define BLOCK_M 32
#define BLOCK_N 32

#define BLOCK_Y 32
#define BLOCK_X 32

#define WARP_SIZE 32

// 检查CUDA错误
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 朴素的SGEMM内核实现
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // 从高层次==>低层次, groupRow  ==>  warpRow ==> threadRow
  // 计算当前线程负责的C矩阵位置
  //const uint x = blockIdx.y * blockDim.y + threadIdx.y;
  //const uint y = blockIdx.x * blockDim.x + threadIdx.x;

  // Group Global Idx
  const uint groupRow = blockIdx.y * BLOCK_M;
  const uint groupCol = blockIdx.x * BLOCK_N;

  // warp/lane ID
  const uint warpId = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
  const uint laneId = (threadIdx.y * blockDim.x + threadIdx.x) % WARP_SIZE;

  // 每个warp需要处理BLOCK的大小
  const uint warp_block = (BLOCK_N * BLOCK_M) /  WARP_SIZE;
  // 假设warap_block=32=WAP_SIZE, 表示每个线程取一个A,B;
  //const uint thread_read_size = warp_block / WARP_SIZE;
  uint warp_block_m = 8;
  uint warp_block_n;
  //for (; warp_block_m <= warp_block; warp_block_m =<< 1) {
  warp_block_n = warp_block / warp_block_m;
  //const uint warpRow = warpId % (BLOCK_M / warp_block_m) * warp_block_m;
  //const uint warpCol = warpId % (BLOCK_N / warp_block_n) * warp_block_n;
  const uint warpRow = (warpId / warp_block_m) * warp_block_m;
  const uint warpCol = (warpId % warp_block_m) * warp_block_n;
  //printf("warpId: %d warpRow: %d warpCol: %d \n", warpId, warpRow, warpCol);
  int threadRow = groupRow + warpRow + laneId / warp_block_n;
  int threadCol = groupCol + warpCol + laneId % warp_block_n;
  float tmp = 0.0f;
  for (int i = 0; i < K; i++) {
    tmp += A[threadRow * K + i] * B[i * N + threadCol];
  }
  // C = α*(A@B)+β*C
  C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
  //}

#if 0
  // 边界检查（处理M或N不是32的倍数的情况）
  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
#endif
}

// CPU端矩阵乘法实现（用于结果校验）
void sgemm_cpu(int M, int N, int K, float alpha, const float *A,
               const float *B, float beta, float *C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float tmp = 0.0f;
      for (int k = 0; k < K; ++k) {
        tmp += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = alpha * tmp + beta * C[i * N + j];
    }
  }
}

// 验证GPU和CPU结果的一致性
bool verify_results(int M, int N, float *gpu_result, float *cpu_result, float tolerance = 1e-4f) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int idx = i * N + j;
      float diff = fabsf(gpu_result[idx] - cpu_result[idx]);
      if (diff > tolerance) {
        printf("Mismatch at [%d,%d]: GPU=%f, CPU=%f, Diff=%f\n", 
               i, j, gpu_result[idx], cpu_result[idx], diff);
        return false;
      }
    }
  }
  return true;
}

// 生成随机浮点数（-1.0到1.0之间）
float random_float() {
  return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}

int main() {
  // 矩阵维度
  const int M = 128;  // A的行数，C的行数
  const int N = 128;  // B的列数，C的列数
  const int K = 128;  // A的列数，B的行数
  
  // 计算矩阵大小（字节）
  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);
  
  // 分配主机内存
  float *h_A = (float*)malloc(size_A);
  float *h_B = (float*)malloc(size_B);
  float *h_C_gpu = (float*)malloc(size_C);  // GPU计算结果
  float *h_C_cpu = (float*)malloc(size_C);  // CPU计算结果
  
  // 初始化矩阵数据
  srand(42);  // 固定随机种子，确保结果可复现
  for (int i = 0; i < M * K; ++i) h_A[i] = random_float();
  for (int i = 0; i < K * N; ++i) h_B[i] = random_float();
  for (int i = 0; i < M * N; ++i) {
    h_C_gpu[i] = random_float();  // 初始化C矩阵
    h_C_cpu[i] = h_C_gpu[i];      // 确保CPU和GPU使用相同的初始值
  }
  
  // 分配设备内存
  float *d_A, *d_B, *d_C;
  CHECK_CUDA_ERROR(cudaMalloc(&d_A, size_A));
  CHECK_CUDA_ERROR(cudaMalloc(&d_B, size_B));
  CHECK_CUDA_ERROR(cudaMalloc(&d_C, size_C));
  
  // 将数据从主机复制到设备
  CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C_gpu, size_C, cudaMemcpyHostToDevice));
  
  // 设置内核启动参数
  dim3 gridDim(CEIL_DIV(N, BLOCK_M), CEIL_DIV(M, BLOCK_N), 1);
  //dim3 blockDim(32, 32, 1);
  dim3 blockDim(BLOCK_Y, BLOCK_X, 1);
  
  // 预热CUDA上下文
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  
  // 计时开始
  cudaEvent_t start, stop;
  float milliseconds = 0;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  
  // 执行GPU内核
  const float alpha = 1.0f;
  const float beta = 0.0f;
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  
  // 等待内核完成并计时
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  
  // 将结果从设备复制回主机
  CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
  
  // 计算性能指标
  float flops = 2.0f * M * N * K;  // 2MNK FLOPs for matrix multiplication
  float gflops = flops / (milliseconds * 1e-3f) / 1e9f;
  
  printf("SGEMM Naive Performance:\n");
  printf("  Matrix Size: %dx%dx%d\n", M, N, K);
  printf("  Time: %.3f ms\n", milliseconds);
  printf("  GFLOPs: %.2f\n", gflops);
  
  // 执行CPU计算（用于验证）
  printf("\nRunning CPU validation...\n");
  sgemm_cpu(M, N, K, alpha, h_A, h_B, beta, h_C_cpu);
  
  // 验证结果
  bool correct = verify_results(M, N, h_C_gpu, h_C_cpu);
  printf("Validation: %s\n", correct ? "PASSED" : "FAILED");

  // 输出结果
 // for (int i = 0; i < M; ++i) {
 //   for (int j = 0; j < N; ++j) {
 //     int idx = i * N + j;
 //     printf("%f, ", h_C_gpu[idx]);
 //   }
 //   printf("\n");
 // }
 // printf("\n");

 // for (int i = 0; i < M; ++i) {
 //   for (int j = 0; j < N; ++j) {
 //     int idx = i * N + j;
 //     printf("%f, ", h_C_cpu[idx]);
 //   }
 //   printf("\n");
 // }
 // printf("\n");
  
  // 清理资源
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_B));
  CHECK_CUDA_ERROR(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C_gpu);
  free(h_C_cpu);
  
  return 0;
}
