#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <cudnn.h>
#include <assert.h>

using namespace std;

// g++-9 -std=c++11 -O3 -o lab3 lab3.cpp
// nvcc -std=c++11 -lcudnn -g -o lab3 lab3.cu
// cuda-memcheck ./lab3

#define BLOCK_SIZE 16
#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define P 1
#define I_0_H (H + 2 * P)
#define I_0_W (W + 2 * P)

#define checkCUDNN(expression)                                 \
{                                                              \
    cudnnStatus_t status = (expression);                       \
    if (status != CUDNN_STATUS_SUCCESS) {                      \
      printf("Error on line %d: %s\n", __LINE__,               \
                          cudnnGetErrorString(status));        \
      std::exit(EXIT_FAILURE);                                 \
    }                                                          \
};

__host__ __device__
int calc_I_0_Idx(int c, int y, int x, int h, int w){
  return (c * (h * w) + y * w + x);
}

__host__ __device__
int calc_F_Idx(int k, int c, int j, int i, int c_num, int fh, int fw){
  return (k * (c_num * fw * fh) + c * (fw * fh) + j * fw + i);
}

__host__ __device__
int calc_O_Idx(int k, int y, int x, int h, int w){
  return (k * (h * w) + y * w + x);
}

// C1
__global__
void naiveConvolution(const double* I_0, const double* F, double* O){

  int idx_k = blockIdx.z;
  int idx_x = blockIdx.x;
  int idx_y = blockIdx.y;
  
  if(idx_k < K && idx_x < W && idx_y < H){
    
    // sum c from 0 to C - 1
    // sum j from 0 to FH - 1
    // sum i from 0 to FW - 1
    // 𝐹[𝑘,𝑐,𝐹𝑊−1−𝑖,𝐹𝐻−1−𝑗] ∙𝐼0[𝑐,𝑥+𝑖,𝑦+𝑗]
    double sum = 0;
    for(int idx_c = 0; idx_c < C; idx_c++){
      for(int idx_j = 0; idx_j < FH; idx_j++){
        for(int idx_i = 0; idx_i < FW; idx_i++){
              
          // calc_F_Idx(int k, int c, int j, int i, int c_num, int fh, int fw)
          // But need to transpose the filters
          int idx_F = calc_F_Idx(idx_k, idx_c, FH - 1 - idx_j, FW - 1 - idx_i, C, FH, FW);
              
          // calc_I_0_Idx(int c, int y, int x, int h, int w)
          int idx_I_0 = calc_I_0_Idx(idx_c, idx_y + idx_j, idx_x + idx_i, I_0_H, I_0_W);
              
          sum += F[idx_F] * I_0[idx_I_0];
              
        }
      }
    }
        
    // calc_O_Idx(int k, int y, int x, int h, int w)
    int idx_O = calc_O_Idx(idx_k, idx_y, idx_x, H, W);
    O[idx_O] = sum;
    // printf("O[%d][%d][%d]=%f\n", idx_k, idx_y, idx_x, O[idx_O]);
    
  }
  
}

// C2
__global__
void convolutionWithSharedMemory(const double* I_0, const double* F, double* O){

  // load F(k=k_0, c, i, j) into shared memory
  __shared__ double shared_F_k_0[C][FH][FW];
  int idx_k = blockIdx.z;
  int blockIdx_x = blockIdx.x;
  int blockIdx_y = blockIdx.y;
  int idx_delta_x = threadIdx.x;
  int idx_delta_y = threadIdx.y;
  if(idx_delta_x < FW && idx_delta_y < FH){
    for(int idx_c = 0; idx_c < C; idx_c++){
      int idx_F = calc_F_Idx(idx_k, idx_c, idx_delta_y, idx_delta_x, C, FH, FW);
      shared_F_k_0[idx_c][idx_delta_y][idx_delta_x] = F[idx_F];
      
      /*
      if(idx_k == 0 && blockIdx_x == 0 && blockIdx_y == 0){
        // printf("F[%d][%d][%d] = %f\n", idx_c, idx_delta_y, idx_delta_x, shared_F_k_0[idx_c][idx_delta_y][idx_delta_x]);
      }
      */
      
    }
  }
  
  int idx_x_start = blockIdx_x * (BLOCK_SIZE - FW + 1);
  int actual_x_BlockSize = (blockIdx_x == gridDim.x - 1)? (I_0_W - idx_x_start) : BLOCK_SIZE;
  int delta_x_exlude_end = actual_x_BlockSize - FW + 1;
  /*
  if(idx_k == 0 && blockIdx_x == 0 && blockIdx_y == 0 && idx_delta_x == 0 && idx_delta_y == 0){
    // printf("gridDim.x=%d, idx_x_start=%d, actual_x_BlockSize=%d, delta_x_exlude_end=%d\n", gridDim.x, idx_x_start, actual_x_BlockSize, delta_x_exlude_end);
  }
  */
    
  int idx_y_start = blockIdx_y * (BLOCK_SIZE - FH + 1);
  int actual_y_BlockSize = (blockIdx_y == gridDim.y - 1)? (I_0_H - idx_y_start) : BLOCK_SIZE;
  int delta_y_exlude_end = actual_y_BlockSize - FH + 1;
  /*
  if(idx_k == 0 && blockIdx_x == 0 && blockIdx_y == 0 && idx_delta_x == 0 && idx_delta_y == 0){
    // printf("gridDim.y=%d, idx_y_start=%d, actual_y_BlockSize=%d, delta_y_exlude_end=%d\n", gridDim.y , idx_y_start, actual_y_BlockSize, delta_y_exlude_end);
  }
  */
  
  __shared__ double shared_sub_I_0[C][BLOCK_SIZE][BLOCK_SIZE];
  if(idx_delta_x < actual_x_BlockSize && idx_delta_y < actual_y_BlockSize){
    for(int idx_c = 0; idx_c < C; idx_c++){
      // calc_I_0_Idx(int c, int y, int x, int h, int w)
      int idx_I_0 = calc_I_0_Idx(idx_c, idx_y_start + idx_delta_y, idx_x_start + idx_delta_x, I_0_H, I_0_W);
      shared_sub_I_0[idx_c][idx_delta_y][idx_delta_x] = I_0[idx_I_0];
      /*
      if(idx_k == 0 && blockIdx_x == 0 && blockIdx_y == 1){
        printf("I_0[%d][%d][%d] = %f\n", idx_c, idx_delta_y, idx_delta_x, shared_sub_I_0[idx_c][idx_delta_y][idx_delta_x]);
      }
      */
    }
  }
  
  __syncthreads();
  
  if(idx_k < K && idx_delta_x < delta_x_exlude_end && idx_delta_y < delta_y_exlude_end){
    
    int idx_x = idx_x_start + idx_delta_x;
    int idx_y = idx_y_start + idx_delta_y;
    
    // sum c from 0 to C - 1
    // sum j from 0 to FH - 1
    // sum i from 0 to FW - 1
    // 𝐹[𝑘,𝑐,𝐹𝑊−1−𝑖,𝐹𝐻−1−𝑗] ∙𝐼0[𝑐,𝑥+𝑖,𝑦+𝑗]
    double sum = 0;
    for(int idx_c = 0; idx_c < C; idx_c++){
      for(int idx_j = 0; idx_j < FH; idx_j++){
        for(int idx_i = 0; idx_i < FW; idx_i++){
              
          sum += shared_F_k_0[idx_c][FH - 1 - idx_j][FW - 1 - idx_i] * shared_sub_I_0[idx_c][idx_delta_y + idx_j][idx_delta_x + idx_i];
          
          /*
          if(idx_x == 0 && idx_y == 3 && idx_c == 1){
            printf("I_0(%d, %d, %d) * F(%d, %d, %d) = %f * %f = %f: idx_delta_x=%d, idx_i=%d, idx_delta_y=%d, idx_j=%d, idx_c=%d, blockIdx_x=%d, blockIdx_y=%d\n", idx_c, idx_delta_x + idx_i, idx_delta_y + idx_j, idx_c, FW - 1 - idx_i, FH - 1 - idx_j, shared_sub_I_0[idx_c][idx_delta_y + idx_j][idx_delta_x + idx_i], shared_F_k_0[idx_c][FH - 1 - idx_j][FW - 1 - idx_i], shared_F_k_0[idx_c][FH - 1 - idx_j][FW - 1 - idx_i] * shared_sub_I_0[idx_c][idx_delta_y + idx_j][idx_delta_x + idx_i], idx_delta_x, idx_i, idx_delta_y, idx_j, idx_c, blockIdx_x, blockIdx_y);
          }
          */
              
        }
      }
    }
        
    // calc_O_Idx(int k, int y, int x, int h, int w)
    int idx_O = calc_O_Idx(idx_k, idx_y, idx_x, H, W);
    O[idx_O] = sum;
    // printf("O[%d][%d][%d] = %f\n", idx_k, idx_y, idx_x, O[idx_O]);
    
  }
  
}

// C3
void convolutionUsingcuDNN(const double* I, const double* F, double* O){
  
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));
  
  cudnnTensorDescriptor_t input_descriptor; /* xDesc */
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));
  

  cudnnTensorDescriptor_t output_descriptor; /* yDesc */
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));

  
  cudnnFilterDescriptor_t kernel_descriptor; /* wDesc */
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW));
  
  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, P, P, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
  
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(
    cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,       /* xDesc */
                                        kernel_descriptor,      /* wDesc */
                                        convolution_descriptor, /* convDesc */
                                        output_descriptor,      /* yDesc */
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        0,
                                        &convolution_algorithm));
  
  size_t workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));
  // printf("Workspace size: %f\n", workspace_bytes);
  
  void* d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  int input_bytes = C * H * W * sizeof(double);
  double* I_d{nullptr};
  cudaMalloc(&I_d, input_bytes);
  cudaMemcpy(I_d, I, input_bytes, cudaMemcpyHostToDevice);

  int output_bytes = K * H * W * sizeof(double);
  double* O_C3_d{nullptr};
  cudaMalloc(&O_C3_d, output_bytes);
  cudaMemset(O_C3_d, 0, output_bytes);
  
  int filter_bytes = K * C * FH * FW * sizeof(double);
  double* F_C3_d{nullptr};
  cudaMalloc(&F_C3_d, filter_bytes);
  cudaMemcpy(F_C3_d, F, filter_bytes, cudaMemcpyHostToDevice);
  
  const double alpha = 1, beta = 0;
  checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     I_d,
                                     kernel_descriptor,
                                     F_C3_d,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     O_C3_d));
  
  cudaMemcpy(O, O_C3_d, output_bytes, cudaMemcpyDeviceToHost);
  
  cudaFree(I_d);
  cudaFree(F_C3_d);
  cudaFree(O_C3_d);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  
  cudnnDestroy(cudnn);
  
}

void check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main(int argc, char *argv[]){
  
  double executionTime; // unit: ms
  struct timespec start, end;
  
  // allocate host memory
  // I_0: size C x (H + 2P) x (W + 2P)
  double* I = (double*)malloc(C * H * W * sizeof(double));
  double* I_0 = (double*)malloc(C * I_0_W * I_0_H * sizeof(double));
  // F: size K x C x FH x FW
  double* F = (double*)malloc(K * C * FH * FW * sizeof(double));
  // O: size K x H x W
  double* O_C1 = (double*)malloc(K * H * W * sizeof(double));
  double* O_C2 = (double*)malloc(K * H * W * sizeof(double));
  double* O_C3 = (double*)malloc(K * H * W * sizeof(double));
  
  // init data
  // init I_0
  for(int idx_c = 0; idx_c < C; idx_c++){
    for(int idx_y = 0; idx_y < I_0_H; idx_y++){
      for(int idx_x = 0; idx_x < I_0_W; idx_x++){
        
        // calc_I_0_Idx(int c, int y, int x, int h, int w)
        int idx_I_0 = calc_I_0_Idx(idx_c, idx_y, idx_x, I_0_H, I_0_W);
        int idx_I_x = idx_x - 1;
        int idx_I_y = idx_y - 1;
        
        // if is ghost cells, then value = 0
        if(idx_x == 0 || idx_x == (I_0_W - 1)
           || idx_y == 0 || idx_y == (I_0_H - 1)){
          I_0[idx_I_0] = 0;
        }else{
          // 𝐼[𝑐,𝑥,𝑦]=𝑐∙(𝑥+𝑦)
          I_0[idx_I_0] = idx_c * (idx_I_x + idx_I_y);
          
          // calc_I_0_Idx(int c, int y, int x, int h, int w)
          int idx_I = calc_I_0_Idx(idx_c, idx_I_y, idx_I_x, H, W);
          I[idx_I] = idx_c * (idx_I_x + idx_I_y);
          
        }
        
      }
    }
  }
  
  // init F
  for(int idx_k = 0; idx_k < K; idx_k++){
    for(int idx_c = 0; idx_c < C; idx_c++){
      for(int idx_j = 0; idx_j < FH; idx_j++){
        for(int idx_i = 0; idx_i < FW; idx_i++){
          
          // calc_F_Idx(int k, int c, int j, int i, int c_num, int fh, int fw)
          int idx_F = calc_F_Idx(idx_k, idx_c, idx_j, idx_i, C, FH, FW);
          
          // 𝐹[𝑘,𝑐,𝑖,𝑗]=(𝑐+𝑘)∙(𝑖+𝑗)
          F[idx_F] = (idx_c + idx_k) * (idx_i + idx_j);
          
        }
      }
    }
  }
  
  // init O
  for(int idx_k = 0; idx_k < K; idx_k++){
    for(int idx_y = 0; idx_y < H; idx_y++){
      for(int idx_x = 0; idx_x < W; idx_x++){
        
        // calc_O_Idx(int k, int y, int x, int h, int w)
        int idx_O = calc_O_Idx(idx_k, idx_y, idx_x, H, W);
        O_C1[idx_O] = 0;
        O_C2[idx_O] = 0;
        O_C3[idx_O] = 0;
        
      }
    }
  }
  
  // allocate device memory
  double *I_0_d, *F_d, *O_C1_d, *O_C2_d;
  cudaMalloc(&I_0_d, C * I_0_W * I_0_H * sizeof(double));
  check_CUDA_Error("malloc I_0 failed");
  cudaMalloc(&F_d, K * C * FH * FW * sizeof(double));
  check_CUDA_Error("malloc F failed");
  cudaMalloc(&O_C1_d, K * H * W * sizeof(double));
  check_CUDA_Error("malloc O_C1 failed");
  cudaMalloc(&O_C2_d, K * H * W * sizeof(double));
  check_CUDA_Error("malloc O_C2 failed");
  
  // copy memory from host to device
  cudaMemcpy(I_0_d, I_0, C * I_0_W * I_0_H * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(F_d, F, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);
  
  // lauch kernel
  // GridDim(x, y, z)
  dim3 GridDim(W, H, K);
  
  clock_gettime(CLOCK_MONOTONIC, &start);
  
  // naiveConvolution(const double* I_0, const double* F, double* O, int I_0_h, int I_0_w, int c_num, int fh, int fw, int O_k, int O_w, int O_h)
  naiveConvolution<<<GridDim, 1>>>(I_0_d, F_d, O_C1_d);
  cudaDeviceSynchronize();
  
  clock_gettime(CLOCK_MONOTONIC, &end);
  
  // wait for the complete
  // then, copy memory from device to host
  cudaMemcpy(O_C1, O_C1_d, K * H * W * sizeof(double), cudaMemcpyDeviceToHost);
  
  // print result
  // 1. checksum: the total sum of the elements of O
  // Expected: checksum=122756344698240.000000
  // 2. the time to execute the CUDA kernel with the convolution
  double totalSum = 0;
  for(int idx_k = 0; idx_k < K; idx_k++){
    for(int idx_y = 0; idx_y < H; idx_y++){
      for(int idx_x = 0; idx_x < W; idx_x++){
        
        // calc_O_Idx(int k, int y, int x, int h, int w)
        int idx_O = calc_O_Idx(idx_k, idx_y, idx_x, H, W);
        totalSum += O_C1[idx_O];
        
      }
    }
  }
  
  executionTime = (end.tv_sec - start.tv_sec) * 1e3;
  executionTime = executionTime + ((end.tv_nsec - start.tv_nsec) * 1e-6);
  
  printf("\n");
  printf("%f,%.3f\n", totalSum, executionTime);
  
  // C2
  // calculate needed dimension
  int blockNum_x = ceil(((double)(I_0_W - BLOCK_SIZE))/(BLOCK_SIZE - FW + 1)) + 1;
  int blockNum_y = ceil(((double)(I_0_H - BLOCK_SIZE))/(BLOCK_SIZE - FH + 1)) + 1;
  int blockNum_z = K;
  int threadNum_x = BLOCK_SIZE;
  int threadNum_y = BLOCK_SIZE;
  dim3 gridDim_C2(blockNum_x, blockNum_y, blockNum_z);
  dim3 blockDim_C2(threadNum_x, threadNum_y);
  
  clock_gettime(CLOCK_MONOTONIC, &start);
  // convolutionWithSharedMemory(const double* I_0, const double* F, double* O, int I_0_h, int I_0_w, int c_num, int fh, int fw, int O_k, int O_w, int O_h)
  convolutionWithSharedMemory<<<gridDim_C2, blockDim_C2>>>(I_0_d, F_d, O_C2_d);
  cudaDeviceSynchronize();
  
  clock_gettime(CLOCK_MONOTONIC, &end);
  
  // wait for the complete
  // then, copy memory from device to host
  cudaMemcpy(O_C2, O_C2_d, K * H * W * sizeof(double), cudaMemcpyDeviceToHost);
  
  // print result
  // 1. checksum: the total sum of the elements of O
  // Expected: checksum=122756344698240.000000
  // 2. the time to execute the CUDA kernel with the convolution
  totalSum = 0;
  for(int idx_k = 0; idx_k < K; idx_k++){
    for(int idx_y = 0; idx_y < H; idx_y++){
      for(int idx_x = 0; idx_x < W; idx_x++){
        
        // calc_O_Idx(int k, int y, int x, int h, int w)
        int idx_O = calc_O_Idx(idx_k, idx_y, idx_x, H, W);
        totalSum += O_C2[idx_O];
        
      }
    }
  }
  
  executionTime = (end.tv_sec - start.tv_sec) * 1e3;
  executionTime = executionTime + ((end.tv_nsec - start.tv_nsec) * 1e-6);
  
  printf("\n");
  printf("%f,%.3f\n", totalSum, executionTime);
  
  // C3
  // convolutionUsingcuDNN(const double* I, const double* F, double* O)
  clock_gettime(CLOCK_MONOTONIC, &start);
  convolutionUsingcuDNN(I, F, O_C3);
  clock_gettime(CLOCK_MONOTONIC, &end);
  
  // print result
  // 1. checksum: the total sum of the elements of O
  // Expected: checksum=122756344698240.000000
  // 2. the time to execute the CUDA kernel with the convolution
  totalSum = 0;
  for(int idx_k = 0; idx_k < K; idx_k++){
    for(int idx_y = 0; idx_y < H; idx_y++){
      for(int idx_x = 0; idx_x < W; idx_x++){
        
        // calc_O_Idx(int k, int y, int x, int h, int w)
        int idx_O = calc_O_Idx(idx_k, idx_y, idx_x, H, W);
        totalSum += O_C3[idx_O];
        
      }
    }
  }
  
  executionTime = (end.tv_sec - start.tv_sec) * 1e3;
  executionTime = executionTime + ((end.tv_nsec - start.tv_nsec) * 1e-6);
  
  printf("\n");
  printf("%f,%.3f\n", totalSum, executionTime);
  
  // free device memory
  // free host memory
  cudaFree(I_0_d);
  check_CUDA_Error("free I_0_d failed");
  cudaFree(F_d);
  check_CUDA_Error("free F_d failed");
  cudaFree(O_C1_d);
  check_CUDA_Error("free O_C1_d failed");
  cudaFree(O_C2_d);
  check_CUDA_Error("free O_C2_d failed");
  
  free(O_C1);
  free(O_C2);
  free(O_C3);
  free(F);
  free(I_0);
  free(I);
  
  return 0;
  
}
