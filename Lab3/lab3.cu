#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// g++-9 -std=c++11 -O3 -o lab3 lab3.cpp
// nvcc -std=c++11 -o lab3 lab3.cu

#define BLOCK_SIZE 12
#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define P 1
#define I_0_H (H + 2 * P)
#define I_0_W (W + 2 * P)


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
  int idx_c = threadIdx.z;
  int idx_delta_x = threadIdx.x;
  int idx_delta_y = threadIdx.y;
  if(idx_c < C && idx_delta_x < FW && idx_delta_y < FH){
    int idx_F = calc_F_Idx(idx_k, idx_c, idx_delta_y, idx_delta_x, C, FH, FW);
    shared_F_k_0[idx_c][idx_delta_y][idx_delta_x] = F[idx_F];
  }
  
  int idx_x_start = blockIdx_x * (BLOCK_SIZE - FW + 1);
  int actual_x_BlockSize = (blockIdx_x == blockDim.x - 1)? (W - idx_x_start) : BLOCK_SIZE;
  int idx_y_start = blockIdx_y * (BLOCK_SIZE - FH + 1);
  int actual_y_BlockSize = (blockIdx_y == blockDim.y - 1)? (H - idx_y_start) : BLOCK_SIZE;
  
  __shared__ double shared_sub_I_0[C][BLOCK_SIZE][BLOCK_SIZE];
  if(idx_c < C && idx_delta_x < actual_x_BlockSize && idx_delta_y < actual_y_BlockSize){
    // calc_I_0_Idx(int c, int y, int x, int h, int w)
    int idx_I_0 = calc_I_0_Idx(idx_c, idx_y_start + idx_delta_y, idx_x_start + idx_delta_x, I_0_H, I_0_W);
    shared_sub_I_0[idx_c][idx_delta_y][idx_delta_x] = I_0[idx_I_0];
  }
  
  __syncthreads();
  
  if(idx_k < K && idx_delta_x < actual_x_BlockSize && idx_delta_y < actual_y_BlockSize){
    
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
              
          sum += shared_F_k_0[idx_c][FH - 1 - idx_j][FW - 1 - idx_i] * shared_sub_I_0[idx_c][idx_y + idx_j][idx_x + idx_i];
              
        }
      }
    }
        
    // calc_O_Idx(int k, int y, int x, int h, int w)
    int idx_O = calc_O_Idx(idx_k, idx_y, idx_x, H, W);
    O[idx_O] = sum;
    
  }
  
}


int main(int argc, char *argv[]){
  
  double executionTime; // unit: ms
  struct timespec start, end;
  
  // allocate host memory
  // I_0: size C x (H + 2P) x (W + 2P)
  double* I_0 = (double*)malloc(C * I_0_W * I_0_H * sizeof(double));
  // F: size K x C x FH x FW
  double* F = (double*)malloc(K * C * FH * FW * sizeof(double));
  // O: size K x H x W
  double* O = (double*)malloc(K * H * W * sizeof(double));
  
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
    for(int idx_y = 0; idx_y < I_0_H; idx_y++){
      for(int idx_x = 0; idx_x < I_0_W; idx_x++){
        
        // calc_O_Idx(int k, int y, int x, int h, int w)
        int idx_O = calc_O_Idx(idx_k, idx_y, idx_x, H, W);
        O[idx_O] = 0;
        
      }
    }
  }
  
  // allocate device memory
  double *I_0_d, *F_d, *O_d;
  cudaMalloc(&I_0_d, C * I_0_W * I_0_H * sizeof(double));
  cudaMalloc(&F_d, K * C * FH * FW * sizeof(double));
  cudaMalloc(&O_d, K * H * W * sizeof(double));
  
  // copy memory from host to device
  cudaMemcpy(I_0_d, I_0, C * I_0_W * I_0_H * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(F_d, F, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);
  
  // lauch kernel
  // GridDim(x, y, z)
  dim3 GridDim(W, H, K);
  
  clock_gettime(CLOCK_MONOTONIC, &start);
  
  // naiveConvolution(const double* I_0, const double* F, double* O, int I_0_h, int I_0_w, int c_num, int fh, int fw, int O_k, int O_w, int O_h)
  naiveConvolution<<<GridDim, 1>>>(I_0_d, F_d, O_d);
  cudaDeviceSynchronize();
  
  clock_gettime(CLOCK_MONOTONIC, &end);
  
  // wait for the complete
  // then, copy memory from device to host
  cudaMemcpy(O, O_d, K * H * W * sizeof(double), cudaMemcpyDeviceToHost);
  
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
        totalSum += O[idx_O];
        
      }
    }
  }
  
  executionTime = (end.tv_sec - start.tv_sec) * 1e9;
  executionTime = (executionTime + (end.tv_nsec - start.tv_nsec)) * 1e-9;
  
  printf("C1(Naive Convolution on GPU):\n");
  printf("1. checksum = %f\n", totalSum);
  printf("2. Execution time = %f s\n", executionTime);
  
  // free device memory
  // free host memory
  cudaFree(I_0_d);
  cudaFree(F_d);
  cudaFree(O_d);
        
  free(F);
  free(O);
  // free(I_0);
  
  return 0;
  
}
