#include <stdio.h>
#include <stdlib.h>

// g++-9 -std=c++11 -O3 -o lab3 lab3.cpp

int H = 1024;
int W = 1024;
int C = 3;
int FW = 3;
int FH = 3;
int K = 64;
int P = 1;
int I_0_H = H + 2 * P;
int I_0_W = W + 2 * P;

int calc_I_0_Idx(int c, int y, int x){
  return (c * (I_0_H * I_0_W) + y * I_0_W + x);
}

int calc_F_Idx(int k, int c, int j, int i){
  return (k * (C * FW * FH) + c * (FW * FH) + j * FW + i);
}

int calc_O_Idx(int k, int y, int x){
  return (k * (H * W) + y * W + x);
}

int main(int argc, char *argv[]){
  
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
        
        // calc_I_0_Idx(int c, int y, int x)
        int idx_I_0 = calc_I_0_Idx(idx_c, idx_y, idx_x);
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
          
          // calc_F_Idx(int k, int c, int j, int i)
          int idx_F = calc_F_Idx(idx_k, idx_c, idx_j, idx_i);
          
          // 𝐹[𝑘,𝑐,𝑖,𝑗]=(𝑐+𝑘)∙(𝑖+𝑗)
          F[idx_F] = (idx_c + idx_k) * (idx_i + idx_j);
          
        }
      }
    }
  }
  
  // allocate device memory
  
  // copy memory from host to device
  
  // lauch kernel
  for(int idx_k = 0; idx_k < K; idx_k++){
    for(int idx_y = 0; idx_y < H; idx_y++){
      for(int idx_x = 0; idx_x < W; idx_x++){
        
        // sum c from 0 to C - 1
        // sum j from 0 to FH - 1
        // sum i from 0 to FW - 1
        // 𝐹[𝑘,𝑐,𝐹𝑊−1−𝑖,𝐹𝐻−1−𝑗] ∙𝐼0[𝑐,𝑥+𝑖,𝑦+𝑗]
        double sum = 0;
        for(int idx_c = 0; idx_c < C; idx_c++){
          for(int idx_j = 0; idx_j < FH; idx_j++){
            for(int idx_i = 0; idx_i < FW; idx_i++){
              
              // calc_F_Idx(int k, int c, int j, int i)
              // But need to transpose the filters
              int idx_F = calc_F_Idx(idx_k, idx_c, FH - 1 - idx_j, FW - 1 - idx_i);
              
              // calc_I_0_Idx(int c, int y, int x)
              int idx_I_0 = calc_I_0_Idx(idx_c, idx_y + idx_j, idx_x + idx_i);
              
              sum += F[idx_F] * I_0[idx_I_0];
              
            }
          }
        }
        
        // calc_O_Idx(int k, int y, int x)
        int idx_O = calc_O_Idx(idx_k, idx_y, idx_x);
        O[idx_O] = sum;
        
      }
    }
  }
  
  // wait for the complete
  // then, copy memory from device to host
  
  // print result
  // 1. checksum: the total sum of the elements of O
  // Expected: checksum=122756344698240.000000
  // 2. the time to execute the CUDA kernel with the convolution
  double totalSum = 0;
  for(int idx_k = 0; idx_k < K; idx_k++){
    for(int idx_y = 0; idx_y < H; idx_y++){
      for(int idx_x = 0; idx_x < W; idx_x++){
        
        int idx_O = calc_O_Idx(idx_k, idx_y, idx_x);
        totalSum += O[idx_O];
        
      }
    }
  }
  printf("checksum=%f\n", totalSum);
  
  // free device memory
  // free host memory
  free(I_0);
  free(F);
  free(O);
  
}
