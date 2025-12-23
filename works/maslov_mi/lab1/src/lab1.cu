#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void circleKernel(int w, char* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < w && j < w) {
        float r = w / 2.0f;
        float x = i - r + 0.5f;
        float y = j - r + 0.5f;
        
        int outputIndex = j * (w + 1) + i;
        
        float dist = sqrtf(x*x + y*y);
        
        if (fabsf(dist - r) < 0.5f)
            output[outputIndex] = '*';
        else
            output[outputIndex] = ' ';
        
        if (i == w - 1) {
            output[outputIndex + 1] = '\n';
        }
    }
}

__global__ void squareKernel(int w, char* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < w && j < w) {
        int outputIndex = j * (w + 1) + i;
        
        if (j == 0 || j == w-1 || i == 0 || i == w-1)
            output[outputIndex] = '*';
        else
            output[outputIndex] = ' ';
        
        if (i == w - 1) {
            output[outputIndex + 1] = '\n';
        }
    }
}

int main()
{
    int w = 100;
    
    int outputSize = w * (w + 1) + 1;
    char* h_output = (char*)malloc(outputSize * sizeof(char));
    h_output[outputSize - 1] = '\0';
    
    char* d_output;
    cudaMalloc(&d_output, outputSize * sizeof(char));
    cudaMemset(d_output, ' ', outputSize * sizeof(char));
    
    dim3 blockSize(32, 32);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, 
                  (w + blockSize.y - 1) / blockSize.y);
    
    printf("Circle:\n");
    circleKernel<<<gridSize, blockSize>>>(w, d_output);
    cudaMemcpy(h_output, d_output, outputSize * sizeof(char), cudaMemcpyDeviceToHost);
    printf("%s", h_output);

    cudaDeviceSynchronize();
    
    cudaMemset(d_output, ' ', outputSize * sizeof(char));
    
    printf("\nSquare:\n");
    squareKernel<<<gridSize, blockSize>>>(w, d_output);
    cudaMemcpy(h_output, d_output, outputSize * sizeof(char), cudaMemcpyDeviceToHost);
    printf("%s", h_output);

    cudaDeviceSynchronize();

    free(h_output);
    cudaFree(d_output);
    cudaDeviceReset();
    return 0;
}