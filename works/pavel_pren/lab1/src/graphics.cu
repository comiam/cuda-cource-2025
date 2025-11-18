#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define WIDTH 40
#define HEIGHT 20


__global__ void drawSquare(int *ascii, int left, int top, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < WIDTH && y < HEIGHT) {
        int idx = y * WIDTH + x;
        if ((x == left || x == left + size - 1) && (y >= top && y < top + size)) {
            ascii[idx] = 1;
        } else if ((y == top || y == top + size - 1) && (x >= left && x < left + size)) {
            ascii[idx] = 1;
        } else {
            ascii[idx] = 0;
        }
    }
}


void printAscii(int *ascii) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%c", ascii[y * WIDTH + x] ? '*' : ' ');
        }
        printf("\n");
    }
}

void renderSquare() {
    
    int *d_ascii;
    int *h_ascii = (int *)malloc(WIDTH * HEIGHT * sizeof(int));
    
    cudaMalloc((void **)&d_ascii, WIDTH * HEIGHT * sizeof(int));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                  (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    drawSquare<<<gridSize, blockSize>>>(d_ascii, 8, 3, 14);
    
    cudaMemcpy(h_ascii, d_ascii, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    
    printAscii(h_ascii);
    
    cudaFree(d_ascii);
    free(h_ascii);
}

int main() {

    //int deviceCount = 0;
    //cudaGetDeviceCount(&deviceCount);
    //printf("%d\n", deviceCount);
    
    renderSquare();
    
    return 0;
}
