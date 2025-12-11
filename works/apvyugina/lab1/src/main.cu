#include <cuda_runtime.h>
#include <iostream>


using namespace std;


__global__
void drawSquare(char* buffer, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = 0; i < size; ++i) {
        int pos = i * size + idx;
        if(pos < size * size) {
            buffer[pos] = (idx == 0 || idx == size-1 || i == 0 || i == size-1 ? '*' : ' ');
        }
    }
}

int main(int argc, char* argv[]) {

    const int N = stoi(argv[1]);
    const int SIZE = N*N+1; // +1 для завершающего '\0'
    char *d_buffer, h_buffer[SIZE];

    cudaMalloc(&d_buffer, SIZE*sizeof(char));

    dim3 threadsPerBlock(N, 1); 
    dim3 numBlocks(1, N);        

    drawSquare<<<numBlocks, threadsPerBlock>>>(d_buffer, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_buffer, d_buffer, SIZE*sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(d_buffer);

    h_buffer[N*N] = '\0'; 
    for(int i=0;i<N;i++) {
        cout.write(h_buffer+i*N,N)<<endl;
    }

    return 0;
}
