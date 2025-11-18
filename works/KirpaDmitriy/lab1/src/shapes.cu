#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "shapes.h"

device bool isInsideCircle(int x, int y, int size) {
    float centerX = size / 2.0f;
    float centerY = size / 2.0f;
    float radius = size / 2.0f - 1.0f;
    
    float dx = x - centerX;
    float dy = y - centerY;
    float dist = sqrtf(dx * dx + dy * dy);
    
    float innerRadius = radius - 0.7f;
    float outerRadius = radius + 0.7f;
    
    return dist >= innerRadius && dist <= outerRadius;
}

device bool isInsideSquare(int x, int y, int size) {
    int border = size / 8;
    if (border < 1) border = 1;
    
    bool onTopBottom = (y < border || y >= size - border) && (x >= border && x < size - border);
    bool onLeftRight = (x < border || x >= size - border) && (y >= border && y < size - border);
    
    return onTopBottom || onLeftRight;
}

device bool isInsideTriangle(int x, int y, int size) {
    int apex = size / 6;
    int base = size - size / 6;
    
    if (y < apex || y > base) return false;
    
    float leftEdge = size / 2.0f - (y - apex) * size / (2.0f * (base - apex));
    float rightEdge = size / 2.0f + (y - apex) * size / (2.0f * (base - apex));
    
    bool onLeftEdge = (x >= leftEdge - 0.7f && x <= leftEdge + 0.7f);
    bool onRightEdge = (x >= rightEdge - 0.7f && x <= rightEdge + 0.7f);
    bool onBase = (y >= base - 1 && x >= leftEdge && x <= rightEdge);
    
    return onLeftEdge || onRightEdge || onBase;
}

global void renderShape(char* output, int size, ShapeType shape) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = size * size;
    
    if (idx >= total) return;
    
    int y = idx / size;
    int x = idx % size;
    
    bool inside = false;
    
    switch(shape) {
        case CIRCLE:
            inside = isInsideCircle(x, y, size);
            break;
        case SQUARE:
            inside = isInsideSquare(x, y, size);
            break;
        case TRIANGLE:
            inside = isInsideTriangle(x, y, size);
            break;
    }
    
    output[idx] = inside ? '*' : ' ';
}

void drawShape(int size, ShapeType shape) {
    int totalSize = size * size;
    char* h_output = new char[totalSize];
    char* d_output;
    
    cudaMalloc(&d_output, totalSize * sizeof(char));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
    
    renderShape<<<blocksPerGrid, threadsPerBlock>>>(d_output, size, shape);
    
    cudaMemcpy(h_output, d_output, totalSize * sizeof(char), cudaMemcpyDeviceToHost);
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            putchar(h_output[y * size + x]);
        }
        putchar('\n');
    }
    
    cudaFree(d_output);
    delete[] h_output;
}
