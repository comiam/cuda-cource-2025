
#include <iostream>
#include <cstring>
#include "shapes.h"

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [size] [shape]\n";
    std::cout << "Shapes: circle, square, triangle\n";
    std::cout << "Size: 5-200\n";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printUsage(argv[0]);
        return 1;
    }

    int size = std::atoi(argv[1]);
    if (size < 5 || size > 200) {
        std::cerr << "Error: size must be between 5 and 200\n";
        return 1;
    }

    ShapeType shape;
    if (std::strcmp(argv[2], "circle") == 0) {
        shape = CIRCLE;
    } else if (std::strcmp(argv[2], "square") == 0) {
        shape = SQUARE;
    } else if (std::strcmp(argv[2], "triangle") == 0) {
        shape = TRIANGLE;
    } else {
        std::cerr << "Error: unknown shape '" << argv[2] << "'\n";
        printUsage(argv[0]);
        return 1;
    }

    drawShape(size, shape);

    return 0;
}
