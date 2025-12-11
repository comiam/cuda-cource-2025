#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <filesystem>


#include "image_utils.h"

using namespace std;

extern vector<float> applySobelCuda(vector<float> input_pixels, int width, int height);


const vector<vector<int>> sobelX = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const vector<vector<int>> sobelY = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};

vector<float> applySobelCPU(const vector<float>& image, int width, int height)
{
    
    vector<float> result(image.size(), 0);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float sumX = 0, sumY = 0;
            
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    
                    const size_t idx = ny * width + nx;
                    
                    sumX += image[idx] * sobelX[dy+1][dx+1];
                    sumY += image[idx] * sobelY[dy+1][dx+1];
                }
            }
            
            float gradientMagnitude = sqrt(sumX*sumX + sumY*sumY);
            result[y*width+x] = fminf(gradientMagnitude, 1.0f);
        }
    }

    return result;
}

int main(int argc, char* argv[])
{

    string inputPath(argv[1]);
    
    Image img = loadImage(inputPath.c_str());
    img = rgbToGrayscale(img);

    const int height = img.height;
    const int width = img.width;

    vector<float> normalized_pixels_vec = normalize(img.data);

    auto start_time = chrono::high_resolution_clock::now();
    auto output_pixels_vec_cpu = applySobelCPU(normalized_pixels_vec, width, height);
    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;
    double cpu_duration = elapsed.count();
    cout << "Время работы на CPU: " << cpu_duration << " секунд.\n";
    vector<unsigned char> output_pixels_cpu = denormalize(output_pixels_vec_cpu);
    

    start_time = chrono::high_resolution_clock::now();
    auto output_pixels_vec_gpu = applySobelCuda(normalized_pixels_vec, width, height);
    elapsed = chrono::high_resolution_clock::now() - start_time;
    double cuda_duration = elapsed.count();
    cout << "Время работы на GPU с учетом обращений к памяти: " << cuda_duration << " секунд.\n";
    vector<unsigned char> output_pixels_gpu = denormalize(output_pixels_vec_gpu);

    vector<unsigned char> stacked_images;
    stacked_images.reserve(2*height*width); // preallocate memory
    stacked_images.insert( stacked_images.end(), output_pixels_cpu.begin(), output_pixels_cpu.end() );
    stacked_images.insert( stacked_images.end(), output_pixels_gpu.begin(), output_pixels_gpu.end() );

    Image output_image(width, 2*height, 1);
    output_image.data = stacked_images;


    auto filePath = filesystem::path(inputPath);
    auto filenameWithoutExtension = filePath.stem();
    string newFilename = "sobel_" + filenameWithoutExtension.string() + filePath.extension().string();
    auto outputPath = (filePath.parent_path() / newFilename).string();
    
    saveImage(output_image, outputPath.c_str());

    printf("Success\n");

    return 0;
}