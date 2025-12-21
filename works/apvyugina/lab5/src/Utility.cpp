#include "Utils.h"


std::vector<cimg_library::CImg<float>> Utility::processInput(Params p, const std::filesystem::path img_folder){
    std::vector<cimg_library::CImg<float>> preprocessedImgList;
    for (auto const& dir_entry : std::filesystem::directory_iterator{img_folder}){
        if (std::filesystem::is_directory(dir_entry)) continue;
        cimg_library::CImg<float> image((char *)dir_entry.path().c_str());
        if (image.is_empty()) continue;

        image.normalize(0.0F, 1.0F);
        image.resize(p.inputWidth, p.inputHeight);

        preprocessedImgList.push_back(image);

    }
    return preprocessedImgList;
};


void Utility::drawResult(
    cimg_library::CImg<float> img, 
    std::vector<Detection> detections, 
    const char* file_name
){
    
    cimg_library::CImg<unsigned char> img_normalized = img.normalize(0, 255);
    unsigned char color[] = {255, 0, 0};
    unsigned char fg_color[] = {255, 255, 255};
    unsigned char bg_color[] = {0, 0, 0};
    
    for (Detection const& det : detections){
        img.draw_rectangle(det.bbox.x0, det.bbox.y0, det.bbox.x1, det.bbox.y1, color, 1, 0xFFFFFFFF);
        img.draw_text(det.bbox.x0, det.bbox.y0, std::to_string(det.classId).c_str(), fg_color, bg_color, 1);
    }

    img.save_png(file_name);
};



std::vector<std::vector<Detection>> Utility::processOutput(float* output, int numImages, Params params)
{
    std::vector<std::vector<Detection>> resultList;
    
    uint32_t row_ptr;
    uint32_t floatsPerImage = params.outputItemSize * params.outputLength;
    for (int i=0; i < numImages; ++i){
        std::vector<Detection> result;
        for (int j=0; j < params.outputLength; ++j){
            if (j==0){
                for (int k=0; k < params.outputItemSize; ++k){
                    std::cout << output[i * floatsPerImage + j * params.outputItemSize + k] << " ";
                }
                std::cout << std::endl;
            }
            if (output[i * floatsPerImage + j * params.outputItemSize + 4] < 0.6) continue;
            row_ptr = i * floatsPerImage + j * params.outputItemSize;
            result.push_back(
                Detection(
                    BBox(output[row_ptr], output[row_ptr+1], output[row_ptr+2], output[row_ptr+3]), 
                    output[row_ptr+4], 
                    output[row_ptr+5]
                )
            );
            
            for (int k=0; k < params.outputItemSize; ++k){
                std::cout << output[row_ptr + k] << " ";
            }
            std::cout << std::endl;
            
            
        }
        resultList.push_back(result);
    }
    
    /* 
    std::cout << "Processed " << resultList.size() << " images with detections: [";
    for (int k=0; k < numImages; ++k){
        std::cout << resultList[k].size() << " ";
    }
    std::cout << "]" << std::endl;
    */
    
    return resultList;
}


void Utility::logInference(Params p, const char* engine, int maxBatchSize, std::vector<double> data){

    // Запись в файл
	if (!std::filesystem::exists(p.outputFileName)) {
		std::ofstream outputFile(p.outputFileName);
		std::vector<int> batchHeaders(maxBatchSize);
    	std::iota(batchHeaders.begin(), batchHeaders.end(), 1); // fill from 1 to maxBatchSize
		
		outputFile << "Backend,Model,NumThreads,";
        std::copy(batchHeaders.begin(), batchHeaders.end(), std::experimental::ostream_joiner(outputFile, ','));
        outputFile << "\n"; // Добавляем символ новой строки после заголовка
		outputFile.close();
    }


	std::ofstream outputFile(p.outputFileName, std::ios_base::app);
	assert(outputFile.is_open());
	

	// поиск названия модели
	std::string model_variant;
	std::regex pattern(R"(v10([a-z]+)_dyn\.onnx)");
	std::smatch match;
	if (std::regex_search(p.onnxFileName, match, pattern)) {
        model_variant = match.str(1); 
    } else {
        throw std::runtime_error("Could not parse model name");
    }

	outputFile << engine << "," << model_variant << "," << p.numThreads << ",";
	std::copy(data.begin(), data.end(), std::experimental::ostream_joiner(outputFile, ','));
    outputFile << "\n";
	outputFile.close();
}