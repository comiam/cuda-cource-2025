#pragma once

#include <vector>
#include "lab5_types.h"


std::vector<Detection> retinanet_postprocess_gpu(
    const float* d_cls_logits,
    const float* d_bbox_deltas,
    int numAnchors,
    int numClasses,
    int inputW,
    int inputH,
    int origW,
    int origH,
    float confThreshold,
    float nmsThreshold,
    int maxCandidates,
    int topK
);


