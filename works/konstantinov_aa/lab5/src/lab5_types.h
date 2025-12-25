#pragma once

#include <array>

struct Detection {
    // x1,y1,x2,y2 in ORIGINAL image coordinates (pixels)
    float box[4];
    float conf;
    int classId;
};

struct Anchor {
    float x1, y1, x2, y2;
};


