#include "postprocess_gpu.h"

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>

// Макрос для проверки ошибок CUDA (версия с исключением для C++ кода)
#define CUDA_CHECK(expr)                                                     \
    do {                                                                    \
        cudaError_t err = (expr);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            throw std::runtime_error(cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

struct Candidate {
    float x1, y1, x2, y2;
    float score;
    int classId;
};

//Sort candidates by score in descending order on thrust
struct CandidateScoreGreater {
    __host__ __device__ bool operator()(const Candidate& a, const Candidate& b) const {
        return a.score > b.score;
    }
};

// Sigmoid activation function for model's logits
__device__ __forceinline__ float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Intersection over Union (IoU) calculation
__device__ __forceinline__ float iou_dev(const Candidate& a, const Candidate& b) {
    float xx1 = fmaxf(a.x1, b.x1);
    float yy1 = fmaxf(a.y1, b.y1);
    float xx2 = fminf(a.x2, b.x2);
    float yy2 = fminf(a.y2, b.y2);
    float w = fmaxf(0.0f, xx2 - xx1);
    float h = fmaxf(0.0f, yy2 - yy1);
    float inter = w * h;
    float areaA = fmaxf(0.0f, a.x2 - a.x1) * fmaxf(0.0f, a.y2 - a.y1);
    float areaB = fmaxf(0.0f, b.x2 - b.x1) * fmaxf(0.0f, b.y2 - b.y1);
    float denom = areaA + areaB - inter;
    return denom > 0.0f ? (inter / denom) : 0.0f;
}

// Dynamic anchor metadata stored in constant memory (tiny; safe).
// We assume 5 FPN levels and 9 anchors per location (3 scales * 3 ratios).
__constant__ int c_level_offsets[6]; // prefix sums in anchors (not cells). offsets[0]=0, offsets[5]=totalAnchors
__constant__ int c_featW[5];
__constant__ int c_featH[5];
__constant__ int c_stride[5];
__constant__ float c_baseSize[5];

static void update_anchor_meta_if_needed(int inputW, int inputH) {
    // Cache last seen input size to avoid repetitive cudaMemcpyToSymbol.
    static int lastW = -1;
    static int lastH = -1;
    if (inputW == lastW && inputH == lastH) return;

    const int stride_h[5] = {8, 16, 32, 64, 128};
    const float baseSize_h[5] = {32.f, 64.f, 128.f, 256.f, 512.f};

    int featW_h[5];
    int featH_h[5];
    int offsets_h[6];
    offsets_h[0] = 0;

    for (int l = 0; l < 5; ++l) {
        int s = stride_h[l];
        featW_h[l] = (inputW + s - 1) / s; // ceil
        featH_h[l] = (inputH + s - 1) / s; // ceil
        int anchorsL = featW_h[l] * featH_h[l] * 9;
        offsets_h[l + 1] = offsets_h[l] + anchorsL;
    }

    // sizeof от целевых переменных для типобезопасности
    CUDA_CHECK(cudaMemcpyToSymbol(c_stride, stride_h, sizeof(c_stride)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_baseSize, baseSize_h, sizeof(c_baseSize)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_featW, featW_h, sizeof(c_featW)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_featH, featH_h, sizeof(c_featH)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_level_offsets, offsets_h, sizeof(c_level_offsets)));

    lastW = inputW;
    lastH = inputH;
}

//anchor calculation from index
__device__ __forceinline__ Anchor anchor_from_index_dynamic(int i) {
    // Find level by offsets (only 5 levels, linear scan is fine)
    int level = 0;
    #pragma unroll
    for (int l = 0; l < 5; ++l) {
        if (i < c_level_offsets[l + 1]) { level = l; break; }
    }

    int local = i - c_level_offsets[level];
    int fw = c_featW[level];
    int stride = c_stride[level];
    float size = c_baseSize[level];

    // local layout: cellIdx * 9 + (scaleIdx*3 + ratioIdx)
    int cell = local / 9;
    int rem = local - cell * 9;
    int scaleIdx = rem / 3;
    int ratioIdx = rem - scaleIdx * 3;

    int cy = cell / fw;
    int cx = cell - cy * fw;

    float ctr_x = (cx + 0.5f) * stride;
    float ctr_y = (cy + 0.5f) * stride;

    const float scales[3] = {1.0f, 1.2599210499f, 1.5874010520f};
    const float ratios[3] = {0.5f, 1.0f, 2.0f};

    float base = size * scales[scaleIdx];
    float area = base * base;
    float ar = ratios[ratioIdx];
    float aw = sqrtf(area / ar);
    float ah = aw * ar;

    Anchor a;
    a.x1 = ctr_x - 0.5f * aw;
    a.y1 = ctr_y - 0.5f * ah;
    a.x2 = ctr_x + 0.5f * aw;
    a.y2 = ctr_y + 0.5f * ah;
    return a;
}

__global__ void k_decode_and_filter(
    const float* cls_logits,   // [A*C]
    const float* bbox_deltas,  // [A*4]
    int numAnchors,
    int numClasses,
    float confTh,
    float scaleX,
    float scaleY,
    int origW,
    int origH,
    Candidate* out,
    int maxOut,
    int* outCount
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAnchors) return;

    // Best class
    float bestS = 0.0f;
    int bestC = -1;
    const float* logits = cls_logits + i * numClasses;
    #pragma unroll 4
    for (int c = 0; c < numClasses; ++c) {
        float s = sigmoidf_dev(logits[c]);
        if (s > bestS) { bestS = s; bestC = c; }
    }
    if (bestS < confTh) return;

    // Decode
    Anchor a = anchor_from_index_dynamic(i);
    float ax = 0.5f * (a.x1 + a.x2);
    float ay = 0.5f * (a.y1 + a.y2);
    float aw = (a.x2 - a.x1);
    float ah = (a.y2 - a.y1);

    float dx = bbox_deltas[i * 4 + 0];
    float dy = bbox_deltas[i * 4 + 1];
    float dw = bbox_deltas[i * 4 + 2];
    float dh = bbox_deltas[i * 4 + 3];

    float px = dx * aw + ax;
    float py = dy * ah + ay;
    float pw = __expf(dw) * aw;
    float ph = __expf(dh) * ah;

    float x1 = (px - 0.5f * pw) * scaleX;
    float y1 = (py - 0.5f * ph) * scaleY;
    float x2 = (px + 0.5f * pw) * scaleX;
    float y2 = (py + 0.5f * ph) * scaleY;

    // Clip
    x1 = fminf(fmaxf(x1, 0.0f), (float)(origW - 1));
    y1 = fminf(fmaxf(y1, 0.0f), (float)(origH - 1));
    x2 = fminf(fmaxf(x2, 0.0f), (float)(origW - 1));
    y2 = fminf(fmaxf(y2, 0.0f), (float)(origH - 1));

    if ((x2 - x1) <= 1.0f || (y2 - y1) <= 1.0f) return;

    int idx = atomicAdd(outCount, 1);
    if (idx >= maxOut) return;
    out[idx] = Candidate{x1, y1, x2, y2, bestS, bestC};
}

__global__ void k_init_int(int* arr, int n, int val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

// Naive parallel suppression: for each (i,j) with i<j, suppress j if IoU(i,j)>thr
__global__ void k_nms_suppress(
    const Candidate* cand,
    int n,
    float iouTh,
    int* suppressed
) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (j >= n) return;
    if (j <= i) return;
    if (suppressed[i]) return;
    if (suppressed[j]) return;
    float iou = iou_dev(cand[i], cand[j]);
    if (iou > iouTh) {
        // cand is sorted by score desc, so i has >= score than j
        suppressed[j] = 1;
    }
}

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
) {
    // Make anchor metadata match current input size.
    update_anchor_meta_if_needed(inputW, inputH);

    // Validate anchors count for the current input size.
    const int strides_h[5] = {8, 16, 32, 64, 128};
    int expected = 0;
    for (int l = 0; l < 5; ++l) {
        int s = strides_h[l];
        int fw = (inputW + s - 1) / s;
        int fh = (inputH + s - 1) / s;
        expected += fw * fh * 9;
    }
    if (numAnchors != expected) {
        // Engine outputs and anchor geometry mismatch -> fallback should trigger.
        return {};
    }

    // Создаём stream для асинхронных операций
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float scaleX = (float)origW / (float)inputW;
    float scaleY = (float)origH / (float)inputH;

    Candidate* d_cand = nullptr;
    int* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cand, sizeof(Candidate) * maxCandidates));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

    int block = 256;
    int grid = (numAnchors + block - 1) / block;
    k_decode_and_filter<<<grid, block, 0, stream>>>(
        d_cls_logits, d_bbox_deltas,
        numAnchors, numClasses,
        confThreshold,
        scaleX, scaleY,
        origW, origH,
        d_cand, maxCandidates, d_count
    );

    // Синхронизируем stream перед чтением счётчика (нужен результат на хосте)
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_count <= 0) {
        cudaFree(d_cand);
        cudaFree(d_count);
        cudaStreamDestroy(stream);
        return {};
    }
    if (h_count > maxCandidates) h_count = maxCandidates;

    // Sort candidates by score desc (device) в stream
    thrust::device_ptr<Candidate> cand_begin(d_cand);
    thrust::device_ptr<Candidate> cand_end(d_cand + h_count);
    thrust::sort(thrust::cuda::par.on(stream), cand_begin, cand_end, CandidateScoreGreater{});

    // Keep only topK for NMS to keep O(N^2) reasonable
    int n = h_count;
    if (topK > 0 && n > topK) n = topK;

    int* d_supp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_supp, sizeof(int) * n));
    k_init_int<<<(n + 255) / 256, 256, 0, stream>>>(d_supp, n, 0);

    dim3 grid2(n, (n + 255) / 256);
    k_nms_suppress<<<grid2, 256, 0, stream>>>(d_cand, n, nmsThreshold, d_supp);

    // Copy back topK candidates + suppression flags (асинхронно)
    std::vector<Candidate> h_cand(n);
    std::vector<int> h_supp(n);
    CUDA_CHECK(cudaMemcpyAsync(h_cand.data(), d_cand, sizeof(Candidate) * n, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_supp.data(), d_supp, sizeof(int) * n, cudaMemcpyDeviceToHost, stream));

    // Синхронизируем перед использованием данных на хосте
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<Detection> out;
    out.reserve(256);
    for (int i = 0; i < n; ++i) {
        if (h_supp[i]) continue;
        Detection d{};
        d.box[0] = h_cand[i].x1;
        d.box[1] = h_cand[i].y1;
        d.box[2] = h_cand[i].x2;
        d.box[3] = h_cand[i].y2;
        d.conf = h_cand[i].score;
        d.classId = h_cand[i].classId;
        out.push_back(d);
    }

    cudaFree(d_supp);
    cudaFree(d_cand);
    cudaFree(d_count);
    cudaStreamDestroy(stream);
    return out;
}


