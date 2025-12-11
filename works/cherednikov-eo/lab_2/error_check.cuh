#ifndef CUDA_COURCE_2025_ERROR_CHECK_CUH
#define CUDA_COURCE_2025_ERROR_CHECK_CUH

#define CUDA_CHECK(call)                                             \
do {                                                             \
cudaError_t err__ = (call);                                  \
    if (err__ != cudaSuccess) {                                  \
        fprintf(stderr, "[CUDA] %s:%d: %s\n", __FILE__, __LINE__,\
        cudaGetErrorString(err__));                      \
        std::abort();                                            \
    }                                                            \
} while (0)

#endif // CUDA_COURCE_2025_ERROR_CHECK_CUH
