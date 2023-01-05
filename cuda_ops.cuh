#ifndef GOL_CUDA_GL_CUDA_OPS_CUH
#define GOL_CUDA_GL_CUDA_OPS_CUH

#include <windows.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#define MAX_BLOCK_SIZE 1024


#define cudaCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }


void cudaAssert(cudaError_t status, const std::string &f_name, const int &line);


void cudaRunStep(uint8_t *frame, uint8_t *buffer, int N, int M, int thread_size = 32);


void cudaUpdateTexture(float *tex, uint8_t *frame, int size, int thread_size = 32);


#endif //GOL_CUDA_GL_CUDA_OPS_CUH
