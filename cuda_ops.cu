#include <string>

#include "cuda_ops.cuh"


void cudaAssert(cudaError_t status, const std::string &f_name, const int &line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s; file: %s; line: %d\n",
                cudaGetErrorString(status),
                f_name.c_str(),
                line);
        exit(1);
    }
}


__global__
void cudaUpdateTextureKernel(float *tex, uint8_t *frame, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < size; idx += gridDim.x * blockDim.x) {
        float v = frame[idx];
        int i_tex = 4 * idx;

        tex[i_tex] = v;
        tex[i_tex + 1] = v;
        tex[i_tex + 2] = v;
        tex[i_tex + 3] = v;
    }
}


void cudaUpdateTexture(float *tex, uint8_t *frame, int size, int thread_size) {
    int threads = ceil(size / ((double)thread_size));
    int block_size = min(MAX_BLOCK_SIZE, threads);
    int grid_size = ceil(threads / ((double)MAX_BLOCK_SIZE));

    cudaUpdateTextureKernel<<<grid_size, block_size>>>(tex, frame, size);
}


__global__
void cudaRunStepKernel(uint8_t *frame, uint8_t *buffer, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int accum;
    int i, j;
    for (; idx < N * M; idx += gridDim.x * blockDim.x) {
        accum = 0;
        i = idx / M;
        j = idx % M;

        if (i == 0 || i == N - 1 || j == 0 || j == M - 1) {
            buffer[idx] = 0;
        }
        else {
            for (int si = -1; si <= 1; si++)
                for (int sj = -1; sj <= 1; sj++)
                    accum += frame[idx + M * si + sj];

            accum += 8 * frame[idx];
            buffer[i * M + j] = (accum == 3 || accum == 11 || accum == 12);
        }
    }
}


void cudaRunStep(uint8_t *frame, uint8_t *buffer, int N, int M, int thread_size) {
    int threads = ceil(N * M / ((double)thread_size));
    int block_size = min(MAX_BLOCK_SIZE, threads);
    int grid_size = ceil(threads / ((double)MAX_BLOCK_SIZE));

    cudaRunStepKernel<<<grid_size, block_size>>>(frame, buffer, N, M);
}
