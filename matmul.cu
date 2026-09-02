#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#define TILE 16   // block size: 16x16 threads per block

// Blocks and threads:
// The grid is made of TILE x TILE (16x16) thread blocks. Each block is
// responsible for computing one 16x16 tile of the output matrix C.
// Each individual thread inside a block computes exactly one output
// element C[row][col]. Threads in the same block share fast on-chip
// shared memory: they cooperatively load a tile of A and a tile of B,
// then each thread reuses those TILE values from shared memory instead
// of re-reading them from slower global memory for every multiply-add.
__global__ void matmulTiledKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];   // shared tile of A for this block
    __shared__ float Bs[TILE][TILE];   // shared tile of B for this block

    int row = blockIdx.y * TILE + threadIdx.y;   // output row this thread handles
    int col = blockIdx.x * TILE + threadIdx.x;   // output col this thread handles

    float acc = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    // slide across the shared K dimension one tile at a time
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // load one element of A and one of B into shared memory
        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();   // wait until whole tile is loaded

        // multiply this tile and add to running total
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();   // wait before loading next tile
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

// plain triple-loop matmul on the CPU, used as the baseline
static void cpuMatmul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) {
                acc += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

// wall-clock time in seconds, used for CPU timing
static double nowSeconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// stop the program and print the CUDA error if any call fails
static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char** argv) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);   // matrix size comes from the command line

    const unsigned SEED = 5330u;   // course SEED, for reproducible random matrices
    srand(SEED);

    size_t bytes = (size_t)N * N * sizeof(float);
    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC_gpu = (float*)malloc(bytes);
    float* hC_cpu = (float*)malloc(bytes);

    // fill A and B with random values
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        hA[i] = (float)(rand() % 100) / 10.0f;
        hB[i] = (float)(rand() % 100) / 10.0f;
    }

    // ---- CPU baseline timing ----
    int cpuReps = (N <= 256) ? 3 : 1;   // repeat small sizes so timing is more stable
    double cpuStart = nowSeconds();
    for (int r = 0; r < cpuReps; ++r) {
        cpuMatmul(hA, hB, hC_cpu, N);
    }
    double cpuEnd = nowSeconds();
    double cpuMs = (cpuEnd - cpuStart) * 1000.0 / cpuReps;

    // ---- allocate GPU memory ----
    float *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, bytes), "malloc dA");
    checkCuda(cudaMalloc(&dB, bytes), "malloc dB");
    checkCuda(cudaMalloc(&dC, bytes), "malloc dC");

    cudaEvent_t evH2Dstart, evH2Dstop, evKstart, evKstop, evD2Hstart, evD2Hstop;
    cudaEventCreate(&evH2Dstart); cudaEventCreate(&evH2Dstop);
    cudaEventCreate(&evKstart);   cudaEventCreate(&evKstop);
    cudaEventCreate(&evD2Hstart); cudaEventCreate(&evD2Hstop);

    // warm-up run so the first-call overhead doesn't skew the timed run below
    checkCuda(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice), "warmup H2D A");
    checkCuda(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice), "warmup H2D B");
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmulTiledKernel<<<grid, block>>>(dA, dB, dC, N);
    checkCuda(cudaDeviceSynchronize(), "warmup kernel");

    // ---- timed copy of A and B to the GPU ----
    cudaEventRecord(evH2Dstart);
    checkCuda(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice), "H2D B");
    cudaEventRecord(evH2Dstop);
    cudaEventSynchronize(evH2Dstop);

    // ---- timed kernel run, averaged over a few repeats ----
    int kernelReps = 5;
    cudaEventRecord(evKstart);
    for (int r = 0; r < kernelReps; ++r) {
        matmulTiledKernel<<<grid, block>>>(dA, dB, dC, N);
    }
    cudaEventRecord(evKstop);
    cudaEventSynchronize(evKstop);

    // ---- timed copy of the result back to the CPU ----
    cudaEventRecord(evD2Hstart);
    checkCuda(cudaMemcpy(hC_gpu, dC, bytes, cudaMemcpyDeviceToHost), "D2H C");
    cudaEventRecord(evD2Hstop);
    cudaEventSynchronize(evD2Hstop);

    float h2dMs = 0, kernelMsTotal = 0, d2hMs = 0;
    cudaEventElapsedTime(&h2dMs, evH2Dstart, evH2Dstop);
    cudaEventElapsedTime(&kernelMsTotal, evKstart, evKstop);
    cudaEventElapsedTime(&d2hMs, evD2Hstart, evD2Hstop);
    float kernelMs = kernelMsTotal / kernelReps;
    float transferMs = h2dMs + d2hMs;
    float gpuEndToEndMs = h2dMs + kernelMs + d2hMs;

    // ---- check the GPU result matches the CPU result ----
    double maxAbsErr = 0.0;
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        double diff = fabs((double)hC_cpu[i] - (double)hC_gpu[i]);
        if (diff > maxAbsErr) maxAbsErr = diff;
    }

    printf("N=%d\n", N);
    printf("CPU time (ms):            %.3f\n", cpuMs);
    printf("GPU kernel time (ms):     %.3f\n", kernelMs);
    printf("H2D+D2H transfer (ms):    %.3f\n", transferMs);
    printf("GPU end-to-end (ms):      %.3f\n", gpuEndToEndMs);
    printf("Speedup (CPU / GPU end-to-end): %.3fx\n", cpuMs / gpuEndToEndMs);
    printf("Max abs error CPU vs GPU: %e\n", maxAbsErr);

    free(hA); free(hB); free(hC_gpu); free(hC_cpu);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(evH2Dstart); cudaEventDestroy(evH2Dstop);
    cudaEventDestroy(evKstart);   cudaEventDestroy(evKstop);
    cudaEventDestroy(evD2Hstart); cudaEventDestroy(evD2Hstop);

    return 0;
}
