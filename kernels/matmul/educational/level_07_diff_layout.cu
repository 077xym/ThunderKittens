#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>

#include "kittens.cuh"
using namespace kittens;

// the tensor core supports largest tile of size m64k16n256, let's try this layout
// initial tile_layout (64, 64) -> ~180TFLOPS
// this tile_layout (64, 256) -> ~282TFLOPS
constexpr int BM = 64;
constexpr int BK = 64;
constexpr int BN = 256;

// 8 warp worker -> 2 wgs
#define NUM_WORKERS  (8)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

struct matmul_globals {
    using sub_tile_a = st_bf<BM, BK>;
    using sub_tile_b = st_bf<BK, BN>;
    using sub_tile_c = st_bf<BM, BN>;
    using tile_gl_a =  gl<bf16,  1, 1, -1, -1, sub_tile_a>;
    using tile_gl_b =  gl<bf16,  1, 1, -1, -1, sub_tile_b>;
    using tile_gl_c =  gl<bf16,  1, 1, -1, -1, sub_tile_c>;
    tile_gl_a A;
    tile_gl_b B;
    tile_gl_c C;
    int N;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BM, BK> (&As)[2] = al.allocate<st_bf<BM,BK>, 2>();
    st_bf<BK, BN> (&Bs)[2] = al.allocate<st_bf<BK,BN>, 2>();

    int tic = 0;
    int toc = 1;

    rt_fl<16, BN> C_accum;

    int row = blockIdx.y;
    int col = blockIdx.x;

    const int warpid      = kittens::warpid();
    const int warpgroupid = warpid/4;

    __shared__ semaphore bar;
    if (threadIdx.x == 0) { // this should be on thread and not warp (SA: note)
        init_semaphore(bar, 0, 1);
        tma::expect_bytes(
            bar,
            size_bytes<typeof(As[0])> +
            size_bytes<typeof(Bs[0])>
        );
        tma::load_async(As[tic], g.A, {0, 0, row, 0}, bar);
        tma::load_async(Bs[tic], g.B, {0, 0, 0, col}, bar);
    }
    __syncthreads();

    zero(C_accum);
    int num_tiles = (g.N + BK - 1) / BK;
    for (int tile = 0; tile < num_tiles; ++tile, tic^=1, toc^=1) {

        // arrive memory
        wait(bar, tic);
        __syncthreads();

        // load next
        if(warpgroupid == 0) {
            warpgroup::decrease_registers<32>();
            if (threadIdx.x == 0 && tile+1 < num_tiles) {
                tma::expect_bytes(bar,
                    size_bytes<typeof(As[0])> +
                    size_bytes<typeof(Bs[0])>
                );
                tma::load_async(As[toc], g.A, {0, 0, row, tile+1}, bar);
                tma::load_async(Bs[toc], g.B, {0, 0, tile+1, col}, bar);
            }
        } else {
            warpgroup::increase_registers<256>();
            warpgroup::mma_AB(C_accum, As[tic], Bs[tic]);
            warpgroup::mma_async_wait();
            if (threadIdx.x == 128) {
                for (int i = 0; i < C_accum.width; i++) {
                    for (int j = 0; j < 4; j++) {
                        printf("%f, %f\n", C_accum.tiles[0][i].data[j].x, C_accum.tiles[0][i].data[j].y);
                    }
                }
            }
        }
        __syncthreads();
    }
    if ( warpgroupid == 1 ) {
        warpgroup::store(g.C, C_accum, {0, 0, row, col});
    }
}

// launch kernel
void matmul(bf16* A, bf16* B, bf16* C, int N) {

    // global pointers
    using a_gl = matmul_globals::tile_gl_a;
    using b_gl = matmul_globals::tile_gl_b;
    using c_gl = matmul_globals::tile_gl_c;
    a_gl  a_arg{A, nullptr, nullptr, N, N};
    b_gl  b_arg{B, nullptr, nullptr, N, N};
    c_gl  c_arg{C, nullptr, nullptr, N, N};
    matmul_globals g{a_arg, b_arg, c_arg, N};

    // launch
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);
    unsigned long mem_size = 100000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

#include "launch.cu"
