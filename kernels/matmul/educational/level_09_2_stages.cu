#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>
#include "prototype.cuh"

#include "kittens.cuh"
using namespace kittens;

// let's first write 2-stage pipelining, it is pretty similar to the ping pong model ~586 TFLOPS

constexpr int BM = 64;
constexpr int BK = 64;
constexpr int BN = 256;
// we have 3 wgs, 1 producer and 2 consumers within a block.
// the producer will load 2 sub_tile of a and 1 sub_tile of b from A and B
// the 2 consumers will be responsible for computing 1 mma each
// the resulting block tile becomes 128 by 256.
constexpr int M_BLOCKS = 2;
#define NUM_PRODUCERS (4)
#define NUM_CONSUMERS (M_BLOCKS*4)
#define NUM_THREADS ((NUM_PRODUCERS+NUM_CONSUMERS)*kittens::WARP_THREADS)

struct matmul_globals {
    using sub_tile_a = st_bf<BM, BK>;
    using sub_tile_b = st_bf<BK, BN>;
    using sub_tile_c = st_bf<BM, BN>;
    using tile_gl_a = gl<bf16, 1, 1, -1, -1, sub_tile_a>;
    using tile_gl_b = gl<bf16, 1, 1, -1, -1, sub_tile_b>;
    using tile_gl_c = gl<bf16, 1, 1, -1, -1, sub_tile_c>;
    tile_gl_a A;
    tile_gl_b B;
    tile_gl_c C;
    int N;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // we can logically make our smem into 2-D, the first dimension denotes which smem are we interested in ping pong mode,
    // the second dimension denotes which sub tile we are interested in when doing mma
    st_bf<BM,BK> (&As)[2][M_BLOCKS] = al.allocate<st_bf<BM, BK>, 2, M_BLOCKS>();
    // since our b tile layout is 64 by 256, we only need to load one subtile of b during the iteration, so no need for another dim here
    st_bf<BK,BN> (&Bs)[2] = al.allocate<st_bf<BK, BN>, 2>();
    // temporary storage of C subtiles on smem, for later faster tma store to global memory.
    st_bf<BM, BN> (&Cs)[M_BLOCKS] = al.allocate<st_bf<BM, BN>, 2>();

    // arrive phase bit, starts from 0, and consumer waits on it
    int phase_bit_arrive[] = {0, 0};
    // finish phase bit, starts from 1 and producer waits on it. The reason to start from 1 is that, we can avoid the corner case when compute hasn't started
    // yet.
    int phase_bit_finish[] = {1, 1};
    // warp-level registers
    rt_fl<16, BN> C_accum;

    // here is the mapping of block Idx to warp Idx.
    // since one block contains M_BLOCKS sub_tile row, we need to multiply M_BLOCKS from blockIdx.y
    int row = blockIdx.y * M_BLOCKS;
    // one block contains 1 sub_tile col, so, no need to change blockIdx.x
    int col = blockIdx.x;

    const int warpid = kittens::warpid();
    const int warpgroupid = warpid / 4;

    bool is_producer = (warpgroupid == 0);
    bool is_consumer = (warpgroupid > 0 && warpgroupid <= M_BLOCKS);

    // 0-indexed consumer index for accessing smem
    int consumer_idx = (is_consumer) ? (warpgroupid - 1) : 0;

    __shared__ semaphore arrived[2], finished[2];
    // init the semaphore
    if (threadIdx.x == 0) {
        for (int i = 0; i < 2; i++) {
            // arrived semaphore only requires 1 thread
            init_semaphore(arrived[i], 0, 1);
            // finished semaphore requires NUM_CONSUMERS * kittens::WARP_THREADS threads
            init_semaphore(finished[i], 0, NUM_CONSUMERS * kittens::WARP_THREADS);
        }
    }

    __syncthreads();

    if (is_consumer) {
        zero(C_accum);
    }

    int num_tiles = (g.N + BK - 1) / BK;

    // producer
    if (is_producer) {
        int input_ring = 0;
        warpgroup::decrease_registers<40>();
        for (int tile = 0; tile < num_tiles; ++tile) {
            // wait on finished
            wait(finished[input_ring], phase_bit_finish[input_ring]);
            phase_bit_finish[input_ring] ^= 1;
            if (threadIdx.x == 0) {
                tma::expect_bytes(
                    arrived[input_ring],
                    M_BLOCKS * size_bytes<typeof(As[0][0])> +
                    size_bytes<typeof(Bs[0])>
                );
                for (int m = 0; m < M_BLOCKS; m++) {
                    tma::load_async(As[input_ring][m], g.A, {0, 0, row + m, tile}, arrived[input_ring]);
                }
                tma::load_async(Bs[input_ring], g.B, {0, 0, tile, col}, arrived[input_ring]);
            }
            input_ring ^= 1;
        }
    } else { // consumer
        int input_ring = 0;
        warpgroup::increase_registers<232>();
        for (int tile = 0; tile < num_tiles; ++tile) {
            wait(arrived[input_ring], phase_bit_arrive[input_ring]);
            phase_bit_arrive[input_ring] ^= 1;
            warpgroup::mma_AB(
                C_accum,
                As[input_ring][consumer_idx],
                Bs[input_ring]
            );
            warpgroup::mma_async_wait();
            arrive(finished[input_ring]);
            input_ring ^= 1;
        }
    }

    // finish
    if (is_consumer) {
        // store C_accum to Cs
        warpgroup::store(Cs[consumer_idx], C_accum);
        warpgroup::sync(warpgroupid+4);

        // only the first warp in each consumer group calls tma
        if (warpid % 4 == 0) {
            tma::store_async(g.C, Cs[consumer_idx], {0, 0, row+consumer_idx, col});
            tma::store_async_read_wait();
        }
    }
}

void matmul(bf16* A, bf16* B, bf16* C, int N) {
    using gl_a = matmul_globals::tile_gl_a;
    using gl_b = matmul_globals::tile_gl_b;
    using gl_c = matmul_globals::tile_gl_c;
    gl_a a_arg{A, nullptr, nullptr, N, N};
    gl_b b_arg{B, nullptr, nullptr, N, N};
    gl_c c_arg{C, nullptr, nullptr, N, N};
    matmul_globals g{a_arg, b_arg, c_arg, N};

    dim3 grid_size((N + BN - 1) / BN, (N + M_BLOCKS * BM - 1) / (M_BLOCKS * BM));
    unsigned long mem_size = 200000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<grid_size, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

#include "launch.cu"


