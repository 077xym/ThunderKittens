#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>
#include "prototype.cuh"

#include "kittens.cuh"
using namespace kittens;

// Now, let's deepen our pipeline stages to 4. In this case, we can no longer have (128, 64) tile_a and (64, 256) tile_b. Let's make our block tile
// as (64, 64)

constexpr int BM = 64;
constexpr int BK = 64;
constexpr int BN = 64;
// we can have 1 producer and 1 consumer
#define NUM_WORKERS (8)
#define NUM_THREADS ((NUM_WORKERS)*kittens::WARP_THREADS)
#define INPUT_PIPE_STAGES 3

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
    // we have 4 stages for both a and b
    st_bf<BM,BK> (&As)[INPUT_PIPE_STAGES] = al.allocate<st_bf<BM, BK>, INPUT_PIPE_STAGES>();
    st_bf<BK,BN> (&Bs)[INPUT_PIPE_STAGES] = al.allocate<st_bf<BK, BN>, INPUT_PIPE_STAGES>();
    // temporary storage of C subtiles on smem, for later faster tma store to global memory.
    st_bf<BM, BN> (&Cs) = al.allocate<st_bf<BM, BN>>();

    // arrive phase bit, starts from 0, and consumer waits on it
    int phase_bit_arrive[INPUT_PIPE_STAGES] = {0};
    // finish phase bit, starts from 1 and producer waits on it. The reason to start from 1 is that, we can avoid the corner case when compute hasn't started
    // yet.
    int phase_bit_finish[INPUT_PIPE_STAGES] = {1};
    // warp-level registers
    rt_fl<16, BN> C_accum;

    int row = blockIdx.y;
    int col = blockIdx.x;

    const int warpid = kittens::warpid();
    const int warpgroupid = warpid / 4;

    bool is_producer = (warpgroupid == 0);
    bool is_consumer = (warpgroupid == 1);

    __shared__ semaphore arrived[INPUT_PIPE_STAGES], finished[INPUT_PIPE_STAGES];
    // init the semaphore
    if (threadIdx.x == 0) {
        for (int i = 0; i < INPUT_PIPE_STAGES; i++) {
            // arrived semaphore only requires 1 thread
            init_semaphore(arrived[i], 0, 1);
            // finished semaphore requires NUM_CONSUMERS * kittens::WARP_THREADS threads
            init_semaphore(finished[i], 0, 4 * kittens::WARP_THREADS);
        }
    }

    __syncthreads();

    if (is_consumer) {
        zero(C_accum);
    }

    __syncthreads();

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
                    size_bytes<typeof(As[0])> +
                    size_bytes<typeof(Bs[0])>
                );

                tma::load_async(As[input_ring], g.A, {0, 0, row, tile}, arrived[input_ring]);

                tma::load_async(Bs[input_ring], g.B, {0, 0, tile, col}, arrived[input_ring]);
            }
            input_ring = (input_ring + 1) % (INPUT_PIPE_STAGES);
        }
    } else { // consumer
        int input_ring = 0;
        warpgroup::increase_registers<232>();
        for (int tile = 0; tile < num_tiles; ++tile) {
            wait(arrived[input_ring], phase_bit_arrive[input_ring]);
            phase_bit_arrive[input_ring] ^= 1;
            warpgroup::mma_AB(
                C_accum,
                As[input_ring],
                Bs[input_ring]
            );
            warpgroup::mma_async_wait();
            arrive(finished[input_ring]);
            input_ring = (input_ring + 1) % (INPUT_PIPE_STAGES);
        }
    }

    // finish
    if (is_consumer) {
        // store C_accum to Cs
        warpgroup::store(Cs, C_accum);
        warpgroup::sync(warpgroupid+4);
        // only the first warp in each consumer group calls tma
        if (warpid % 4 == 0) {
            tma::store_async(g.C, Cs, {0, 0, row, col});
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

    dim3 grid_size((N + BN - 1) / BN, (N + BM - 1) / BM);
    unsigned long mem_size = 200000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<grid_size, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

#include "launch.cu"


