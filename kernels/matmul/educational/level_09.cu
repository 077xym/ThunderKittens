#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>
#include "prototype.cuh"

#include "kittens.cuh"
using namespace kittens;

// this code snippet shows how to manually implement a 4-stage pipelining matmul kernel
// To implement the scheduling, here are some prerequisites to know
// 1. we use mbarrier and test_wait.parity to check completion
// 2. within mbarrier, we have a phase bit that starts at 0, the phase bit will be flipped when an async task related to this mbar has finished
// 3. test_wait.parity requires 2 inputs, mbar and a phase bit (this phase bit is provided by the user). test_wait.parity will return and unblock the
//    process when the mbar's phase bit and phase bit provided by the user is not eqaul. And will be hang if they are equal

// the schedule for our 4-stage pipelining will be
// init 4 mbars arrived for producer, which track the completion of async load launched by producer.
// init 4 mbars finished for consumer, which track the completion of async compute launched by consumer.
// 4 because we have a 4-stage pipeline.
// The producer will wait on finished, and consumer will wait on arrived.
// Initially, producer will have its 4 phase bits to be provided to test_wait.parity set to be 1, such that the wait on the arrived will always
// continue since the initial phase bit of mbar of compute is 0
// consumer will have its 4 phase bits set to be 0.
// during the producer loop, the producer get the phase bit, and wait for arrived. As the wait finished, it will begin tma loading, and flip the phase bit
// for next wait.
// consumer does the same thing.

// Since we have 4 mbars for producer and consumer each, we also need 4 phase bit for producer and consumer each, to make sure they are waiting on the correct
// slot.
// Here, I encountered a bug as I use an array to store the phase bit. The array then spills to stack frame of the thread. The so-called local memory is actually
// located in GMEM, which leads to
// 1. detrimental to efficiency
// 2. reordering may occur, making the sync scheduling break.

// The better is to use a bitfield, 0xFFFF0000, and 2 helper functions get_phasebit and update_phasebit for wait and phase bit flip within the iterations.


// set up:
// 1. base_tile_a: 64 by 64, base_tile_b: 64 by 256, base_tile_c: 64 by 256
// 2. 1 producer and 2 consumers
// 3. block_tile_a: 128 by 64, block_tile_b: 64 by 256, block_tile_c: 128 by 256
// 4. 4 stages pipeline

constexpr int BM = 64;
constexpr int BK = 64;
constexpr int BN = 256;
#define NUM_WORKERS (12)
#define NUM_THREADS ((NUM_WORKERS)*kittens::WARP_THREADS)
#define INPUT_PIPE_STAGES 4
// M_BLOCK denotes number of consumers
#define M_BLOCK 2

// get_phasebit, get the phase bit consumer or producer will use for wait
template<int half> __device__ static inline bool get_phasebit(uint32_t bitfield, int ring_id) {
    return (bitfield & (1 << (half*16 + ring_id))) != 0;
}

// update_phasebit, update the phase bit consumer or producer will wait on next round
template<int half> __device__ static inline void update_phasebit(uint32_t &bitfield, int ring_id) {
    bitfield ^= (1 << (half*16 + ring_id));
}

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
    st_bf<BM,BK> (&As)[INPUT_PIPE_STAGES][M_BLOCK] = al.allocate<st_bf<BM, BK>, INPUT_PIPE_STAGES, M_BLOCK>();
    st_bf<BK,BN> (&Bs)[INPUT_PIPE_STAGES] = al.allocate<st_bf<BK, BN>, INPUT_PIPE_STAGES>();

    // we need to reuse shm for epilogue, or the smem requirement will exceed limitation
    shared_allocator al_ep((int*)&__shm[0]);
    st_bf<BM,BN> (&Cs)[M_BLOCK] = al_ep.allocate<st_bf<BM,BN>, M_BLOCK>();

    // create 2 groups such that we can sync on warp group level instead of using __syncthreads
    // the underlying logic is bar.sync
    using producers = group<4>;
    using consumers = group<M_BLOCK * 4>;

    // the bitfield
    uint32_t semaphore_bitfield = 0xFFFF0000;
    // the reg files
    rt_fl<16, BN> C_accum;

    // the block wise row and col each block is responsible for
    int row = blockIdx.y * M_BLOCK;
    int col = blockIdx.x;

    const int warpgroupid = warpgroup::groupid();

    // wg0 and wg1 are consumers, while wg2 is producer
    bool is_producer = (warpgroupid == 2);
    bool is_consumer = (warpgroupid < 2);

    // the mbarrier
    __shared__ semaphore arrived[INPUT_PIPE_STAGES], finished[INPUT_PIPE_STAGES];
    // init the semaphore
    if (threadIdx.x == 0) {
        for (int i = 0; i < INPUT_PIPE_STAGES; i++) {
            // arrived semaphore only requires 1 thread
            init_semaphore(arrived[i], 0, 1);
            // finished semaphore requires NUM_CONSUMERS * kittens::WARP_THREADS threads
            init_semaphore(finished[i], 0, 2);
        }
    }

    // syncthreads here as we want the initialized mbar to be visible by all threads within the block, so no need to
    // partition the wg
    __syncthreads();

    if (is_consumer) {
        zero(C_accum);
    }

    int num_tiles = (g.N + BK - 1) / BK;

    // the producer and consumer model.

    // producer
    if (is_producer) {
        int input_ring = 0;
        warpgroup::decrease_registers<40>();
        for (int tile = 0; tile < num_tiles; ++tile) {
            wait(finished[input_ring], get_phasebit<1>(semaphore_bitfield, input_ring));
            update_phasebit<1>(semaphore_bitfield, input_ring);
            if (warpgroup::laneid() == 0) {
                tma::expect_bytes(
                    arrived[input_ring],
                    M_BLOCK * size_bytes<typeof(As[0][0])> +
                    size_bytes<typeof(Bs[0])>
                );
                for (int idx = 0; idx < M_BLOCK; idx++) {
                    tma::load_async(As[input_ring][idx], g.A, {0, 0, row+idx, tile}, arrived[input_ring]);
                }

                tma::load_async(Bs[input_ring], g.B, {0, 0, tile, col}, arrived[input_ring]);
            }
            input_ring = (input_ring + 1) % (INPUT_PIPE_STAGES);
        }
        producers::sync(14);
    } else { // consumer
        int input_ring = 0;
        int consumer_idx = warpgroupid;
        warpgroup::increase_registers<232>();
        for (int tile = 0; tile < num_tiles; ++tile) {
            wait(arrived[input_ring], get_phasebit<0>(semaphore_bitfield, input_ring));
            update_phasebit<0>(semaphore_bitfield, input_ring);
            warpgroup::mma_AB(
                C_accum,
                As[input_ring][consumer_idx],
                Bs[input_ring]
            );
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(finished[input_ring]);
            }
            input_ring = (input_ring + 1) % (INPUT_PIPE_STAGES);
        }
        consumers::sync(13);

        warpgroup::store(Cs[consumer_idx], C_accum);
        warpgroup::sync(7);
        if (warpgroup::warpid() % 4 == 0) {
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

    dim3 grid_size((N + BN - 1) / BN, (N + M_BLOCK * BM - 1) / (M_BLOCK * BM));
    unsigned long mem_size = 220000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<grid_size, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

#include "launch.cu"
