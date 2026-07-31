#include "getrows.cuh"
#include "dequantize.cuh"
#include "convert.cuh"

#include <atomic>
#include <cstring>
#include <vector>

#include <unistd.h>

template <typename dst_t>
static __global__ void k_s2b_ownership_mask(
        const int32_t * __restrict__ idx, dst_t * __restrict__ dst,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t s10, const size_t s11, const size_t s12,
        const size_t d1, const size_t d2, const size_t d3,
        const int64_t shard_base, const int64_t shard_rows) {
    const int64_t i10 = blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t z   = blockIdx.y;
    if (i10 >= ne10 || z >= ne11*ne12) {
        return;
    }
    const int64_t i11 = z % ne11;
    const int64_t i12 = z / ne11;
    const int64_t global_row = idx[i10*s10 + i11*s11 + i12*s12];
    dst[i10*d1 + i11*d2 + i12*d3] = ggml_cuda_cast<dst_t>(
        global_row >= shard_base && global_row < shard_base + shard_rows ? 0.0f : -INFINITY);
}

// Shared-top-k verify (mode 3): -inf for gathered rows that duplicate one of
// the batch-tail rows (whose live copies are appended explicitly after the
// shared top-k set) - without this a duplicated row would be double-counted
// in the softmax. tail values come from src0 (an I32 input view, [1, n_tail]).
template <typename dst_t>
static __global__ void k_spec_tail_collision_mask(
        const int32_t * __restrict__ tail, const int32_t * __restrict__ idx, dst_t * __restrict__ dst,
        const int64_t ne10, const int64_t n_tail,
        const size_t s10, const size_t d1) {
    const int64_t i10 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i10 >= ne10) {
        return;
    }
    const int32_t v = idx[i10*s10];
    float out = 0.0f;
    for (int64_t t = 0; t < n_tail; ++t) {
        if (tail[t] == v) {
            out = -INFINITY;
            break;
        }
    }
    dst[i10*d1] = ggml_cuda_cast<dst_t>(out);
}

// S2b's hot gather is Q8_0 -> F16 with 576 values per selected row.  The
// generic quantized get_rows kernel assigns a whole 256-thread block to one
// row and needs two column tiles, producing 2*n_sel tiny blocks.  Decode is
// launch/topology bound on the TP2xPP7 3090 setup, so instead assign one warp
// to each complete row: eight rows per block and 16x fewer blocks at n_sel =
// 2048, while preserving the exact dequantization and foreign-row zero fill.
static __global__ void k_s2b_get_rows_q8_0_f16_warp(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, half * __restrict__ dst,
        const int64_t ne00, const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12,
        const int64_t shard_base, const int64_t shard_rows) {
    constexpr int warps_per_block = 256/WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t i10 = (int64_t) blockIdx.x*warps_per_block + warp;
    const int64_t z = blockIdx.y;

    if (i10 >= ne10 || z >= ne11*ne12) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i11 = z % ne11;
    const int64_t i12 = z / ne11;
    int64_t i01 = src1[i10*s10 + i11*s11 + i12*s12];
    half * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    i01 -= shard_base;
    if (i01 < 0 || i01 >= shard_rows) {
        for (int64_t i00 = lane; i00 < ne00; i00 += WARP_SIZE) {
            dst_row[i00] = __float2half(0.0f);
        }
        return;
    }

    const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
    for (int64_t i00 = 2*lane; i00 < ne00; i00 += 2*WARP_SIZE) {
        const int ib  = i00/QK8_0;
        const int iqs = i00 % QK8_0;
        float2 v;
        dequantize_q8_0(src0_row, ib, iqs, v);
        dst_row[i00 + 0] = __float2half(v.x);
        if (i00 + 1 < ne00) {
            dst_row[i00 + 1] = __float2half(v.y);
        }
    }
}

static void get_rows_cuda_s2b_q8_0_f16_warp(
        const void * src0_d, const int32_t * src1_d, half * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream, const int64_t shard_base, const int64_t shard_rows) {
    constexpr int threads = 256;
    constexpr int warps_per_block = threads/WARP_SIZE;
    const dim3 blocks((unsigned) ((ne10 + warps_per_block - 1)/warps_per_block),
                      (unsigned) (ne11*ne12), 1);
    k_s2b_get_rows_q8_0_f16_warp<<<blocks, threads, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, ne10, ne11, ne12,
        nb1/sizeof(half), nb2/sizeof(half), nb3/sizeof(half),
        nb01, nb02, nb03,
        nb10/sizeof(int32_t), nb11/sizeof(int32_t), nb12/sizeof(int32_t),
        shard_base, shard_rows);
    CUDA_CHECK(cudaGetLastError());
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/,
        const int64_t shard_base, const int64_t shard_rows) {

    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 =  blockIdx.x;
            const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
            const int i11 =  dm.x;
            const int i12 =  dm.y;

            int64_t i01 = src1[i10*s10 + i11*s11 + i12*s12];

            const int ib   =  i00/qk;      // block index
            const int iqs  = (i00%qk)/qr;  // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

            // position-sharded gather: rebase global row id to this member's
            // shard; foreign rows are ZERO-FILLED so the member-sum (allreduce)
            // reconstructs the complete gather
            if (shard_rows > 0) {
                i01 -= shard_base;
                if (i01 < 0 || i01 >= shard_rows) {
                    dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(0.0f);
                    dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(0.0f);
                    continue;
                }
            }

            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

            // dequantize
            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
        const src0_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/,
        const int64_t shard_base, const int64_t shard_rows) {

    ggml_cuda_pdl_lc();
    const src0_t  * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const int32_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 = blockIdx.x;
            const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
            const int i11 = dm.x;
            const int i12 = dm.y;

            if (i00 >= ne00) {
                return;
            }

            int64_t i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

            if (shard_rows > 0) {
                i01 -= shard_base;
                if (i01 < 0 || i01 >= shard_rows) {
                    dst_row[i00] = ggml_cuda_cast<dst_t>(0.0f);
                    continue;
                }
            }

            const src0_t * src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

            dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
        }
    }
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst,
        const int64_t ncols, const int64_t nrows_grad, const int64_t nrows_dst) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    ggml_cuda_pdl_sync();

    // grid.y is clamped to the CUDA grid limit, so stride over the destination rows
    for (int64_t dst_row = blockIdx.y; dst_row < nrows_dst; dst_row += gridDim.y) {
        float sum = 0.0f;

        for (int64_t i = 0; i < nrows_grad; ++i) {
            if (rows[i] != dst_row) {
                continue;
            }
            sum += grad[i*ncols + col];
        }

        dst[dst_row*ncols + col] = sum;
    }
}

template<int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void get_rows_cuda_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream, const int64_t shard_base = 0, const int64_t shard_rows = 0) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/,
        shard_base, shard_rows);
}

template<typename src0_t, typename dst_t>
static void get_rows_cuda_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream, const int64_t shard_base = 0, const int64_t shard_rows = 0) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(k_get_rows_float<src0_t, dst_t>, launch_params,
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/,
        shard_base, shard_rows);
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(
        const void * src0_d, const ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream, const int64_t shard_base = 0, const int64_t shard_rows = 0) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float((const half *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float((const float *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_I32:
            get_rows_cuda_float((const int32_t *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_BF16:
            get_rows_cuda_float((const nv_bfloat16 *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_Q1_0:
            get_rows_cuda_q<QK1_0, QR1_0, dequantize_q1_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda_q<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda_q<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda_q<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda_q<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda_q<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        default:
            // TODO: k-quants
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream, int64_t shard_base, int64_t shard_rows) {
    static const bool s2b_warp_gather = []() {
        const char * e = getenv("LLAMA_S2B_WARP_GATHER");
        return e == nullptr || atoi(e) != 0;
    }();
    if (s2b_warp_gather && shard_rows > 0 &&
            src0_type == GGML_TYPE_Q8_0 && dst_type == GGML_TYPE_F16) {
        get_rows_cuda_s2b_q8_0_f16_warp(
            src0_d, src1_d, (half *) dst_d,
            ne00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12,
            nb1, nb2, nb3, stream, shard_base, shard_rows);
        return;
    }
    switch (dst_type) {
        case GGML_TYPE_F32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (float *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_I32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (int32_t *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (half *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (nv_bfloat16 *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, shard_base, shard_rows);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    // LLAMA_SPARSE_DEBUG: one-shot dump of DSA top-k gather indices (sanity:
    // range vs cache extent, recency coverage, degenerate/duplicate patterns)
    static const int sparse_dbg = [](){
        const char * e = getenv("LLAMA_SPARSE_DEBUG");
        return e ? atoi(e) : 0;
    }();
    int64_t sp_cmp_row = -1;
    cudaStreamCaptureStatus sp_cs = cudaStreamCaptureStatusNone;
    if (sparse_dbg > 0 && src1->name[0] != '\0' && strstr(src1->name, "top_k") != nullptr &&
            cudaStreamIsCapturing(stream, &sp_cs) == cudaSuccess && sp_cs == cudaStreamCaptureStatusNone) {
        static std::atomic<int> n_dumps{0};
        if (n_dumps.load() < sparse_dbg) {
            n_dumps++;
            const int64_t n_idx = ggml_nelements(src1);
            std::vector<int32_t> h(n_idx);
            CUDA_CHECK(cudaMemcpyAsync(h.data(), src1->data, n_idx*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            int32_t mn = INT32_MAX, mx = INT32_MIN;
            for (int64_t i = 0; i < n_idx; i++) { mn = std::min(mn, h[i]); mx = std::max(mx, h[i]); }
            int sp_dev = -1; cudaGetDevice(&sp_dev);
            fprintf(stderr, "[SPDBG] dev=%d get_rows idx=%s n=%lld src0_rows=%lld dst_ne1=%lld min=%d max=%d first16:",
                    sp_dev, src1->name, (long long) n_idx, (long long) ne01, (long long) ne11, mn, mx);
            for (int64_t i = 0; i < std::min<int64_t>(16, n_idx); i++) { fprintf(stderr, " %d", h[i]); }
            fprintf(stderr, " last8:");
            for (int64_t i = std::max<int64_t>(0, n_idx-8); i < n_idx; i++) { fprintf(stderr, " %d", h[i]); }
            fprintf(stderr, "\n");
            fflush(stderr);
            sp_cmp_row = h[0]; // compare after the kernel runs
        }
    }

    // LLAMA_SPARSE_TOPK_WATCH="lo:hi[:max_prints]": per-launch count of DSA
    // top-k gather indices falling in cell range [lo,hi). Discriminates
    // selection-side failures (needle cells never selected) from
    // attention/gather-side failures (selected but dropped) on eager (non-
    // captured) passes. Inert unless the env is set.
    {
        static const auto watch = []() -> std::tuple<int32_t,int32_t,int> {
            const char * e = getenv("LLAMA_SPARSE_TOPK_WATCH");
            if (e == nullptr) return {0, 0, 0};
            int32_t lo = 0, hi = 0; int mx = 64;
            if (sscanf(e, "%d:%d:%d", &lo, &hi, &mx) >= 2 && hi > lo) return {lo, hi, mx};
            return {0, 0, 0};
        }();
        const auto [w_lo, w_hi, w_max] = watch;
        // optional file gate (LLAMA_SPARSE_TOPK_WATCH_GATE=<path>): watch (and
        // its per-launch device sync) is active only while <path> exists, so a
        // specific request can be instrumented without perturbing the ones
        // before it (state-dependent bugs may hide under the extra syncs).
        static const char * w_gate = getenv("LLAMA_SPARSE_TOPK_WATCH_GATE");
        cudaStreamCaptureStatus w_cs = cudaStreamCaptureStatusNone;
        if (w_hi > w_lo && src1->name[0] != '\0' && strstr(src1->name, "top_k") != nullptr &&
                (w_gate == nullptr || access(w_gate, F_OK) == 0) &&
                cudaStreamIsCapturing(stream, &w_cs) == cudaSuccess && w_cs == cudaStreamCaptureStatusNone) {
            static std::atomic<int> n_watch{0};
            if (n_watch.fetch_add(1) < w_max) {
                const int64_t n_idx = ggml_nelements(src1);
                std::vector<int32_t> h((size_t) n_idx);
                CUDA_CHECK(cudaMemcpyAsync(h.data(), src1->data, h.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                int64_t in_range = 0; int32_t mn = INT32_MAX, mx2 = INT32_MIN;
                for (int64_t i = 0; i < n_idx; i++) {
                    if (h[(size_t) i] >= w_lo && h[(size_t) i] < w_hi) in_range++;
                    mn = std::min(mn, h[(size_t) i]); mx2 = std::max(mx2, h[(size_t) i]);
                }
                int w_dev = -1; cudaGetDevice(&w_dev);
                fprintf(stderr, "[TOPKWATCH] dev=%d dst=%s n_idx=%lld watch=[%d,%d) hit=%lld min=%d max=%d\n",
                        w_dev, dst->name, (long long) n_idx, w_lo, w_hi, (long long) in_range, mn, mx2);
                fflush(stderr);
            }
        }
    }

    // LLAMA_CACHE_ROWSUM="lo:hi[:max]": FNV checksum of RAW cache rows
    // [lo,hi) of src0 (the sharded MLA cache) on eager top-k gather launches.
    // Bit-identical prompts must give bit-identical row sums; a mismatch
    // between a failing and a passing run of the SAME prompt pins the
    // corruption on the cache WRITE side (prefill set_rows), an identical sum
    // pins it on the compute/exchange side. Optional file gate via
    // LLAMA_SPARSE_TOPK_WATCH_GATE (same file as the top-k watch).
    {
        static const auto rowsum = []() -> std::tuple<int64_t,int64_t,int> {
            const char * e = getenv("LLAMA_CACHE_ROWSUM");
            if (e == nullptr) return {0, 0, 0};
            long lo = 0, hi = 0; int mx = 64;
            if (sscanf(e, "%ld:%ld:%d", &lo, &hi, &mx) >= 2 && hi > lo) return {lo, hi, mx};
            return {0, 0, 0};
        }();
        const auto [r_lo, r_hi, r_max] = rowsum;
        static const char * r_gate = getenv("LLAMA_SPARSE_TOPK_WATCH_GATE");
        cudaStreamCaptureStatus r_cs = cudaStreamCaptureStatusNone;
        // member-0 raw cache only (sh_base == 0): its LOCAL rows equal global
        // cell ids, and the checked range must be inside its owned extent.
        // src0's view ne[1] is proportionally distributed and cannot be used
        // as a bound (that is what op_params[2] is for).
        const int32_t r_mode = ((const int32_t *) dst->op_params)[1];
        const int32_t r_base = ((const int32_t *) dst->op_params)[0];
        const int32_t r_rows = ((const int32_t *) dst->op_params)[2];
        if (r_hi > r_lo && src1->name[0] != '\0' && strstr(src1->name, "top_k") != nullptr &&
                strncmp(dst->name, "s2b_local_k-", 12) == 0 &&
                (r_gate == nullptr || access(r_gate, F_OK) == 0) &&
                r_mode == 1 && r_base == 0 && (int64_t) r_rows >= r_hi &&
                cudaStreamIsCapturing(stream, &r_cs) == cudaSuccess && r_cs == cudaStreamCaptureStatusNone) {
            static std::atomic<int> n_sums{0};
            if (n_sums.fetch_add(1) < r_max) {
                const int64_t n_rows = r_hi - r_lo;
                std::vector<uint8_t> h((size_t) (n_rows*nb01));
                CUDA_CHECK(cudaMemcpyAsync(h.data(), (const char *) src0->data + r_lo*nb01,
                        h.size(), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                int r_dev = -1; cudaGetDevice(&r_dev);
                fprintf(stderr, "[ROWSUM] dev=%d dst=%s rows=[%lld,%lld) nb01=%zu sums:",
                        r_dev, dst->name, (long long) r_lo, (long long) r_hi, (size_t) nb01);
                for (int64_t b = 0; b < n_rows; b += 16) {
                    uint64_t f = 1469598103934665603ULL;
                    const size_t lo_b = (size_t) (b*nb01);
                    const size_t hi_b = std::min(h.size(), (size_t) ((b + 16)*nb01));
                    for (size_t k = lo_b; k < hi_b; k++) {
                        f = (f ^ h[k]) * 1099511628211ULL;
                    }
                    fprintf(stderr, " %lld:%08x", (long long) (r_lo + b), (uint32_t) (f & 0xffffffff));
                }
                fprintf(stderr, "\n");
                fflush(stderr);
            }
        }
    }

    // KV-shard sparse gather (meta-injected op_params): [0]=member row base,
    // [1]=shard mode flag; rebase + zero-fill foreign rows (see meta backend)
    const int32_t sh_mode = ((const int32_t *) dst->op_params)[1];
    const int64_t sh_base = sh_mode != 0 ? ((const int32_t *) dst->op_params)[0] : 0;
    // owned extent comes from op_params (parent-cache split), NOT the view's
    // proportionally-distributed ne01
    const int64_t sh_rows = sh_mode != 0 ? ((const int32_t *) dst->op_params)[2] : 0;

    if (sh_mode == 2) {
        GGML_ASSERT(ne00 == 1 &&
            (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16));
        const int threads = 256;
        const dim3 blocks((unsigned) ((ne10 + threads - 1)/threads),
                          (unsigned) (ne11*ne12), 1);
        if (dst->type == GGML_TYPE_F16) {
            k_s2b_ownership_mask<<<blocks, threads, 0, stream>>>(
                (const int32_t *) src1->data, (half *) dst->data,
                ne10, ne11, ne12,
                nb10/sizeof(int32_t), nb11/sizeof(int32_t), nb12/sizeof(int32_t),
                nb1/sizeof(half), nb2/sizeof(half), nb3/sizeof(half),
                sh_base, sh_rows);
        } else {
            k_s2b_ownership_mask<<<blocks, threads, 0, stream>>>(
                (const int32_t *) src1->data, (float *) dst->data,
                ne10, ne11, ne12,
                nb10/sizeof(int32_t), nb11/sizeof(int32_t), nb12/sizeof(int32_t),
                nb1/sizeof(float), nb2/sizeof(float), nb3/sizeof(float),
                sh_base, sh_rows);
        }
        CUDA_CHECK(cudaGetLastError());
        // LLAMA_S2B_AUDIT2: per-device dump of the generated-ownership-mask
        // inputs (top-k idx identity across members + owned counts). First 8
        // launches per device (covers ~2 decode steps on the tiny model).
        static const bool s2b_audit2 = getenv("LLAMA_S2B_AUDIT2") != nullptr;
        if (s2b_audit2) {
            static std::atomic<int> a2_count[GGML_CUDA_MAX_DEVICES];
            int dev = -1;
            CUDA_CHECK(cudaGetDevice(&dev));
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            if (dev >= 0 && dev < GGML_CUDA_MAX_DEVICES &&
                    cudaStreamIsCapturing(stream, &capture) == cudaSuccess &&
                    capture == cudaStreamCaptureStatusNone &&
                    a2_count[dev].fetch_add(1) < 8) {
                const int64_t n_idx = ne10*ne11*ne12;
                std::vector<int32_t> hidx((size_t) n_idx);
                CUDA_CHECK(cudaMemcpyAsync(hidx.data(), src1->data, hidx.size()*sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                uint64_t fnv = 1469598103934665603ULL;
                int64_t owned = 0;
                int32_t mn = INT32_MAX, mx = INT32_MIN;
                for (int64_t i = 0; i < n_idx; ++i) {
                    fnv = (fnv ^ (uint32_t) hidx[(size_t) i]) * 1099511628211ULL;
                    mn = std::min(mn, hidx[(size_t) i]);
                    mx = std::max(mx, hidx[(size_t) i]);
                    if (hidx[(size_t) i] >= sh_base && hidx[(size_t) i] < sh_base + sh_rows) {
                        owned++;
                    }
                }
                fprintf(stderr,
                        "[S2BAUDIT2-OWNGEN] dev=%d name=%s base=%lld rows=%lld n_idx=%lld owned=%lld min=%d max=%d idxhash=%016llx\n",
                        dev, dst->name, (long long) sh_base, (long long) sh_rows,
                        (long long) n_idx, (long long) owned, mn, mx,
                        (unsigned long long) fnv);
                fflush(stderr);
            }
        }
        static const bool s2b_audit_own = getenv("LLAMA_S2B_AUDIT") != nullptr;
        if (s2b_audit_own && dst->type == GGML_TYPE_F32) {
            static std::atomic<uint32_t> audited_devices{0};
            int dev = -1;
            CUDA_CHECK(cudaGetDevice(&dev));
            const uint32_t bit = dev >= 0 && dev < 32 ? 1u << dev : 0;
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            if (bit != 0 && cudaStreamIsCapturing(stream, &capture) == cudaSuccess &&
                    capture == cudaStreamCaptureStatusNone &&
                    (audited_devices.fetch_or(bit) & bit) == 0) {
                std::vector<float> h((size_t) ggml_nelements(dst));
                CUDA_CHECK(cudaMemcpyAsync(h.data(), dst->data, h.size()*sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                size_t zeros = 0, neg_inf = 0, other = 0;
                for (float x : h) {
                    if (x == 0.0f) zeros++;
                    else if (isinf(x) && x < 0.0f) neg_inf++;
                    else other++;
                }
                fprintf(stderr,
                        "[S2BAUDIT-OWNGEN] dev=%d base=%lld rows=%lld zero=%zu -inf=%zu other=%zu first16=",
                        dev, (long long) sh_base, (long long) sh_rows, zeros, neg_inf, other);
                for (size_t i = 0; i < std::min<size_t>(16, h.size()); ++i) {
                    fprintf(stderr, "%c", h[i] == 0.0f ? 'U' : isinf(h[i]) && h[i] < 0.0f ? '.' : '?');
                }
                fprintf(stderr, "\n");
                fflush(stderr);
            }
        }
    } else if (sh_mode == 3) {
        // shared-top-k verify: batch-tail collision mask (see kernel comment).
        // src0 = I32 tail input view [1, n_tail]; op_params are build-time
        // constants (member-independent, safe under graph reuse).
        GGML_ASSERT(ne00 == 1 && ne11 == 1 && ne12 == 1);
        GGML_ASSERT(src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F16);
        const int64_t n_tail = src0->ne[1];
        const int threads = 256;
        const dim3 blocks((unsigned) ((ne10 + threads - 1)/threads), 1, 1);
        k_spec_tail_collision_mask<<<blocks, threads, 0, stream>>>(
            (const int32_t *) src0->data, (const int32_t *) src1->data, (half *) dst->data,
            ne10, n_tail, nb10/sizeof(int32_t), nb1/sizeof(half));
        CUDA_CHECK(cudaGetLastError());
    } else {
        get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,
            ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, sh_base, sh_rows);
    }

    // One-launch S2b audit. For layer 0 on every device, verify the global
    // selection/ownership split and sample the actual post-kernel rows. A
    // foreign row must be exactly zero; an owned row must be populated.
    static const bool s2b_audit = getenv("LLAMA_S2B_AUDIT") != nullptr;
    if (s2b_audit && strcmp(dst->name, "s2b_local_k-0") == 0 &&
            dst->type == GGML_TYPE_F32 && sh_mode != 0) {
        static std::atomic<uint32_t> audited_devices{0};
        int dev = -1;
        CUDA_CHECK(cudaGetDevice(&dev));
        const uint32_t bit = dev >= 0 && dev < 32 ? 1u << dev : 0;
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (bit != 0 && cudaStreamIsCapturing(stream, &capture) == cudaSuccess &&
                capture == cudaStreamCaptureStatusNone &&
                (audited_devices.fetch_or(bit) & bit) == 0) {
            const int64_t n_idx = ggml_nelements(src1);
            std::vector<int32_t> idx((size_t) n_idx);
            CUDA_CHECK(cudaMemcpyAsync(idx.data(), src1->data, idx.size()*sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            int64_t n_owned = 0, n_foreign = 0;
            int64_t first_owned = -1, first_foreign = -1;
            int32_t mn = INT32_MAX, mx = INT32_MIN;
            for (int64_t i = 0; i < n_idx; ++i) {
                mn = std::min(mn, idx[(size_t) i]);
                mx = std::max(mx, idx[(size_t) i]);
                const bool owned = idx[(size_t) i] >= sh_base &&
                    idx[(size_t) i] < sh_base + sh_rows;
                if (owned) {
                    n_owned++;
                    if (first_owned < 0) first_owned = i;
                } else {
                    n_foreign++;
                    if (first_foreign < 0) first_foreign = i;
                }
            }
            auto row_stats = [&](int64_t row, float & max_abs, float & sum_abs, float first[4]) {
                max_abs = 0.0f;
                sum_abs = 0.0f;
                std::vector<float> h((size_t) ne00);
                if (row >= 0) {
                    CUDA_CHECK(cudaMemcpyAsync(h.data(),
                            (const char *) dst->data + row*nb1,
                            h.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    for (float x : h) {
                        max_abs = std::max(max_abs, fabsf(x));
                        sum_abs += fabsf(x);
                    }
                    for (int i = 0; i < 4; ++i) first[i] = i < ne00 ? h[(size_t) i] : 0.0f;
                } else {
                    for (int i = 0; i < 4; ++i) first[i] = 0.0f;
                }
            };
            float own_max, own_sum, foreign_max, foreign_sum;
            float own_first[4], foreign_first[4];
            row_stats(first_owned, own_max, own_sum, own_first);
            row_stats(first_foreign, foreign_max, foreign_sum, foreign_first);
            fprintf(stderr,
                    "[S2BAUDIT-ROWS] dev=%d base=%lld rows=%lld idx=%lld range=[%d,%d] owned=%lld foreign=%lld "
                    "owned_pos=%lld gidx=%d max=%.6g sum=%.6g first=%.5g,%.5g,%.5g,%.5g "
                    "foreign_pos=%lld gidx=%d max=%.6g sum=%.6g first=%.5g,%.5g,%.5g,%.5g\n",
                    dev, (long long) sh_base, (long long) sh_rows, (long long) n_idx, mn, mx,
                    (long long) n_owned, (long long) n_foreign,
                    (long long) first_owned, first_owned >= 0 ? idx[(size_t) first_owned] : -1,
                    own_max, own_sum, own_first[0], own_first[1], own_first[2], own_first[3],
                    (long long) first_foreign, first_foreign >= 0 ? idx[(size_t) first_foreign] : -1,
                    foreign_max, foreign_sum,
                    foreign_first[0], foreign_first[1], foreign_first[2], foreign_first[3]);
            fflush(stderr);
        }
    }

    // gather-vs-truth (post-kernel): dequantize cache row sp_cmp_row host-side
    // (q8_0) and compare against the gathered dst row 0 (F32)
    if (sp_cmp_row >= 0 && src0->type == GGML_TYPE_Q8_0 && dst->type == GGML_TYPE_F32) {
        const int64_t n_el = ne00; // full-row verification
        std::vector<uint8_t> raw(nb01);
        CUDA_CHECK(cudaMemcpyAsync(raw.data(), (const char *) src0->data + sp_cmp_row*nb01, nb01, cudaMemcpyDeviceToHost, stream));
        std::vector<float> g(n_el);
        CUDA_CHECK(cudaMemcpyAsync(g.data(), dst->data, n_el*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float max_diff = 0.0f;
        int64_t diff_at = -1;
        for (int64_t i = 0; i < n_el; i++) {
            const uint8_t * blk = raw.data() + (i/32)*34; // block_q8_0: f16 d + 32x int8
            const float d = __half2float(*(const __half *) blk);
            const int8_t qv = ((const int8_t *) (blk + 2))[i % 32];
            const float diff = fabsf(d * qv - g[i]);
            if (diff > max_diff) { max_diff = diff; diff_at = i; }
        }
        fprintf(stderr, "[SPCMP] row=%lld ne00=%lld nb01=%zu max_diff=%.6g at=%lld first4: %.4g %.4g %.4g %.4g\n",
                (long long) sp_cmp_row, (long long) ne00, (size_t) nb01, max_diff, (long long) diff_at,
                g.size() > 0 ? g[0] : 0.f, g.size() > 1 ? g[1] : 0.f, g.size() > 2 ? g[2] : 0.f, g.size() > 3 ? g[3] : 0.f);
        fflush(stderr);
    }
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, MIN(ne1, (int64_t)UINT16_MAX), 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10, ne1);
}
