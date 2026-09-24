#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

template <typename Tdata>
class PagedCachingKernel {
public:
    __aicore__ inline PagedCachingKernel() {}

    __aicore__ inline void init(
        GM_ADDR k_cache,
        GM_ADDR v_cache,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR slot_mapping,
        size_t num_kv_heads,
        size_t head_size,
        size_t v_head_size,
        size_t block_size,
        ptrdiff_t k_src_stride,
        ptrdiff_t v_src_stride,
        ptrdiff_t k_src_head_stride,
        ptrdiff_t v_src_head_stride,
        ptrdiff_t k_cache_block_stride,
        ptrdiff_t v_cache_block_stride,
        ptrdiff_t k_cache_head_stride,
        ptrdiff_t v_cache_head_stride,
        ptrdiff_t k_cache_slot_stride,
        ptrdiff_t v_cache_slot_stride) {
        _num_kv_heads = num_kv_heads;
        _head_size = head_size;
        _v_head_size = v_head_size;
        _block_size = block_size;
        _k_src_stride = k_src_stride;
        _v_src_stride = v_src_stride;
        _k_src_head_stride = k_src_head_stride;
        _v_src_head_stride = v_src_head_stride;
        _k_cache_block_stride = k_cache_block_stride;
        _v_cache_block_stride = v_cache_block_stride;
        _k_cache_head_stride = k_cache_head_stride;
        _v_cache_head_stride = v_cache_head_stride;
        _k_cache_slot_stride = k_cache_slot_stride;
        _v_cache_slot_stride = v_cache_slot_stride;

        _k_cache_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(k_cache));
        _v_cache_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(v_cache));
        _k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(k));
        _v_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(v));
        _slot_mapping_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slot_mapping));

        // Fast path: the per-(token, head) K/V slice is contiguous and every
        // stride is 32B-aligned in both source and destination, so a single
        // GM->UB->GM DataCopy pair replaces head_size scalar GetValue/SetValue
        // round trips (each of which is a separate GM transaction).
        const ptrdiff_t align_elems = static_cast<ptrdiff_t>(BYTE_ALIGN / sizeof(Tdata));
        _k_fast = (static_cast<ptrdiff_t>(head_size) % align_elems == 0) &&
                  (k_src_stride % align_elems == 0) &&
                  (k_src_head_stride % align_elems == 0) &&
                  (k_cache_block_stride % align_elems == 0) &&
                  (k_cache_head_stride % align_elems == 0) &&
                  (k_cache_slot_stride % align_elems == 0);
        _v_fast = (static_cast<ptrdiff_t>(v_head_size) % align_elems == 0) &&
                  (v_src_stride % align_elems == 0) &&
                  (v_src_head_stride % align_elems == 0) &&
                  (v_cache_block_stride % align_elems == 0) &&
                  (v_cache_head_stride % align_elems == 0) &&
                  (v_cache_slot_stride % align_elems == 0);

        if (_k_fast) {
            _pipe.InitBuffer(_k_buf, alignTileLen<Tdata>(_head_size, BYTE_ALIGN) * sizeof(Tdata));
        }
        if (_v_fast) {
            _pipe.InitBuffer(_v_buf, alignTileLen<Tdata>(_v_head_size, BYTE_ALIGN) * sizeof(Tdata));
        }
    }

    __aicore__ inline void copyKV() {
        const size_t work_idx = GetBlockIdx();
        const size_t head_idx = work_idx % _num_kv_heads;
        const size_t token_idx = work_idx / _num_kv_heads;

        const int64_t slot_idx = _slot_mapping_gm.GetValue(token_idx);
        if (slot_idx < 0) {
            return;
        }

        const int64_t physical_block_idx = slot_idx / static_cast<int64_t>(_block_size);
        const int64_t block_offset = slot_idx % static_cast<int64_t>(_block_size);

        const ptrdiff_t k_src_base = static_cast<ptrdiff_t>(token_idx) * _k_src_stride
                                   + static_cast<ptrdiff_t>(head_idx) * _k_src_head_stride;
        const ptrdiff_t v_src_base = static_cast<ptrdiff_t>(token_idx) * _v_src_stride
                                   + static_cast<ptrdiff_t>(head_idx) * _v_src_head_stride;
        const ptrdiff_t k_dst_base = static_cast<ptrdiff_t>(physical_block_idx) * _k_cache_block_stride
                                   + static_cast<ptrdiff_t>(head_idx) * _k_cache_head_stride
                                   + static_cast<ptrdiff_t>(block_offset) * _k_cache_slot_stride;
        const ptrdiff_t v_dst_base = static_cast<ptrdiff_t>(physical_block_idx) * _v_cache_block_stride
                                   + static_cast<ptrdiff_t>(head_idx) * _v_cache_head_stride
                                   + static_cast<ptrdiff_t>(block_offset) * _v_cache_slot_stride;

        if (_k_fast && _v_fast) {
            // Both slices are contiguous and 32B-aligned: vector copy.
            LocalTensor<Tdata> k_local = _k_buf.Get<Tdata>();
            LocalTensor<Tdata> v_local = _v_buf.Get<Tdata>();
            DataCopy(k_local, _k_gm[k_src_base], _head_size);
            DataCopy(v_local, _v_gm[v_src_base], _v_head_size);
            PipeBarrier<PIPE_ALL>();
            DataCopy(_k_cache_gm[k_dst_base], k_local, _head_size);
            DataCopy(_v_cache_gm[v_dst_base], v_local, _v_head_size);
            PipeBarrier<PIPE_ALL>();
            return;
        }

        if (_k_fast) {
            LocalTensor<Tdata> k_local = _k_buf.Get<Tdata>();
            DataCopy(k_local, _k_gm[k_src_base], _head_size);
            PipeBarrier<PIPE_ALL>();
            DataCopy(_k_cache_gm[k_dst_base], k_local, _head_size);
            PipeBarrier<PIPE_ALL>();
        } else {
            for (size_t d = 0; d < _head_size; ++d) {
                _k_cache_gm.SetValue(k_dst_base + static_cast<ptrdiff_t>(d),
                                     _k_gm.GetValue(k_src_base + static_cast<ptrdiff_t>(d)));
            }
        }

        if (_v_fast) {
            LocalTensor<Tdata> v_local = _v_buf.Get<Tdata>();
            DataCopy(v_local, _v_gm[v_src_base], _v_head_size);
            PipeBarrier<PIPE_ALL>();
            DataCopy(_v_cache_gm[v_dst_base], v_local, _v_head_size);
            PipeBarrier<PIPE_ALL>();
        } else {
            for (size_t d = 0; d < _v_head_size; ++d) {
                _v_cache_gm.SetValue(v_dst_base + static_cast<ptrdiff_t>(d),
                                     _v_gm.GetValue(v_src_base + static_cast<ptrdiff_t>(d)));
            }
        }
    }

private:
    GlobalTensor<Tdata> _k_cache_gm;
    GlobalTensor<Tdata> _v_cache_gm;
    GlobalTensor<Tdata> _k_gm;
    GlobalTensor<Tdata> _v_gm;
    GlobalTensor<int64_t> _slot_mapping_gm;

    TPipe _pipe;
    TBuf<TPosition::VECCALC> _k_buf;
    TBuf<TPosition::VECCALC> _v_buf;

    size_t _num_kv_heads;
    size_t _head_size;
    size_t _v_head_size;
    size_t _block_size;
    ptrdiff_t _k_src_stride;
    ptrdiff_t _v_src_stride;
    ptrdiff_t _k_src_head_stride;
    ptrdiff_t _v_src_head_stride;
    ptrdiff_t _k_cache_block_stride;
    ptrdiff_t _v_cache_block_stride;
    ptrdiff_t _k_cache_head_stride;
    ptrdiff_t _v_cache_head_stride;
    ptrdiff_t _k_cache_slot_stride;
    ptrdiff_t _v_cache_slot_stride;
    bool _k_fast;
    bool _v_fast;
};

#define DEFINE_PAGED_CACHING_KERNEL(KERNEL_NAME, TYPE)                 \
    extern "C" __global__ __aicore__ void KERNEL_NAME(                 \
        GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR k, GM_ADDR v,        \
        GM_ADDR slot_mapping, size_t num_kv_heads, size_t head_size,   \
        size_t v_head_size, size_t block_size, ptrdiff_t k_src_stride, \
        ptrdiff_t v_src_stride, ptrdiff_t k_src_head_stride,           \
        ptrdiff_t v_src_head_stride, ptrdiff_t k_cache_block_stride,   \
        ptrdiff_t v_cache_block_stride, ptrdiff_t k_cache_head_stride, \
        ptrdiff_t v_cache_head_stride, ptrdiff_t k_cache_slot_stride,  \
        ptrdiff_t v_cache_slot_stride) {                               \
        PagedCachingKernel<TYPE> op;                                   \
        op.init(k_cache, v_cache, k, v, slot_mapping, num_kv_heads,    \
                head_size, v_head_size, block_size, k_src_stride,      \
                v_src_stride, k_src_head_stride, v_src_head_stride,    \
                k_cache_block_stride, v_cache_block_stride,            \
                k_cache_head_stride, v_cache_head_stride,              \
                k_cache_slot_stride, v_cache_slot_stride);             \
        op.copyKV();                                                   \
    }

DEFINE_PAGED_CACHING_KERNEL(paged_caching_kernel_f16, half)
DEFINE_PAGED_CACHING_KERNEL(paged_caching_kernel_bf16, bfloat16_t)
DEFINE_PAGED_CACHING_KERNEL(paged_caching_kernel_f32, float)

#undef DEFINE_PAGED_CACHING_KERNEL

extern "C" infiniStatus_t paged_caching_kernel_launch(
    void *k_cache,
    void *v_cache,
    const void *k,
    const void *v,
    const void *slot_mapping,
    infiniDtype_t dtype,
    size_t num_tokens,
    size_t num_kv_heads,
    size_t head_size,
    size_t v_head_size,
    size_t block_size,
    ptrdiff_t k_src_stride,
    ptrdiff_t v_src_stride,
    ptrdiff_t k_src_head_stride,
    ptrdiff_t v_src_head_stride,
    ptrdiff_t k_cache_block_stride,
    ptrdiff_t v_cache_block_stride,
    ptrdiff_t k_cache_head_stride,
    ptrdiff_t v_cache_head_stride,
    ptrdiff_t k_cache_slot_stride,
    ptrdiff_t v_cache_slot_stride,
    void *stream) {
    const size_t block_dim = num_tokens * num_kv_heads;
    if (block_dim == 0) {
        return INFINI_STATUS_SUCCESS;
    }

#define LAUNCH_PAGED_CACHING(DTYPE_ENUM, KERNEL_NAME)                       \
    case DTYPE_ENUM:                                                        \
        KERNEL_NAME<<<block_dim, nullptr, stream>>>(                        \
            k_cache, v_cache, const_cast<void *>(k), const_cast<void *>(v), \
            const_cast<void *>(slot_mapping), num_kv_heads,                 \
            head_size, v_head_size, block_size, k_src_stride, v_src_stride, \
            k_src_head_stride, v_src_head_stride,                           \
            k_cache_block_stride, v_cache_block_stride,                     \
            k_cache_head_stride, v_cache_head_stride,                       \
            k_cache_slot_stride, v_cache_slot_stride);                      \
        return INFINI_STATUS_SUCCESS;

    switch (dtype) {
        LAUNCH_PAGED_CACHING(INFINI_DTYPE_F16, paged_caching_kernel_f16)
        LAUNCH_PAGED_CACHING(INFINI_DTYPE_BF16, paged_caching_kernel_bf16)
        LAUNCH_PAGED_CACHING(INFINI_DTYPE_F32, paged_caching_kernel_f32)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_PAGED_CACHING
}

