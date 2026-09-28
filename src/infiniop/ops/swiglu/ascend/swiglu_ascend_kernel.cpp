#include "../../../devices/ascend/ascend_kernel_common.h"
#include <type_traits> // for std::is_same

using namespace AscendC;

// swiglu 并行度优化：
//   新默认路径 = M 维（token 行）切分 x 整行 hidden，block 数由 host 按硬件 AIV 核数传入；
//   legacy 路径 = 原 hidden 一维切分（逐行保留原实现语义），用于 A/B 基线与安全回退。
//
// ★ legacy 判定由 host 显式决定（env=8 / env=1..7 归一 / 核数回退 / 小 M 收敛），
//   随 legacy 标志传入 kernel。kernel 严格以该标志二选一，绝不依据 block_num 数值隐式推断，
//   从而保证「launch block 数」与「实际有效切分数」在所有 env 取值下都一致。
constexpr size_t SWIGLU_BLOCK_NUM_CAP = 64;

template <typename T>
class SwigluKernel {
public:
    __aicore__ inline SwigluKernel() {}
    __aicore__ inline void init(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                size_t batch_, size_t seq, size_t hd,
                                ptrdiff_t stride_batch_c,
                                ptrdiff_t stride_batch_a,
                                ptrdiff_t stride_batch_b,
                                ptrdiff_t stride_seq_c,
                                ptrdiff_t stride_seq_a,
                                ptrdiff_t stride_seq_b,
                                size_t block_num, bool legacy);
    __aicore__ inline void process();

private:
    __aicore__ inline void copyIn(size_t i);
    __aicore__ inline void compute(size_t i);
    __aicore__ inline void copyOut(size_t i);

private:
    GlobalTensor<T> _c_gm, _a_gm, _b_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> _in_queue_a, _in_queue_b;
    TQue<QuePosition::VECOUT, BUFFER_NUM> _out_queue_c;
    TBuf<QuePosition::VECCALC> _tempFp16_a, _tempFp16_b, _tempFp16_c;

    TPipe _pipe;
    float _beta_value = 1.0f;
    size_t _block_idx, _tile_len, _copy_len,
        _batch, _seq_len, _hidden_size,
        _stride_seq_a, _stride_seq_b, _stride_seq_c;
    int64_t _stride_batch_a = 1, _stride_batch_b = 1, _stride_batch_c = 1;

    // M 维（token 行）切分状态：仅新路径使用
    size_t _row_start = 0, _row_end = 0;
    bool _legacy = false;
};

template <typename T>
__aicore__ inline void SwigluKernel<T>::init(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                             size_t batch_, size_t seq, size_t hd,
                                             ptrdiff_t stride_batch_c,
                                             ptrdiff_t stride_batch_a,
                                             ptrdiff_t stride_batch_b,
                                             ptrdiff_t stride_seq_c,
                                             ptrdiff_t stride_seq_a,
                                             ptrdiff_t stride_seq_b,
                                             size_t block_num, bool legacy) {
    // Init Shape & StrideVariables
    _batch = batch_;
    _seq_len = seq;
    _hidden_size = hd;
    _stride_batch_a = stride_batch_a;
    _stride_batch_b = stride_batch_b;
    _stride_batch_c = stride_batch_c;
    _stride_seq_a = stride_seq_a;
    _stride_seq_b = stride_seq_b;
    _stride_seq_c = stride_seq_c;

    _block_idx = GetBlockIdx();
    _legacy = legacy; // 切分路径由 host 显式传入，kernel 不做数值推断

    if (_legacy) {
        // ---- legacy 路径：hidden 一维切分（与优化前逐行一致）----
        // 注意：legacy 必须固定按 BLOCK_NUM(=8) 份切，故 host 侧在 legacy 时固定 launch 8 个 block，
        //       保证「launch block 数 == 切分份数」，杜绝只算部分 hidden 的静默错误。
        _tile_len = _block_idx < (_hidden_size % BLOCK_NUM) ? (_hidden_size / BLOCK_NUM) + 1 : (_hidden_size / BLOCK_NUM);
        _row_start = 0;
        _row_end = _batch * _seq_len;
    } else {
        // ---- 新路径：M 维（token 行）切分 x 整行 hidden ----
        size_t m = _batch * _seq_len;
        size_t rows_per_block = (m + block_num - 1) / block_num;
        _row_start = rows_per_block == 0 ? 0 : _block_idx * rows_per_block;
        _row_end = _row_start + rows_per_block;
        if (_row_end > m) {
            _row_end = m;
        }
        if (_row_start >= _row_end) {
            return; // 空 block 早退（避免空转/越界）
        }
        _tile_len = _hidden_size; // 整行 hidden
    }
    _copy_len = alignTileLen<T>(_tile_len, BYTE_ALIGN);

    // Set global tensor
    _a_gm.SetGlobalBuffer((__gm__ T *)a);
    _b_gm.SetGlobalBuffer((__gm__ T *)b);
    _c_gm.SetGlobalBuffer((__gm__ T *)c);

    // _pipe alloc memory to queue, the unit is bytes
    _pipe.InitBuffer(_in_queue_a, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_b, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_out_queue_c, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_tempFp16_a, _copy_len * sizeof(float));
    _pipe.InitBuffer(_tempFp16_b, _copy_len * sizeof(float));
    _pipe.InitBuffer(_tempFp16_c, _copy_len * sizeof(float));
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::copyIn(size_t i) {
    // Alloc tensor from queue memory
    LocalTensor<T> aLocal = _in_queue_a.AllocTensor<T>();
    LocalTensor<T> bLocal = _in_queue_b.AllocTensor<T>();
    // Get idx of current tile
    auto batch_idx = _batch == 1 ? 0 : i / _seq_len;
    auto seq_idx = _batch == 1 ? i : i % _seq_len;

    // 新路径整行 hidden -> 无列偏移；legacy 路径保留 _block_idx*_tile_len 列偏移
    ptrdiff_t col_off = _legacy ? static_cast<ptrdiff_t>(_block_idx * _tile_len) : 0;
    ptrdiff_t idxa = batch_idx * _stride_batch_a + seq_idx * _stride_seq_a + col_off;
    ptrdiff_t idxb = batch_idx * _stride_batch_b + seq_idx * _stride_seq_b + col_off;
    // Copy process_th tile from global tensor to local tensor
    DataCopy(aLocal, _a_gm[idxa], _copy_len);
    DataCopy(bLocal, _b_gm[idxb], _copy_len);

    // Enque input tensor to VECIN queue
    _in_queue_a.EnQue(aLocal);
    _in_queue_b.EnQue(bLocal);
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::compute(size_t i) {
    LocalTensor<T> aLocal = _in_queue_a.DeQue<T>();
    LocalTensor<T> bLocal = _in_queue_b.DeQue<T>();
    LocalTensor<T> cLocal = _out_queue_c.AllocTensor<T>();

    // BF16 类型特殊处理：先 cast 到 float，执行 float 版 SwiGLU，再 cast 回 BF16
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        // 从 TBuf 获取 float 类型的临时 tensor
        LocalTensor<float> aFloat = _tempFp16_a.Get<float>();
        LocalTensor<float> bFloat = _tempFp16_b.Get<float>();
        LocalTensor<float> cFloat = _tempFp16_c.Get<float>();

        // BF16 -> float
        Cast(aFloat, aLocal, AscendC::RoundMode::CAST_NONE, _copy_len);
        Cast(bFloat, bLocal, AscendC::RoundMode::CAST_NONE, _copy_len);

        // 执行 float 版本的 SwiGLU
        SwiGLU<float, false>(cFloat, aFloat, bFloat, _beta_value, _copy_len);

        // float -> BF16
        Cast(cLocal, cFloat, AscendC::RoundMode::CAST_RINT, _copy_len);

    } else {
        // 原有的 F16/F32 逻辑
        SwiGLU<T, false>(cLocal, aLocal, bLocal, _beta_value, _copy_len);
    }

    _out_queue_c.EnQue<T>(cLocal);
    _in_queue_a.FreeTensor(aLocal);
    _in_queue_b.FreeTensor(bLocal);
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::copyOut(size_t i) {
    // Deque output tensor from VECOUT queue
    LocalTensor<T> cLocal = _out_queue_c.DeQue<T>();
    auto batch_idx = _batch == 1 ? 0 : i / _seq_len;
    auto seq_idx = _batch == 1 ? i : i % _seq_len;
    ptrdiff_t col_off = _legacy ? static_cast<ptrdiff_t>(_block_idx * _tile_len) : 0;
    ptrdiff_t idxc = batch_idx * _stride_batch_c + seq_idx * _stride_seq_c + col_off;
    // Copy progress_th tile from local tensor to global tensor
    if (_tile_len * sizeof(T) % BYTE_ALIGN != 0) {
        DataCopyExtParams dcep = {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0};
        DataCopyPad(_c_gm[idxc], cLocal, dcep);
    } else {
        DataCopy(_c_gm[idxc], cLocal, _tile_len);
    }
    // Free output Local tensor
    _out_queue_c.FreeTensor(cLocal);
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::process() {
    // 新路径：只遍历本 block 负责的 token 行区间；legacy 路径：[0, M) 全量（原行为）
    for (size_t i = _row_start; i < _row_end; ++i) {
        copyIn(i);
        compute(i);
        copyOut(i);
    }
}

#define DEFINE_SWIGLU_KERNEL(KERNEL_NAME, TYPE)                                 \
    __global__ __aicore__ void KERNEL_NAME(GM_ADDR c, GM_ADDR a, GM_ADDR b,     \
                                           size_t batch, size_t seq, size_t hd, \
                                           ptrdiff_t stride_batch_c,            \
                                           ptrdiff_t stride_batch_a,            \
                                           ptrdiff_t stride_batch_b,            \
                                           ptrdiff_t stride_seq_c,              \
                                           ptrdiff_t stride_seq_a,              \
                                           ptrdiff_t stride_seq_b,              \
                                           size_t block_num, bool legacy) {     \
        SwigluKernel<TYPE> op;                                                  \
        op.init(c, a, b,                                                        \
                batch, seq, hd,                                                 \
                stride_batch_c, stride_batch_a, stride_batch_b,                 \
                stride_seq_c, stride_seq_a, stride_seq_b,                       \
                block_num, legacy);                                             \
        op.process();                                                           \
    }

DEFINE_SWIGLU_KERNEL(swiglu_kernel_half, half)
DEFINE_SWIGLU_KERNEL(swiglu_kernel_float, float)
DEFINE_SWIGLU_KERNEL(swiglu_kernel_bf16, bfloat16_t)

#undef DEFINE_SWIGLU_KERNEL

extern "C" infiniStatus_t swiglu_kernel_launch(
    void *c, void *a, void *b,
    infiniDtype_t dtype, size_t batch, size_t seq, size_t hd,
    ptrdiff_t stride_batch_c, ptrdiff_t stride_batch_a, ptrdiff_t stride_batch_b,
    ptrdiff_t stride_seq_c, ptrdiff_t stride_seq_a, ptrdiff_t stride_seq_b,
    size_t block_num, bool legacy, void *stream) {

    // 防御：block_num 至少 1，且不超过硬上限（host 侧已 clamp，此处双保险）
    if (block_num == 0) {
        block_num = 1;
    }
    if (block_num > SWIGLU_BLOCK_NUM_CAP) {
        block_num = SWIGLU_BLOCK_NUM_CAP;
    }
    // legacy 必须按 BLOCK_NUM(=8) 份切，故严格固定 launch 8 个 block，保证与切分份数一致。
    if (legacy) {
        block_num = BLOCK_NUM;
    }

#define LAUNCH_SWIGLU_KERNEL(DTYPE_ENUM, KERNEL_NAME)       \
    case DTYPE_ENUM:                                        \
        KERNEL_NAME<<<block_num, nullptr, stream>>>(        \
            c, a, b,                                        \
            batch,                                          \
            seq,                                            \
            hd,                                             \
            stride_batch_c, stride_batch_a, stride_batch_b, \
            stride_seq_c, stride_seq_a, stride_seq_b,       \
            block_num, legacy);                             \
        return INFINI_STATUS_SUCCESS;

    switch (dtype) {
        LAUNCH_SWIGLU_KERNEL(INFINI_DTYPE_F16, swiglu_kernel_half)
        LAUNCH_SWIGLU_KERNEL(INFINI_DTYPE_F32, swiglu_kernel_float)
        LAUNCH_SWIGLU_KERNEL(INFINI_DTYPE_BF16, swiglu_kernel_bf16) // 用 float kernel
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_SWIGLU_KERNEL
}
