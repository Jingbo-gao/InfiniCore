#include "swiglu_ascend.h"
#include "../../../devices/ascend/common_ascend.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

namespace op::swiglu::ascend {
Descriptor::~Descriptor() = default;

// swiglu 并行度优化：目标 block 数与切分路径的确定
//   env INFINICORE_SWIGLU_BLOCK_NUM 未设 : 按硬件 AIV 核数（新默认，M 维切分）
//   env = 8   : 严格 legacy（kernel 侧走原 hidden 一维切分，逐行复现旧行为）
//   env = N>8 : 强制 N（仍走新路径 M 维切分）
//   env = N<8 (1..7) : 【冲突】N 与 legacy(8) 语义无法并存，统一归一到 legacy（block_num=8）并 warn
//   env 非法/非正 : 按未设处理（新默认）
//   acl 查询失败 / 核数异常偏小 : 回退 legacy（block_num=8，安全回退）
//
// ★ 关键：legacy 判定（切分路径）由 host 显式决定，并经 legacy 标志随 block_num 一并传入 kernel。
//   kernel 不再以 block_num<=BLOCK_NUM 隐式推断 legacy，杜绝「launch block 数 != 实际有效切分数」的错配。
namespace {
constexpr size_t SWIGLU_BLOCK_NUM_FALLBACK = 8;  // = ascend_kernel_common.h 的 BLOCK_NUM
constexpr size_t SWIGLU_BLOCK_NUM_CAP = 64;      // 上限，防异常值越界

size_t resolve_swiglu_block_num(size_t m, size_t aiv_num) {
    // m = batch*seq（token 数），block 数再按可用 token 收敛，避免空核
    size_t block_num = (aiv_num > 0) ? aiv_num : SWIGLU_BLOCK_NUM_FALLBACK;
    if (block_num == 0) {
        block_num = SWIGLU_BLOCK_NUM_FALLBACK;
    }
    if (block_num > SWIGLU_BLOCK_NUM_CAP) {
        block_num = SWIGLU_BLOCK_NUM_CAP;
    }
    if (m > 0 && block_num > m) {
        block_num = m;
    }
    return block_num;
}

// 按调用打点去重：(M, block_num, legacy) 组合首次出现时打印一条。
// 目的：prefill / decode 各自的 M 与生效 block_num/legacy 都能被观测到，
//       同时避免逐层刷屏（80 层 x 多次调用只打各不同组合一次）。
void log_swiglu_call_once(size_t m, size_t block_num, bool legacy) {
    static std::mutex s_mtx;
    static std::set<std::string> s_seen;
    char key[64];
    std::snprintf(key, sizeof(key), "%zu|%zu|%d", m, block_num, legacy ? 1 : 0);
    std::lock_guard<std::mutex> lk(s_mtx);
    if (s_seen.insert(std::string(key)).second) {
        printf("[swiglu-call] M=%zu block_num=%zu legacy=%d\n", m, block_num, legacy ? 1 : 0);
        fflush(stdout);
    }
}
} // namespace

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t c_desc,
                                  std::vector<infiniopTensorDescriptor_t> input_descs) {
    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);

    auto dtype = c_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

    const auto &a_desc = input_descs[0];
    const auto &b_desc = input_descs[1];

    auto result = SwigluInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);
    SwigluInfo info = result.take();

    // https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0777.html
    size_t workspace_size = 0;

    // 读取一次 env（进程内不变，避免每次 forward getenv）并解析 block 数配置
    //   new_default : 未设置 or 非法 -> 按 AIV 核数
    //   forced      : env 正整数 N>8
    //   legacy      : env == 8（严格 legacy 路径）
    //   env N<8     : 归一到 legacy 并 warn（消除 host/kernel 阈值歧义）
    size_t forced_block_num = 0;   // 0 表示未强制
    bool legacy_requested = false;
    const char *env_str = std::getenv("INFINICORE_SWIGLU_BLOCK_NUM");
    if (env_str != nullptr) {
        char *end = nullptr;
        long long env_val = std::strtoll(env_str, &end, 10);
        if (end != env_str && env_val > 0) {
            if (env_val == static_cast<long long>(SWIGLU_BLOCK_NUM_FALLBACK)) {
                // env == 8：严格 legacy
                legacy_requested = true;
                forced_block_num = 0;
            } else if (env_val < static_cast<long long>(SWIGLU_BLOCK_NUM_FALLBACK)) {
                // env == 1..7：与 legacy 语义冲突，统一归一到 legacy（block_num=8）并 warn
                legacy_requested = true;
                forced_block_num = 0;
                static bool warned_conflict = false;
                if (!warned_conflict) {
                    warned_conflict = true;
                    printf("[swiglu-opt] INFINICORE_SWIGLU_BLOCK_NUM=%lld (<8) 与 legacy 语义冲突，"
                           "按 8 (legacy) 处理\n",
                           env_val);
                }
            } else {
                // env == N>8：强制 N（新路径）
                forced_block_num = static_cast<size_t>(env_val);
            }
        } else {
            static bool warned = false;
            if (!warned) {
                warned = true;
                printf("[swiglu-opt] invalid INFINICORE_SWIGLU_BLOCK_NUM=\"%s\", ignore (use new default)\n", env_str);
            }
        }
    }

    *desc_ptr = new Descriptor(std::move(info), workspace_size, handle_ascend->device, handle_ascend->device_id,
                               forced_block_num, legacy_requested);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *c,
                                     std::vector<const void *> inputs,
                                     void *stream) const {
    auto batch = _info.ndim == 2 ? 1 : _info.shape[0];
    auto seq_len = _info.ndim == 2 ? _info.shape[0] : _info.shape[1];
    auto hidden_size = _info.shape[_info.ndim - 1];
    auto stride_batch_c = _info.ndim == 2 ? 1 : _info.c_strides[0];
    auto stride_batch_a = _info.ndim == 2 ? 1 : _info.a_strides[0];
    auto stride_batch_b = _info.ndim == 2 ? 1 : _info.b_strides[0];
    auto stride_seq_c = _info.ndim == 2 ? _info.c_strides[0] : _info.c_strides[1];
    auto stride_seq_a = _info.ndim == 2 ? _info.a_strides[0] : _info.a_strides[1];
    auto stride_seq_b = _info.ndim == 2 ? _info.b_strides[0] : _info.b_strides[1];

    const size_t m = batch * seq_len;

    // 1) 唯一确定 (block_num, used_legacy) 二元组；launch block 数与切分路径严格同源
    size_t block_num = 0;
    size_t hw_aiv = 0;
    aclError acl_ret = ACL_ERROR_NONE;
    bool used_legacy = false;

    if (_legacy_requested) {
        // env=8 或 env=1..7（已归一）: 严格 legacy（kernel 侧 hidden 一维切分，固定 8 block）
        block_num = SWIGLU_BLOCK_NUM_FALLBACK;
        used_legacy = true;
    } else if (_forced_block_num > 0) {
        // env=N>8：强制 N（新路径 M 维切分）。N 恒 > 8，与 kernel 的 legacy 判定不冲突。
        block_num = _forced_block_num;
        if (block_num > SWIGLU_BLOCK_NUM_CAP) {
            block_num = SWIGLU_BLOCK_NUM_CAP;
        }
        if (m > 0 && block_num > m) {
            block_num = m;
        }
        // 多核收敛后可能 <= 8（小 M 场景），此时必须显式回退 legacy，
        // 否则 kernel 新路径 rows_per_block 与 legacy 语义重叠。→ 保持与 kernel 一致。
        if (block_num <= SWIGLU_BLOCK_NUM_FALLBACK) {
            used_legacy = true;
            block_num = SWIGLU_BLOCK_NUM_FALLBACK;
        }
        if (m == 0) {
            used_legacy = true;
            block_num = SWIGLU_BLOCK_NUM_FALLBACK;
        }
    } else {
        // 未设置：按硬件 AIV 核数（aclrtGetDeviceInfo），失败回退 8
        int64_t aiv_num = 0;
        acl_ret = aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv_num);
        if (acl_ret == ACL_SUCCESS && aiv_num > 0) {
            hw_aiv = static_cast<size_t>(aiv_num);
        }
        block_num = resolve_swiglu_block_num(m, hw_aiv);
        if (block_num <= SWIGLU_BLOCK_NUM_FALLBACK) {
            // 核数异常偏小/读失败 -> 走 legacy，保证与旧行为一致（安全回退）
            used_legacy = true;
        }
    }

    // 2) 总开关状态打点（每个进程仅一次，避免刷屏）
    static std::once_flag swiglu_opt_log_once;
    std::call_once(swiglu_opt_log_once, [&]() {
        if (used_legacy) {
            if (_legacy_requested) {
                printf("[swiglu-opt] disabled/legacy block_num=8\n");
            } else {
                printf("[swiglu-opt] disabled/legacy block_num=%zu (aclrtGetDeviceInfo ret=%d, hw aiv=%zu)\n",
                       block_num, static_cast<int>(acl_ret), hw_aiv);
            }
        } else {
            printf("[swiglu-opt] enabled block_num=%zu (hw aiv=%zu, M=%zu)\n", block_num, hw_aiv, m);
        }
        fflush(stdout);
    });

    // 3) 按调用打点（仅 (M, block_num, legacy) 组合首次出现时打印）：
    //    可观测 prefill / decode 各自生效的 M 与 block_num/legacy，且不刷屏。
    //    打印值即最终 launch 值，与 kernel 实际执行严格一致。
    log_swiglu_call_once(m, block_num, used_legacy);

    auto status = swiglu_kernel_launch(c, (void *)inputs[0], (void *)inputs[1], _info.dtype, batch, seq_len, hidden_size, stride_batch_c, stride_batch_a, stride_batch_b, stride_seq_c, stride_seq_a, stride_seq_b, block_num, used_legacy, stream);
    return status;
}

} // namespace op::swiglu::ascend