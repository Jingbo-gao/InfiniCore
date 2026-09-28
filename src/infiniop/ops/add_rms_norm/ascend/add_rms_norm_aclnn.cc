#include "add_rms_norm_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_add_rms_norm.h>

namespace op::add_rms_norm::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t x1;    // a（Add 第一个输入）
    aclnnTensorDescriptor_t x2;    // b（Add 第二个输入）
    aclnnTensorDescriptor_t gamma; // weight（RmsNorm 缩放因子）
    aclnnTensorDescriptor_t y;     // y（RmsNorm 输出）
    aclnnTensorDescriptor_t rstd;  // dummy（官方注明当前产品场景下无效，但需非空）
    aclnnTensorDescriptor_t x;     // residual_out（= x1 + x2）
    size_t workspaceSize;          // 官方 GetWorkspaceSize 返回的 workspace 大小
    size_t rstd_size;              // rstd dummy 的字节数
    aclOpExecutor *executor;

    Opaque(aclnnTensorDescriptor_t x1_, aclnnTensorDescriptor_t x2_,
           aclnnTensorDescriptor_t gamma_, aclnnTensorDescriptor_t y_,
           aclnnTensorDescriptor_t rstd_, aclnnTensorDescriptor_t x_,
           size_t ws, size_t rstd_sz, aclOpExecutor *exec)
        : x1(x1_), x2(x2_), gamma(gamma_), y(y_), rstd(rstd_), x(x_),
          workspaceSize(ws), rstd_size(rstd_sz), executor(exec) {}

    ~Opaque() {
        delete x1;
        delete x2;
        delete gamma;
        delete y;
        delete rstd;
        delete x;
        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t residual_out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t weight_desc,
    float epsilon) {

    auto result = AddRMSNormInfo::create(y_desc, residual_out_desc, a_desc, b_desc, weight_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    // aclnnAddRmsNorm 官方约束：gamma 的数据类型需与 x1 保持一致。
    // 跨半精度（F16/BF16 互换）或 F32 weight 需先 cast 到 atype，本版暂不支持，
    // 直接报错（LLM 常见场景 weight 与 activation 同 dtype，如 BF16/BF16）。
    if (info.wtype != info.atype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);

    aclnnTensorDescriptor_t x1 = new aclnnTensorDescriptor(a_desc);
    aclnnTensorDescriptor_t x2 = new aclnnTensorDescriptor(b_desc);
    aclnnTensorDescriptor_t gamma = new aclnnTensorDescriptor(weight_desc);
    aclnnTensorDescriptor_t y = new aclnnTensorDescriptor(y_desc);
    aclnnTensorDescriptor_t x = new aclnnTensorDescriptor(residual_out_desc);

    // rstdOut：官方文档注明「在当前产品使用场景下无效」，但「不支持空 Tensor」，
    // 故仍需构造一个合法的 dummy tensor。shape = x1 前 ndim-1 维 + [1]，dtype 必须 FLOAT32。
    //   例：x1 [batch, dim]          -> rstd [batch, 1]
    //       x1 [batch, nhead, dim]   -> rstd [batch, nhead, 1]
    std::vector<int64_t> rstd_shape;
    rstd_shape.reserve(info.ndim());
    for (size_t i = 0; i + 1 < info.ndim(); ++i) {
        rstd_shape.push_back(static_cast<int64_t>(info.shape[i]));
    }
    rstd_shape.push_back(1);
    std::vector<int64_t> rstd_strides(rstd_shape.size(), 1);
    for (ptrdiff_t i = static_cast<ptrdiff_t>(rstd_shape.size()) - 2; i >= 0; --i) {
        rstd_strides[i] = rstd_strides[i + 1] * rstd_shape[i + 1];
    }
    aclnnTensorDescriptor_t rstd = new aclnnTensorDescriptor(
        toAclDataType(INFINI_DTYPE_F32), rstd_shape, rstd_strides);

    size_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    // 两段式接口：先 GetWorkspaceSize 拿 workspace 大小与 executor，再 call 执行。
    // 参数顺序（tensor 参数，跳过标量 epsilon）即 AclSetTensorAddr 的 index：
    //   x1=0, x2=1, gamma=2, yOut=3, rstdOut=4, xOut=5
    CHECK_ACL(aclnnAddRmsNormGetWorkspaceSize(
        x1->tensor,
        x2->tensor,
        gamma->tensor,
        static_cast<double>(epsilon),
        y->tensor,
        rstd->tensor,
        x->tensor,
        &workspace_size,
        &executor));

    aclSetAclOpExecutorRepeatable(executor);

    // workspace 布局：[官方 workspace_size][rstd dummy]。
    // rstd dummy 紧跟官方 workspace 之后（官方 workspace 通常已对齐，F32 仅需 4 字节对齐）。
    size_t rstd_size = rstd->numel() * aclDataTypeSize(rstd->dataType);
    size_t all_workspace_size = workspace_size + rstd_size;

    auto *opaque = new Opaque{x1, x2, gamma, y, rstd, x, workspace_size, rstd_size, executor};
    *desc_ptr = new Descriptor(
        opaque,
        std::move(info),
        all_workspace_size,
        handle_ascend->device,
        handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, void *residual_out, const void *a, const void *b, const void *weight,
    void *stream) const {

    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto *wsb = static_cast<uint8_t *>(workspace);
    // rstd dummy 地址：紧跟官方 workspace 之后，内容无效，仅满足「非空」约束。
    void *rstd_ptr = wsb + _opaque->workspaceSize;

    AclSetTensorAddr(_opaque->executor, 0, _opaque->x1->tensor, const_cast<void *>(a));
    AclSetTensorAddr(_opaque->executor, 1, _opaque->x2->tensor, const_cast<void *>(b));
    AclSetTensorAddr(_opaque->executor, 2, _opaque->gamma->tensor, const_cast<void *>(weight));
    AclSetTensorAddr(_opaque->executor, 3, _opaque->y->tensor, y);
    AclSetTensorAddr(_opaque->executor, 4, _opaque->rstd->tensor, rstd_ptr);
    AclSetTensorAddr(_opaque->executor, 5, _opaque->x->tensor, residual_out);

    CHECK_ACL(aclnnAddRmsNorm(
        workspace, _opaque->workspaceSize, _opaque->executor, stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::add_rms_norm::ascend
