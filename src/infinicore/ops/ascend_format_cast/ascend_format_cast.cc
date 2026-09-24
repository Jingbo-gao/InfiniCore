#include "infinicore/ops/ascend_format_cast.hpp"

#include "infinicore/context/context.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ENABLE_ASCEND_API
#include "../../../infiniop/devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_trans_matmul_weight.h>
#endif

namespace infinicore::op {

#ifdef ENABLE_ASCEND_API
namespace {

void check_acl(aclnnStatus status, const char *api) {
    if (status == ACL_SUCCESS) {
        return;
    }
    const char *detail = aclGetRecentErrMsg();
    throw std::runtime_error(
        std::string(api) + " failed with status " + std::to_string(status)
        + (detail == nullptr ? "" : std::string(": ") + detail));
}

std::vector<int64_t> as_int64(const Shape &shape) {
    return std::vector<int64_t>(shape.begin(), shape.end());
}

std::vector<int64_t> as_int64(const Strides &strides) {
    return std::vector<int64_t>(strides.begin(), strides.end());
}

} // namespace
#endif

Tensor ascend_format_cast_nz(const Tensor &src) {
#ifndef ENABLE_ASCEND_API
    (void)src;
    throw std::runtime_error(
        "ascend_format_cast_nz requires an InfiniCore build with Ascend enabled");
#else
    if (src->device().getType() != Device::Type::ASCEND) {
        throw std::invalid_argument(
            "ascend_format_cast_nz requires an Ascend tensor");
    }
    if (src->ndim() != 2 || !src->is_contiguous()) {
        throw std::invalid_argument(
            "ascend_format_cast_nz requires a contiguous 2-D tensor");
    }
    if (src->dtype() != DataType::F16 && src->dtype() != DataType::BF16) {
        throw std::invalid_argument(
            "ascend_format_cast_nz only supports FP16 and BF16 weights");
    }
    if (src->size(0) % 16 != 0 || src->size(1) % 16 != 0) {
        throw std::invalid_argument(
            "ascend_format_cast_nz requires both dimensions to be multiples of 16");
    }

    // aclnnMatmulWeightNz requires its right-hand weight to be preprocessed by
    // aclnnTransMatmulWeight. Generic NpuFormatCast packing is not equivalent.
    const auto logical_shape = as_int64(src->shape());
    const auto logical_strides = as_int64(src->strides());
    const auto acl_dtype = toAclDataType(src->desc()->dtype());

    auto dst = Tensor::empty(src->shape(), src->dtype(), src->device());
    dst->copy_from(src);

    // The official API sample uses a flat storage shape for mutable mmWeightRef.
    // K and N are 16-aligned, so the affinity layout needs no extra allocation.
    const std::vector<int64_t> storage_shape{
        static_cast<int64_t>(dst->numel())};
    aclnnTensorDescriptor weight_desc(
        acl_dtype,
        logical_shape,
        logical_strides,
        ACL_FORMAT_ND,
        storage_shape,
        dst->data());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    const auto prepare_status = aclnnTransMatmulWeightGetWorkspaceSize(
        weight_desc.tensor, &workspace_size, &executor);
    if (prepare_status != ACL_SUCCESS || executor == nullptr) {
        const char *detail = aclGetRecentErrMsg();
        std::fprintf(
            stderr,
            "[ascend_format_cast_nz] shape=[%lld,%lld] status=%d detail=%s\n",
            logical_shape[0], logical_shape[1],
            static_cast<int>(prepare_status),
            detail == nullptr ? "(null)" : detail);
    }
    check_acl(prepare_status, "aclnnTransMatmulWeightGetWorkspaceSize");
    if (executor == nullptr) {
        throw std::runtime_error(
            "aclnnTransMatmulWeightGetWorkspaceSize returned null executor");
    }

    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size != 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    auto stream = reinterpret_cast<aclrtStream>(context::getStream());
    check_acl(aclnnTransMatmulWeight(
                  workspace_ptr, workspace_size, executor, stream),
              "aclnnTransMatmulWeight");
    check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    return dst;
#endif
}

} // namespace infinicore::op
