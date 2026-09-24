#include "infinicore/ops/argmax.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"
#endif

#ifdef ENABLE_ASCEND_API
#include "../../../infiniop/devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_cast.h>
#endif

namespace infinicore::op {
namespace {

#ifdef ENABLE_ASCEND_API

void check_argmax_acl(aclnnStatus status, const char *api) {
    if (status == ACL_SUCCESS) {
        return;
    }
    const char *detail = aclGetRecentErrMsg();
    throw std::runtime_error(
        std::string(api) + " failed with status "
        + std::to_string(status)
        + (detail == nullptr ? std::string()
                             : std::string(": ") + detail));
}

std::vector<int64_t> as_int64(const Shape &shape) {
    return std::vector<int64_t>(shape.begin(), shape.end());
}

std::vector<int64_t> as_int64(const Strides &strides) {
    return std::vector<int64_t>(strides.begin(), strides.end());
}

void run_argmax_acl(const Tensor &input, Tensor &output, int64_t dim) {
    const auto shape = as_int64(input->shape());
    const auto strides = as_int64(input->strides());
    const auto out_shape = as_int64(output->shape());
    const auto out_strides = as_int64(output->strides());

    aclnnTensorDescriptor input_desc(
        toAclDataType(input->desc()->dtype()), shape, strides,
        ACL_FORMAT_ND, shape, const_cast<std::byte *>(input->data()));
    aclnnTensorDescriptor output_desc(
        toAclDataType(output->desc()->dtype()), out_shape, out_strides,
        ACL_FORMAT_ND, out_shape, output->data());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    check_argmax_acl(
        aclnnArgMaxGetWorkspaceSize(
            input_desc.tensor, dim, false, output_desc.tensor,
            &workspace_size, &executor),
        "aclnnArgMaxGetWorkspaceSize");

    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size != 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }
    auto stream = reinterpret_cast<aclrtStream>(context::getStream());
    check_argmax_acl(
        aclnnArgMax(workspace_ptr, workspace_size, executor, stream),
        "aclnnArgMax");
}

void run_cast_acl(const Tensor &input, Tensor &output) {
    const auto shape = as_int64(input->shape());
    const auto strides = as_int64(input->strides());

    aclnnTensorDescriptor input_desc(
        toAclDataType(input->desc()->dtype()), shape, strides,
        ACL_FORMAT_ND, shape, const_cast<std::byte *>(input->data()));
    aclnnTensorDescriptor output_desc(
        toAclDataType(output->desc()->dtype()), shape, strides,
        ACL_FORMAT_ND, shape, output->data());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    check_argmax_acl(
        aclnnCastGetWorkspaceSize(
            input_desc.tensor, ACL_FLOAT, output_desc.tensor,
            &workspace_size, &executor),
        "aclnnCastGetWorkspaceSize");

    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size != 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }
    auto stream = reinterpret_cast<aclrtStream>(context::getStream());
    check_argmax_acl(
        aclnnCast(workspace_ptr, workspace_size, executor, stream),
        "aclnnCast");
}

void argmax_ascend(Tensor output, const Tensor &input, int64_t dim) {
    if (!input->is_contiguous() || !output->is_contiguous()) {
        throw std::runtime_error(
            "argmax_ ascend path expects contiguous input/output tensors");
    }

    switch (input->dtype()) {
    case DataType::F32:
    case DataType::F16:
        run_argmax_acl(input, output, dim);
        return;
    case DataType::BF16: {
        // aclnnArgMax documents FLOAT/FLOAT16 inputs only. BF16 logits are
        // therefore widened to F32 first; this is exact for BF16 values.
        auto widened = Tensor::empty(
            input->shape(), DataType::F32, input->device());
        run_cast_acl(input, widened);
        run_argmax_acl(widened, output, dim);
        return;
    }
    default:
        throw std::runtime_error(
            "argmax_ does not support this input dtype on Ascend");
    }
}

#endif // ENABLE_ASCEND_API

void check_argmax_args(const Tensor &output, const Tensor &input, int64_t dim) {
    if (!output || !input) {
        throw std::runtime_error("argmax_ expects non-null tensors");
    }
    if (input->ndim() == 0) {
        throw std::runtime_error("argmax_ expects a non-scalar input");
    }
    if (output->device().getType() != input->device().getType()
        || output->device().getIndex() != input->device().getIndex()) {
        throw std::runtime_error("argmax_ expects input/output on one device");
    }
    const auto ndim = static_cast<int64_t>(input->ndim());
    if (dim < -ndim || dim >= ndim) {
        throw std::runtime_error("argmax_ dim is out of range");
    }
    auto expected_shape = input->shape();
    expected_shape.erase(
        expected_shape.begin() + (dim < 0 ? dim + ndim : dim));
    if (output->shape() != expected_shape) {
        throw std::runtime_error(
            "argmax_ output shape does not match input shape minus dim");
    }
    if (output->dtype() != DataType::I32
        && output->dtype() != DataType::I64) {
        throw std::runtime_error("argmax_ output must be int32 or int64");
    }
}

} // namespace

Tensor argmax(const Tensor &input, int64_t dim) {
    if (!input || input->ndim() == 0) {
        throw std::runtime_error("argmax expects a non-scalar input");
    }
    const auto ndim = static_cast<int64_t>(input->ndim());
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("argmax dim is out of range");
    }
    auto output_shape = input->shape();
    output_shape.erase(output_shape.begin() + dim);
    auto output = Tensor::empty(
        output_shape, DataType::I64, input->device());
    argmax_(output, input, dim);
    return output;
}

void argmax_(Tensor output, const Tensor &input, int64_t dim) {
    check_argmax_args(output, input, dim);

#ifdef ENABLE_ASCEND_API
    if (input->device().getType() == Device::Type::ASCEND) {
        const auto ndim = static_cast<int64_t>(input->ndim());
        const auto normalized_dim = dim < 0 ? dim + ndim : dim;
        argmax_ascend(output, input, normalized_dim);
        return;
    }
#endif

#if defined(ENABLE_ATEN)
    adaptor::set_aten_stream_to_infinicore();
    auto output_at = adaptor::to_aten_tensor(output);
    const auto input_at = adaptor::to_aten_tensor(input);
    output_at.copy_(input_at.argmax(dim, false));
    return;
#else
    throw std::runtime_error(
        "argmax_ is only implemented for Ascend and ATen builds");
#endif
}

} // namespace infinicore::op
