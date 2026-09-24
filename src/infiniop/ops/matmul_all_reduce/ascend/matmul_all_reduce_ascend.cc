#include "matmul_all_reduce_ascend.h"
#include "../../../devices/ascend/common_ascend.h"

#include <aclnnop/aclnn_matmul_all_reduce.h>

#include <memory>
#include <string>

namespace op::matmul_all_reduce::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t output;
    aclnnTensorDescriptor_t input;
    aclnnTensorDescriptor_t weight;
    aclnnTensorDescriptor_t bias;
    std::string group_name;

    ~Opaque() {
        delete output;
        delete input;
        delete weight;
        delete bias;
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    const char *group_name) {
    return createWithFormat(
        handle_, desc_ptr, output_desc, input_desc, weight_desc, bias_desc,
        group_name, false);
}

infiniStatus_t Descriptor::createWithFormat(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    const char *group_name,
    bool weight_is_fractal_nz) {
    if (desc_ptr == nullptr || group_name == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
    CHECK_API_OR(output_desc->dtype() == dtype, true,
                 return INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_API_OR(weight_desc->dtype() == dtype, true,
                 return INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_API_OR(bias_desc == nullptr || bias_desc->dtype() == dtype, true,
                 return INFINI_STATUS_BAD_TENSOR_DTYPE);

    CHECK_API_OR(input_desc->ndim() == 2, true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(weight_desc->ndim() == 2, true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(output_desc->ndim() == 2, true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(bias_desc == nullptr || bias_desc->ndim() == 1, true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);

    const auto &input_shape = input_desc->shape();
    const auto &weight_shape = weight_desc->shape();
    const auto &output_shape = output_desc->shape();
    CHECK_API_OR(input_shape[1] == weight_shape[0], true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(output_shape[0] == input_shape[0], true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(output_shape[1] == weight_shape[1], true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(bias_desc == nullptr || bias_desc->shape()[0] == output_shape[1],
                 true, return INFINI_STATUS_BAD_TENSOR_SHAPE);

    auto output = new aclnnTensorDescriptor(output_desc);
    auto input = new aclnnTensorDescriptor(input_desc);
    aclnnTensorDescriptor_t weight = nullptr;
    if (weight_is_fractal_nz) {
        const int64_t k = static_cast<int64_t>(weight_shape[0]);
        const int64_t n = static_cast<int64_t>(weight_shape[1]);
        CHECK_API_OR(k % 16 == 0 && n % 16 == 0, true,
                     return INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_API_OR(weight_desc->stride(0)
                             == static_cast<ptrdiff_t>(weight_shape[1])
                         && weight_desc->stride(1) == 1,
                     true, return INFINI_STATUS_BAD_TENSOR_STRIDES);
        weight = new aclnnTensorDescriptor(
            toAclDataType(dtype),
            {k, n}, {n, 1},
            ACL_FORMAT_FRACTAL_NZ,
            {n / 16, k / 16, 16, 16});
    } else {
        weight = new aclnnTensorDescriptor(weight_desc);
    }
    auto bias = bias_desc == nullptr
                  ? nullptr
                  : new aclnnTensorDescriptor(bias_desc);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnMatmulAllReduceGetWorkspaceSize(
        input->tensor,
        weight->tensor,
        bias == nullptr ? nullptr : bias->tensor,
        group_name,
        "sum",
        0,
        1,
        output->tensor,
        &workspace_size,
        &executor));
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    *desc_ptr = new Descriptor(
        workspace_size,
        new Opaque{output, input, weight, bias, group_name},
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream) const {
    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    aclnnTensorDescriptor call_input(
        _opaque->input->dataType, _opaque->input->shape,
        _opaque->input->strides, _opaque->input->format,
        _opaque->input->storageShape, const_cast<void *>(input));
    aclnnTensorDescriptor call_weight(
        _opaque->weight->dataType, _opaque->weight->shape,
        _opaque->weight->strides, _opaque->weight->format,
        _opaque->weight->storageShape, const_cast<void *>(weight));
    aclnnTensorDescriptor call_output(
        _opaque->output->dataType, _opaque->output->shape,
        _opaque->output->strides, _opaque->output->format,
        _opaque->output->storageShape, output);
    std::unique_ptr<aclnnTensorDescriptor> call_bias;
    if (_opaque->bias != nullptr) {
        call_bias = std::make_unique<aclnnTensorDescriptor>(
            _opaque->bias->dataType, _opaque->bias->shape,
            _opaque->bias->strides, _opaque->bias->format,
            _opaque->bias->storageShape, const_cast<void *>(bias));
    }

    uint64_t required_workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnMatmulAllReduceGetWorkspaceSize(
        call_input.tensor,
        call_weight.tensor,
        call_bias == nullptr ? nullptr : call_bias->tensor,
        _opaque->group_name.c_str(),
        "sum",
        0,
        1,
        call_output.tensor,
        &required_workspace_size,
        &executor));
    if (workspace_size < required_workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto status = aclnnMatmulAllReduce(
        workspace, required_workspace_size,
        executor, stream);
    CHECK_ACL(status);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::matmul_all_reduce::ascend
