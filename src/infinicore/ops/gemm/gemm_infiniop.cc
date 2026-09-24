#include "../infiniop_impl.hpp"
#include "infinicore/ops/gemm.hpp"

namespace infinicore::op::gemm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Gemm, 100);

thread_local common::OpCache<size_t, std::shared_ptr<Descriptor>> nz_caches(
    100,
    [](std::shared_ptr<Descriptor> &desc) {
        desc = nullptr;
    });

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, a, b;
    float alpha, beta;
};

void *plan(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    size_t seed = hash_combine(c, a, b);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Gemm,
        seed, c->desc(), a->desc(), b->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Gemm, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        alpha, beta};

    return planned;
}

void *plan_nz(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    size_t seed = hash_combine(c, a, b);
    std::shared_ptr<Descriptor> descriptor;
    {
        auto device = context::getDevice();
        auto &cache = nz_caches.getCache(device);
        descriptor = cache.get(seed).value_or(nullptr);
        if (!descriptor) {
            descriptor = std::make_shared<Descriptor>(nullptr);
            INFINICORE_CHECK_ERROR(infiniopCreateGemmNzDescriptor(
                context::getInfiniopHandle(device),
                &descriptor->desc,
                c->desc(), a->desc(), b->desc()));
            cache.put(seed, descriptor);
        }
    }

    INFINIOP_WORKSPACE_TENSOR(workspace, Gemm, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        alpha, beta};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopGemm(
        planned->descriptor->desc, planned->workspace->data(), planned->workspace->numel(),
        planned->c->data(), planned->a->data(), planned->b->data(), planned->alpha, planned->beta, context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Gemm, &plan, &run, &cleanup);

#ifdef ENABLE_ASCEND_API
static bool nz_registered = []() {
    GemmNz::plan_dispatcher().registerDevice(Device::Type::ASCEND, &plan_nz);
    GemmNz::run_dispatcher().registerDevice(Device::Type::ASCEND, &run);
    GemmNz::cleanup_dispatcher().registerDevice(Device::Type::ASCEND, &cleanup);
    return true;
}();
#endif

} // namespace infinicore::op::gemm_impl::infiniop
