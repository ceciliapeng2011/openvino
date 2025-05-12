// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "multi_stage_primitive.hpp"

#include "lora_inst.h"
#include "lora.hpp"
#include "lora/lora_kernel_selector.h"
#include "lora/lora_kernel_base.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>
#include "intel_gpu/graph/fused_primitive_desc.hpp"

namespace cldnn {
namespace ocl {
dnnl::memory::data_type convert_data_type(cldnn::data_types dt);
dnnl::memory convert2dnnl(const memory::ptr& ptr, const std::vector<int64_t>& dim, dnnl::memory::format_tag tag, int offset = 0) ;

dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
    switch (dt) {
    case cldnn::data_types::f32:
        return dnnl::memory::data_type::f32;
    case cldnn::data_types::f16:
        return dnnl::memory::data_type::f16;
    case cldnn::data_types::i8:
        return dnnl::memory::data_type::s8;
    case cldnn::data_types::u8:
        return dnnl::memory::data_type::u8;
    case cldnn::data_types::i32:
        return dnnl::memory::data_type::s32;
    case cldnn::data_types::i4:
        return dnnl::memory::data_type::s4;
    case cldnn::data_types::u4:
        return dnnl::memory::data_type::u4;
    default:
        throw std::invalid_argument("[clDNN] Unsupported conversion from cldnn to onednn post_ops_type");
    }
}

struct onednn_matmul {
    dnnl::matmul m_prim;
    dnnl::memory::desc m_wei_md;
    dnnl::memory::data_type m_w_type;
    dnnl::memory::data_type m_a_type;  // activation dtype
    dnnl::memory::dim m_K;
    dnnl::memory::dim m_N;
    dnnl::memory::dim m_M;
    dnnl::memory::dim m_K_groups;

    dnnl::primitive_attr attr;
    dnnl::post_ops postops;

    onednn_matmul(dnnl::memory::data_type act_dtype, dnnl::memory::data_type weight_dtype, int batch_size, int ic, int oc) {
        m_a_type = act_dtype;
        m_w_type = weight_dtype;
        m_K_groups = 0;
        m_K = ic;
        m_N = oc;
        m_M = DNNL_RUNTIME_DIM_VAL;
        if (batch_size > 0) {
            // jit-gemm kernel only support static batch size
            m_M = batch_size;
        }
    }

    onednn_matmul& fpmath_f16() {
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
        return *this;
    }
    onednn_matmul& post_op_silu() {
        float alpha = 1.0f;
        float beta = 0.0f;
        postops.append_eltwise(dnnl::algorithm::eltwise_swish, alpha, beta);
        return *this;
    }
    onednn_matmul& post_op_bin_mul(bool per_oc = true, bool broad_cast = false) {
        dnnl::memory::dim batch_size = m_M;
        if (batch_size == DNNL_RUNTIME_DIM_VAL)
            batch_size = 1024 * 1024;  // big enough fake static batch

        dnnl::memory::desc bin_mul_md = dnnl::memory::desc(dnnl::memory::dims({broad_cast ? 1 : batch_size, per_oc ? m_N : 1}), m_a_type, dnnl::memory::format_tag::ab);
        postops.append_binary(dnnl::algorithm::binary_mul, bin_mul_md);
        return *this;
    }
    onednn_matmul& post_op_bin_add(bool per_oc = true) {
        dnnl::memory::dim batch_size = m_M;
        if (batch_size == DNNL_RUNTIME_DIM_VAL)
            batch_size = 1024*1024; // big enough fake static batch

        dnnl::memory::desc bin_add_md = dnnl::memory::desc(dnnl::memory::dims({batch_size, per_oc ? m_N : 1}), m_a_type, dnnl::memory::format_tag::ab);
        postops.append_binary(dnnl::algorithm::binary_add, bin_add_md);
        return *this;
    }

    onednn_matmul& post_op_sum(float scale = 1.f, int32_t zero_point = 0) {
        postops.append_sum(scale, zero_point, dnnl::memory::data_type::undef);
        return *this;
    }

    void create(dnnl::engine eng) {
        if (postops.len() > 0) {
            attr.set_post_ops(postops);
        }

        dnnl::memory::desc src_md = dnnl::memory::desc(dnnl::memory::dims({m_M, m_K}), m_a_type, dnnl::memory::format_tag::ab);
        dnnl::memory::desc dst_md = dnnl::memory::desc(dnnl::memory::dims({m_M, m_N}), m_a_type, dnnl::memory::format_tag::ab);
        // memory::desc wei_md = memory::desc(memory::dims({m_K, m_N}), m_w_type, memory::format_tag::any);

        // use fixed weight-layout to prevent shape-dependent weight-layout changes
        dnnl::memory::desc wei_md = dnnl::memory::desc(dnnl::memory::dims({m_K, m_N}), m_w_type, dnnl::memory::format_tag::ab);
        // dnnl::memory::desc wei_md = dnnl::memory::desc(dnnl::memory::dims({m_K, m_N}), m_w_type, dnnl::memory::format_tag::ab);

        // Create primitive descriptor.
        auto matmul_pd = dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);

        // Pre-packed weights stored as int8_t
        m_wei_md = matmul_pd.weights_desc();

        // Create the primitive.
        m_prim = dnnl::matmul(matmul_pd);
    }

    // this creator is for predefined matmul primitive types
    enum class post_ops_type {
        none,
        with_bin_mul,
        with_bin_add,
        with_bin_add_add,
        with_bin_add_mul,
        with_bin_add_silu_mul,
    };

    onednn_matmul(dnnl::engine eng,
                  dnnl::memory::data_type act_dtype,
                  dnnl::memory::data_type weight_dtype,
                  int batch,
                  int ic,
                  int oc,
                  post_ops_type t)
        : onednn_matmul(act_dtype, weight_dtype, batch, ic, oc) {
        if (t == post_ops_type::with_bin_mul) {
            post_op_bin_mul(true, true);
        }
        if (t == post_ops_type::with_bin_add) {
            post_op_bin_add(true);
        }
        if (t == post_ops_type::with_bin_add_add) {
            post_op_bin_add(true);
            post_op_bin_add(true);
        }
        if (t == post_ops_type::with_bin_add_mul) {
            post_op_bin_mul(true, true);
            post_op_bin_mul(true, true);
        }
        if (t == post_ops_type::with_bin_add_silu_mul) {
            post_op_bin_add(true);
            post_op_silu();
            post_op_bin_mul(true);
        }

        create(eng);
    }
};

// all jit-based/performance-aware function should be a functor/callable because:
//   - it needs to hold reference to kernel (to save build time & resources)
//   - it needs to do other compile time preparation work and hold the relevant
//     runtime-data-struct (to make runtime faster)
// to optimze compile-time-workload itself, the functor instance itself should be
// cached with compile-time parameter as the key.
//
// because it's a functor, which supposed to have no states, so cache-factory should
// always return shared_ptr to constant object, so it won't behave differently when being
// called by different caller, and this also ensure it's multi-threading safe since it
// won't modify it's content.
//
template <typename... TTypes>
class tuple_hasher {
private:
    typedef std::tuple<TTypes...> Tuple;
    template <int N>
    size_t hash(Tuple& value) const {
        return 0;
    }
    template <int N, typename THead, typename... TTail>
    size_t hash(Tuple& value) const {
        constexpr int Index = N - sizeof...(TTail) - 1;
        return std::hash<THead>()(std::get<Index>(value)) ^ hash<N, TTail...>(value);
    }

public:
    size_t operator()(Tuple value) const {
        auto hv = hash<sizeof...(TTypes), TTypes...>(value);
        return hv;
    }
};

// create const object with internal cache with constructor-args as the key
// this helps reduces construction time overhead, and perfectly suitable
// for caching functor/callable.
template <class T, typename... CArgs>
std::shared_ptr<const T> make_cacheable(dnnl::engine eng, CArgs... cargs) {
    std::shared_ptr<const T> sptr;
    auto key = std::make_tuple(cargs...);
    static std::unordered_map<decltype(key), std::weak_ptr<const T>, tuple_hasher<CArgs...>> cache;
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        auto& wptr = it->second;
        sptr = wptr.lock();
        if (!sptr) {
            sptr = std::make_shared<T>(eng, cargs...);
            // ECOUT("make_cacheable re-constructed: ", typeid(T).name(), "(", cargs..., ")");
            wptr = sptr;
        }
    } else {
        sptr = std::make_shared<T>(eng, cargs...);
        // ECOUT("make_cacheable constructed: ", typeid(T).name(), "(", cargs..., ")");
        cache.emplace(std::make_pair(key, std::weak_ptr<const T>(sptr)));
    }
    return sptr;
}

struct onednn_linear {
    std::shared_ptr<const onednn_matmul> mm;
    dnnl::memory weight;
    // dnnl::memory scale;
    // dnnl::memory zp;
    dnnl::matmul m_prim;
    dnnl::memory::dim m_K;
    dnnl::memory::dim m_N;
    dnnl::memory::dim m_batch;
    dnnl::memory::data_type m_a_type;
    std::vector<std::pair<int, onednn_matmul::post_ops_type>> bin_post_ops;

    static onednn_linear create(dnnl::engine eng,
                                dnnl::memory::data_type act_dtype,
                                dnnl::memory::data_type weight_dtype,
                                int batch,
                                int ic,
                                int oc,
                                onednn_matmul::post_ops_type post_ops) {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("onednn_linear::create()"));
        auto mm = make_cacheable<onednn_matmul>(eng, act_dtype, weight_dtype, batch, ic, oc, post_ops);
        onednn_linear linear;
        linear.mm = mm;
        linear.m_prim = mm->m_prim;
        linear.m_K = mm->m_K;
        linear.m_N = mm->m_N;
        linear.m_batch = batch;
        linear.m_a_type = mm->m_a_type;

        return linear;
    }

    void forward(dnnl::stream& stream, int m, dnnl::memory src_mem, dnnl::memory weight_mem, dnnl::memory dst_mem, std::vector<dnnl::memory>& bin_mems) {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("onednn_linear::forward()"));
        dnnl::memory::dim M = m;

        OPENVINO_ASSERT(m_batch == 0 || m_batch == M, "m_batch=", m_batch, " M=", M);

        dnnl::memory::desc rt_src_md = dnnl::memory::desc(dnnl::memory::dims({M, m_K}), m_a_type, dnnl::memory::format_tag::ab);
        dnnl::memory::desc rt_dst_md = dnnl::memory::desc(dnnl::memory::dims({M, m_N}), m_a_type, dnnl::memory::format_tag::ab);
        dnnl::memory::desc rt_bin_md;
        rt_bin_md = dnnl::memory::desc(dnnl::memory::dims({M, m_N}), m_a_type, dnnl::memory::format_tag::ab);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        // args.insert({DNNL_ARG_BIAS, bias_mem});
        args.insert({DNNL_ARG_DST, dst_mem});

        auto bin_mem = bin_mems.cbegin();
        auto postops_len = mm->postops.len();
        for (int bin_post_id = 0; bin_post_id < postops_len; bin_post_id++) {
            if (mm->postops.kind(bin_post_id) == dnnl::primitive::kind::binary) {
                args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_post_id) | DNNL_ARG_SRC_1, *bin_mem});
                bin_mem++;
            }
        }
        m_prim.execute(stream, args);
    }
};

dnnl::memory convert2dnnl(const memory::ptr& ptr, const std::vector<int64_t>& dim, dnnl::memory::format_tag tag, int offset) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("convert2dnnl"));
    return ptr->get_onednn_memory(dnnl::memory::desc(dnnl::memory::dims(dim), convert_data_type(ptr->get_layout().data_type), tag), offset);
}

struct lora_impl : multi_stage_primitive<lora> {
    using parent = multi_stage_primitive<lora>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lora_kernel_selector;
    using kernel_params_t = kernel_selector::lora_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::lora_impl);

    const uint32_t optimized_kernel = 0;
    const uint32_t reference_kernel = 1;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<lora_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            for (auto& kd : _kernels_data) {
                if (kd.kernelName.length() != 0) {
                    auto kernel_impl = kernel_selector.GetImplementation(kd.kernelName);
                    kernel_impl->GetUpdateDispatchDataFunc(kd);
                }
            }
        }
    }

    kernel_arguments_data get_arguments(const lora_inst& instance, size_t stage) const override {
        kernel_arguments_data args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_fused_primitives()) {
            size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; i++) {
                args.fused_op_inputs.push_back(instance.fused_memory(i));
            }
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        args.shape_info = instance.shape_info_memory_ptr();

        return args;
    }

    bool is_optimized_kernel_supported(const lora_inst& instance) {
        const auto& in_dtype = instance.get_input_layout().data_type;
        size_t subgroup_size = in_dtype == ov::element::f16 ? 16 : 8;

        const auto& state_a_layout = instance.get_input_layout(2);
        size_t input_state = state_a_layout.get_shape().back();
        if (input_state % subgroup_size != 0) {
            return false;
        }

        const auto& alpha_layout = instance.get_input_layout(3);
        size_t lora_rank = alpha_layout.get_shape().back();
        if (lora_rank % subgroup_size != 0) {
            return false;
        }

        const auto& state_b_layout = instance.get_input_layout(4);
        size_t output_state = state_b_layout.get_shape().front();
        if (output_state % subgroup_size != 0) {
            return false;
        }

        return true;
    }

// #ifdef ENABLE_ONEDNN_FOR_GPU
    struct onednn_kernel {
        onednn_linear gemm_a;
        onednn_linear gemm_b;
    };
    struct PairHash {
        template <class T1, class T2>
        size_t operator()(const std::pair<T1, T2>& p) const {
            // Combine hash values of the pair elements
            return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
        }
    };
    std::unordered_map<std::pair<int, int>, onednn_kernel, PairHash> onednn_kernels;

    onednn_kernel& get_kernel(dnnl::stream& dnnl_stream, lora_inst& instance, int n_token, int lora_rank, int in_state_size, int out_state_size, int expert_no=0) {
        auto key = std::make_pair(n_token, expert_no);
        if (onednn_kernels.count(key))
            return onednn_kernels[key];

        auto activation_dt = convert_data_type(instance.input_memory_ptr(1)->get_layout().data_type);
        auto weights_dt = convert_data_type(instance.input_memory_ptr(2)->get_layout().data_type);

        // as a temprory approach to alliviate performance loss, only support limited situations observed in HP lora cases.
        auto get_fused_post_ops = [&](lora_inst& instance) {
            const auto& node = instance.get_node();
            const auto& fused_ops = node.get_fused_primitives();
            const auto fusedops_len = fused_ops.size();

            if (fusedops_len == 0) return onednn_matmul::post_ops_type::with_bin_add;
            if (fusedops_len == 1) {
                const auto& fused_op = fused_ops.front();
                if (fused_op.is_type<eltwise>() && fused_op.typed_desc<eltwise>()->mode == cldnn::eltwise_mode::sum)
                    return onednn_matmul::post_ops_type::with_bin_add_add;
                if (fused_op.is_type<eltwise>() && fused_op.typed_desc<eltwise>()->mode == cldnn::eltwise_mode::prod)
                    return onednn_matmul::post_ops_type::with_bin_add_mul;
            }
            if (fusedops_len == 2) {
                const auto& fused_op0 = fused_ops.front();
                const auto& fused_op1 = fused_ops.back();
                if ((fused_op0.is_type<activation>() && fused_op0.typed_desc<activation>()->activation_function == cldnn::activation_func::swish) &&
                    (fused_op1.is_type<eltwise>() && fused_op1.typed_desc<eltwise>()->mode == cldnn::eltwise_mode::prod))
                    return onednn_matmul::post_ops_type::with_bin_add_silu_mul;
            }

            OPENVINO_ASSERT(0, "[GPU] Unsupported fused post ops pattern... should not be here.");
        };

        onednn_kernel kernel;
        // down
        {
            kernel.gemm_a = onednn_linear::create(dnnl_stream.get_engine(),
                                                activation_dt,
                                                weights_dt,
                                                n_token,
                                                in_state_size,
                                                lora_rank,
                                                onednn_matmul::post_ops_type::with_bin_mul);
        }

        // up
        {
            auto fused_ops = get_fused_post_ops(instance);
            kernel.gemm_b = onednn_linear::create(dnnl_stream.get_engine(),
                                            activation_dt,
                                            weights_dt,
                                            n_token,
                                            lora_rank,
                                            out_state_size,
                                            fused_ops);
        }
        onednn_kernels[key] = kernel;
        return onednn_kernels[key];
    }

    event::ptr execute_stage(const std::vector<event::ptr>& events, lora_inst& instance) {
        std::cout << "============================= execute onednn_lora ==============================" << std::endl;
        auto& cur_net = instance.get_network();
        auto& stream = cur_net.get_stream();
        auto& dnnl_stream = stream.get_onednn_stream();
        cldnn::event::ptr result_event;

        const auto& lora_input_layout = instance.get_input_layout(1); 
        const auto& lora_input_shape = lora_input_layout.get_shape();   // (1, M, in_state_size)
        size_t n_token = lora_input_shape[1];

        const auto& lora_a_mem = instance.input_memory_ptr(2);
        const auto& lora_alpha_mem = instance.input_memory_ptr(3);      // (1, lora_rank)
        const auto& lora_b_mem = instance.input_memory_ptr(4);

        const auto& lora_a_shape = lora_a_mem->get_layout().get_shape();  // (lora_rank, in_state_size)
        const auto& lora_b_shape = lora_b_mem->get_layout().get_shape();  // (out_state_size, lora_rank)
        auto lora_rank = lora_a_shape[0];
        auto in_state_size = lora_a_shape[1];
        auto out_state_size = lora_b_shape[0];

        onednn_kernel& kernel = get_kernel(dnnl_stream, instance, n_token, lora_rank, in_state_size, out_state_size);

        const auto main_input = instance.input_memory_ptr(0);
        const auto lora_input = instance.input_memory_ptr(1);
        const auto& lora_output = instance.output_memory_ptr();
        const auto scracth_mem = instance.get_intermediates_memories().front();

        // src, weight, dst, bin
        std::vector<dnnl::memory> bin_mems_a = {convert2dnnl(lora_alpha_mem, {static_cast<int>(1), static_cast<int>(lora_rank)}, dnnl::memory::format_tag::ab)};
        kernel.gemm_a.forward(dnnl_stream, n_token,
                            convert2dnnl(lora_input, {static_cast<int>(n_token), static_cast<int>(in_state_size)}, dnnl::memory::format_tag::ab),
                            convert2dnnl(lora_a_mem, {static_cast<int>(in_state_size), static_cast<int>(lora_rank)}, dnnl::memory::format_tag::ab),
                            convert2dnnl(scracth_mem, {static_cast<int>(n_token), static_cast<int>(lora_rank)}, dnnl::memory::format_tag::ab),
                            bin_mems_a);

        std::vector<dnnl::memory> bin_mems_b = {convert2dnnl(main_input, {static_cast<int>(n_token), static_cast<int>(out_state_size)}, dnnl::memory::format_tag::ab)};
        if (instance.get_fused_mem_count() > 0) {
            OPENVINO_ASSERT(instance.get_fused_mem_count()==1, "Unsupported fuse pattern for lora.");
            auto bin_mem = instance.fused_memory(0);
            bin_mems_b.push_back(convert2dnnl(bin_mem, {static_cast<int>(n_token), static_cast<int>(out_state_size)}, dnnl::memory::format_tag::ab));
        }
        kernel.gemm_b.forward(dnnl_stream, n_token,
                            convert2dnnl(scracth_mem, {static_cast<int>(n_token), static_cast<int>(lora_rank)}, dnnl::memory::format_tag::ab),
                            convert2dnnl(lora_b_mem, {static_cast<int>(lora_rank), static_cast<int>(out_state_size)}, dnnl::memory::format_tag::ab),
                            convert2dnnl(lora_output, {static_cast<int>(n_token), static_cast<int>(out_state_size)}, dnnl::memory::format_tag::ab),
                            bin_mems_b);
        if (instance.needs_completion_event())
            result_event = stream.enqueue_marker({});

        return result_event;
    }
// #endif

    event::ptr execute_impl(const std::vector<event::ptr>& events, lora_inst& instance) override {
// #ifdef ENABLE_ONEDNN_FOR_GPU
        if (instance.is_onednn_lora_prefered()) {
            return execute_stage(events, instance);
        }
// #endif
        if (is_optimized_kernel_supported(instance)) {
            return execute_stage(events, instance, optimized_kernel);
        } else {
            return execute_stage(events, instance, reference_kernel);
        }
    }

    void set_arguments_impl(lora_inst& instance) override {}

    event::ptr execute_stage(const std::vector<event::ptr>& events, lora_inst& instance, size_t stage) {
        stream& stream = instance.get_network().get_stream();
        std::vector<event::ptr> tmp_events(events);
        std::vector<event::ptr> all_events;
        size_t kernel_offset = 0;
        bool skip_full_lora = true;

        for (size_t s = 0; s < stage; s++) {
            kernel_offset += _kernels_data[s].kernels.size();
        }
        for (size_t kd_idx = 0; kd_idx < _kernels_data[stage].kernels.size(); ++kd_idx) {
            if (_kernels_data[stage].kernels[kd_idx].skip_execution) {
                continue;
            } else {
                skip_full_lora = false;
            }

            size_t idx_final = kernel_offset + kd_idx;
            // If any user of the desc's users is CPU implementation or network's output, set desc as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernels_data[stage].kernels[kd_idx].params;
            auto args = get_arguments(instance, stage);
            args.scalars = &params.scalars;

            if (stage == optimized_kernel) {
                for (const auto& m : instance.get_intermediates_memories()) {
                    args.intermediates.push_back(m);
                }
            }

            stream.set_arguments(*_kernels[idx_final], _kernels_data[stage].kernels[kd_idx].params, args);

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage << " kernel " << idx_final << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

            auto ev = stream.enqueue_kernel(*_kernels[idx_final], params, args, tmp_events, needs_completion_event);
            if (_kernels_data[stage].needs_sub_kernels_sync) {
                tmp_events = {ev};
            }
            all_events.push_back(ev);
        }

        if (skip_full_lora) {
            for (auto& ev : events) {
                all_events.push_back(ev);
            }
        }

        return stream.aggregate_events(all_events, all_events.size() > 1);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false, bool is_ref_kernel = false) {
        auto params = get_default_params<kernel_selector::lora_params>(impl_param, is_shape_agnostic);

        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        size_t fused_dep_size = 0;
        for (const auto& fused_op : params.fused_ops) {
            fused_dep_size += fused_op.dep_size;
        }
        params.lora_count = (params.inputs.size() - fused_dep_size - 2ul) / 3ul;
        params.is_ref_kernel = is_ref_kernel;
        params.set_dynamic_shape_offsets();

        return params;
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<lora>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        auto& kernel_selector = kernel_selector_t::Instance();
        auto canonicalized_params = static_canonicalize_shapes(impl_param);

        auto optimized_kernel_params = get_kernel_params(canonicalized_params, impl_param.is_dynamic());
        kernels_data.push_back(kernel_selector.get_best_kernel(optimized_kernel_params));

        auto reference_kernel_params = get_kernel_params(canonicalized_params, impl_param.is_dynamic(), true);
        kernels_data.push_back(kernel_selector.get_best_kernel(reference_kernel_params));

        return std::make_unique<lora_impl>(kernels_data);
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = impl_params;
        for (size_t i = 0; i < 2; ++i) {
            if (impl_params.get_input_layout(i).get_partial_shape().size() == 2) {
                auto input_pshape = impl_params.input_layouts[i].get_partial_shape();
                input_pshape.insert(input_pshape.begin(), 1);
                updated_impl_params.input_layouts[i].set_partial_shape(input_pshape);
            }
        }
        return primitive_impl::static_canonicalize_shapes(updated_impl_params);
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        for (size_t kd_idx = 0; kd_idx < _kernels_data.size(); ++kd_idx) {
            auto& kd = _kernels_data[kd_idx];
            // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
            if (kd.params == nullptr) {
                bool is_ref_kernel = static_cast<bool>(kd_idx);
                kd.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true, is_ref_kernel));
            }

            update_shapes(*kd.params, impl_param);
            (kd.update_dispatch_data_func)(*kd.params, kd);
        }
    }
};

std::unique_ptr<primitive_impl> LoraImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<lora>());
    return cldnn::ocl::lora_impl::create(static_cast<const lora_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::lora_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::lora)
