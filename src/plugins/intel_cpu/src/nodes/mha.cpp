// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "ie_parallel.hpp"
#include "mha.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"
#include <utils/general_utils.h>
#include <cpu/x64/jit_generator.hpp>
#include "emitters/jit_dnnl_emitters.hpp"
#include "emitters/jit_load_store_emitters.hpp"
#include "common/cpu_convert.h"
#include "ngraph_transformations/op/mha.hpp"
#include "dnnl_extension_utils.h"

using namespace InferenceEngine;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::cpu::x64::matmul;
using namespace Xbyak;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

namespace ov {
namespace intel_cpu {
namespace node {

template <cpu_isa_t isa>
struct jit_mul_add_softmax_kernel : public jit_uni_mul_add_softmax_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_mul_add_softmax_kernel)

    explicit jit_mul_add_softmax_kernel(const jit_mul_add_softmax_compile_params& jcp) : jit_uni_mul_add_softmax_kernel(jcp), jit_generator() {
        exp_emitter = std::make_shared<jit_dnnl_aux_emitter>(this, isa, dnnl_eltwise_exp, 0.f, 0.f);
        load_emitter.reset(new jit_load_emitter(this, isa));
        store_emitter.reset(new jit_store_emitter(this, isa));

        vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / sizeof(float);
    }
    virtual ~jit_mul_add_softmax_kernel() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41, Xmm, isa == cpu_isa_t::avx2, Ymm, Zmm>::type;

    void generate() override {
        this->preamble();

#define GET_OFF(field) offsetof(jit_mul_add_softmax_call_args, field)
        mov(reg_mul_in0, ptr[reg_params + GET_OFF(p_mul_in0)]);
        mov(reg_mul_in1, ptr[reg_params + GET_OFF(p_mul_in1)]);
        mov(reg_add_in1, ptr[reg_params + GET_OFF(p_add_in1)]);
        mov(reg_out, ptr[reg_params + GET_OFF(p_out)]);
        mov(reg_buffer, ptr[reg_params + GET_OFF(p_buffer)]);

        Xbyak::Label mul_add_max_loop_label;
        Xbyak::Label mul_add_max_end_label;
        Xbyak::Label sub_exp_reduce_loop_label;
        Xbyak::Label sub_exp_reduce_end_label;
        Xbyak::Label mul_loop_label;
        Xbyak::Label mul_end_label;

        size_t tail_size = jcp_.work_amount % vec_size;

        mov(reg_buffer_aux, reg_buffer);
        mov(reg_work_amount, jcp_.work_amount);
        mov(reg_work_amount_aux, reg_work_amount);
        uni_vpxor(get_vmm_max(0), get_vmm_max(0), get_vmm_max(0));

        // mul1 input is const and always float
        uni_vmovss(Xmm(get_vmm_in(1).getIdx()), ptr[reg_mul_in1]);
        uni_vbroadcastss(get_vmm_in(1), Xmm(get_vmm_in(1).getIdx()));

        if (jcp_.with_scales0) {
            mov(reg_scales, ptr[reg_params + GET_OFF(p_scales0)]);

            mov(reg_tmp, dnnl::impl::float2int(-128.0f));
            vmovq(xmm_tmp, reg_tmp);
            vbroadcastss(vmm_crop_low, xmm_tmp);

            mov(reg_tmp, dnnl::impl::float2int(127.0f));
            vmovq(xmm_tmp, reg_tmp);
            vbroadcastss(vmm_crop_high, xmm_tmp);
        }

        if (jcp_.with_scales0 && jcp_.broadcast_scales0) {
            uni_vmovss(Xmm(vmm_scales.getIdx()), ptr[reg_scales]);
            uni_vbroadcastss(vmm_scales, Xmm(vmm_scales.getIdx()));
        }

        L(mul_add_max_loop_label);
        {
            cmp(reg_work_amount_aux, vec_size);
            jl(mul_add_max_end_label, T_NEAR);

            mul_add_max(vec_size);

            sub(reg_work_amount_aux, vec_size);

            jmp(mul_add_max_loop_label, T_NEAR);
        }
        L(mul_add_max_end_label);
        if (tail_size) {
            mul_add_max(tail_size);
        }

        sub(rsp, sizeof(float) * vec_size);
        uni_vmovups(ptr[rsp], get_vmm_max(0));
        uni_vpxor(get_vmm_max(0), get_vmm_max(0), get_vmm_max(0));
        for (size_t i = 0; i < vec_size; i++) {
            mov(reg_tmp_32, ptr[rsp + i * sizeof(float)]);
            vmovq(xmm_tmp, reg_tmp);
            uni_vmaxps(get_xmm_max(0), get_xmm_max(0), xmm_tmp);
        }
        uni_vbroadcastss(get_vmm_max(0), get_xmm_max(0));
        add(rsp, sizeof(float) * vec_size);

        uni_vpxor(get_vmm_denom(0), get_vmm_denom(0), get_vmm_denom(0));
        mov(reg_work_amount_aux, reg_work_amount);
        mov(reg_buffer_aux, reg_buffer);
        L(sub_exp_reduce_loop_label);
        {
            cmp(reg_work_amount_aux, vec_size);
            jl(sub_exp_reduce_end_label, T_NEAR);

            sub_exp_reduce(vec_size);

            sub(reg_work_amount_aux, vec_size);

            jmp(sub_exp_reduce_loop_label, T_NEAR);
        }
        L(sub_exp_reduce_end_label);
        if (tail_size) {
            sub_exp_reduce(tail_size);
        }

        sub(rsp, sizeof(float) * vec_size);
        uni_vmovups(ptr[rsp], get_vmm_denom(0));
        uni_vpxor(get_vmm_aux(0), get_vmm_aux(0), get_vmm_aux(0));
        for (size_t i = 0; i < vec_size; i++) {
            mov(reg_tmp_32, ptr[rsp + i * sizeof(float)]);
            vmovq(xmm_tmp, reg_tmp);
            uni_vaddps(get_xmm_aux(0), get_xmm_aux(0), xmm_tmp);
        }
        vbroadcastss(get_vmm_aux(0), get_xmm_aux(0));
        add(rsp, sizeof(float) * vec_size);

        mov(reg_tmp, dnnl::impl::float2int(1.0f));
        vmovq(xmm_tmp, reg_tmp);
        vbroadcastss(get_vmm_denom(0), xmm_tmp);
        uni_vdivps(get_vmm_denom(0), get_vmm_denom(0), get_vmm_aux(0));

        if (jcp_.with_scales1)
            mov(reg_scales, ptr[reg_params + GET_OFF(p_scales1)]);

        if (jcp_.with_scales1 && jcp_.broadcast_scales1) {
            uni_vmovss(Xmm(vmm_scales.getIdx()), ptr[reg_scales]);
            uni_vbroadcastss(vmm_scales, Xmm(vmm_scales.getIdx()));
        }

        mov(reg_work_amount_aux, reg_work_amount);
        L(mul_loop_label);
        {
            cmp(reg_work_amount_aux, vec_size);
            jl(mul_end_label, T_NEAR);

            mul_loop(vec_size);

            sub(reg_work_amount_aux, vec_size);

            jmp(mul_loop_label, T_NEAR);
        }
        L(mul_end_label);
        if (tail_size) {
            mul_loop(tail_size);
        }

        this->postamble();

        load_emitter->emit_data();
        store_emitter->emit_data();
        exp_emitter->emit_data();
    }

    void mul_add_max(size_t step) {
        bool is_tail = step < vec_size;

        load(get_vmm_in(0), reg_mul_in0, jcp_.src_prc, step, is_tail);
        load(get_vmm_in(2), reg_add_in1, Precision::FP32, step, is_tail);

        if (jcp_.src_prc == Precision::I32) {
            if (jcp_.with_scales0) {
                if (!jcp_.broadcast_scales0) {
                    load(vmm_scales, reg_scales, Precision::FP32, step, is_tail);
                    add(reg_scales,  sizeof(float) * step);
                }
                uni_vmulps(get_vmm_in(0), get_vmm_in(0), vmm_scales);
                uni_vmaxps(get_vmm_in(0), get_vmm_in(0), vmm_crop_low);
                uni_vminps(get_vmm_in(0), get_vmm_in(0), vmm_crop_high);
            }

            uni_vaddps(get_vmm_aux(0), get_vmm_in(0), get_vmm_in(2));
            uni_vmulps(get_vmm_aux(0), get_vmm_aux(0), get_vmm_in(1));
        } else {
            uni_vmulps(get_vmm_aux(0), get_vmm_in(0), get_vmm_in(1));
            uni_vaddps(get_vmm_aux(0), get_vmm_aux(0), get_vmm_in(2));
        }

        uni_vmaxps(get_vmm_max(0), get_vmm_max(0), get_vmm_aux(0));

        store(reg_buffer_aux, get_vmm_aux(0), Precision::FP32, step);

        if (!is_tail) {
            add(reg_mul_in0, jcp_.src_prc.size() * step);
            add(reg_add_in1, sizeof(float) * step);
            add(reg_buffer_aux, sizeof(float) * step);
        }
    }

    void sub_exp_reduce(size_t step) {
        bool is_tail = step < vec_size;

        load(get_vmm_in(0), reg_buffer_aux, Precision::FP32, step, is_tail);

        uni_vsubps(get_vmm_in(0), get_vmm_in(0), get_vmm_max(0));

        auto vmm_exp_idx = static_cast<size_t>(get_vmm_in(0).getIdx());
        exp_emitter->emit_code({vmm_exp_idx}, {vmm_exp_idx}, {}, {});

        uni_vaddps(get_vmm_denom(0), get_vmm_denom(0), get_vmm_in(0));

        store(reg_buffer_aux, get_vmm_in(0), Precision::FP32, step);

        if (!is_tail) {
            add(reg_buffer_aux, sizeof(float) * step);
        }
    }

    void mul_loop(size_t step) {
        bool is_tail = step < vec_size;

        load(get_vmm_in(0), reg_buffer, Precision::FP32, step, is_tail);

        uni_vmulps(get_vmm_in(0), get_vmm_in(0), get_vmm_denom(0));

        if (jcp_.src_prc == Precision::I32) {
            if (jcp_.with_scales1) {
                if (!jcp_.broadcast_scales1) {
                    load(vmm_scales, reg_scales, Precision::FP32, step, is_tail);
                    add(reg_scales,  sizeof(float) * step);
                }
                uni_vmulps(get_vmm_in(0), get_vmm_in(0), vmm_scales);
            }
        }

        store(reg_out, get_vmm_in(0), jcp_.dst_prc, step);

        if (!is_tail) {
            add(reg_buffer, sizeof(float) * step);
            add(reg_out, jcp_.dst_prc.size() * step);
        }
#undef GET_OFF
    }

    inline void load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, Precision src_prc, const int& elt_num, bool fill) {
        load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())},
            std::make_shared<load_emitter_context>(src_prc, Precision::FP32, elt_num, 0, fill, "float_min"),
            pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }
    inline void store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, Precision dst_prc, const int& elt_num) {
        store_emitter->emit_code({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
            std::make_shared<store_emitter_context>(Precision::FP32, dst_prc, elt_num, 0),
            pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }

    size_t unroll_factor = 3;
    size_t vec_size;

    Vmm get_vmm_in(int idx) {
        return Vmm(1 + 0 * unroll_factor + idx);
    }

    Vmm get_vmm_aux(int idx) {
        return Vmm(1 + 1 * unroll_factor + idx);
    }
    Xmm get_xmm_aux(int idx) {
        return Xmm(1 + 1 * unroll_factor + idx);
    }

    Vmm get_vmm_max(int idx) {
        return Vmm(1 + 2 * unroll_factor + idx);
    }
    Xmm get_xmm_max(int idx) {
        return Xmm(1 + 2 * unroll_factor + idx);
    }


    Vmm get_vmm_denom(int idx) {
        return Vmm(1 + 3 * unroll_factor + idx);
    }

    Xmm xmm_tmp = Xmm(0);
    Xmm vmm_ones = Vmm(0);

    Vmm vmm_scales = Vmm(0);
    Vmm vmm_crop_low = Vmm(14);
    Vmm vmm_crop_high = Vmm(15);

    Reg64 reg_mul_in0 = r8;
    Reg64 reg_mul_in1 = r9;
    Reg64 reg_add_in1 = r10;
    Reg64 reg_out = r11;
    Reg64 reg_scales = r12;
    Reg64 reg_work_amount = r13;
    Reg64 reg_work_amount_aux = r14;
    Reg64 reg_buffer = r15;
    Reg64 reg_buffer_aux = rax;
    Reg64 reg_tmp = rbx;
    Reg32 reg_tmp_32 = Reg32(rbx.getIdx());
    Reg64 reg_max = rdx;
    Reg32 reg_max_32 = Reg32(rdx.getIdx());
    Reg64 reg_params = abi_param1;

    const std::vector<size_t> pool_aux_gpr_idxs = { static_cast<size_t>(rsi.getIdx()), static_cast<size_t>(rbp.getIdx()) };
    const std::vector<size_t> pool_aux_vmm_idxs = { static_cast<size_t>(xmm_tmp.getIdx()) };

    std::shared_ptr<jit_dnnl_aux_emitter> exp_emitter;

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};


template <cpu_isa_t isa>
struct jit_convert_reorder_kernel : public jit_uni_convert_reorder_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_convert_reorder_kernel)

    explicit jit_convert_reorder_kernel(const jit_convert_reorder_compile_params& jcp) : jit_uni_convert_reorder_kernel(jcp), jit_generator() {
        load_emitter.reset(new jit_load_emitter(this, isa));
        store_emitter.reset(new jit_store_emitter(this, isa));

        vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / sizeof(float);
    }
    virtual ~jit_convert_reorder_kernel() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41, Xmm, isa == cpu_isa_t::avx2, Ymm, Zmm>::type;

    void generate() override {
        this->preamble();

#define GET_OFF(field) offsetof(jit_convert_reorder_call_args, field)
        mov(reg_in, ptr[reg_params + GET_OFF(p_in)]);
        mov(reg_out, ptr[reg_params + GET_OFF(p_out)]);
        mov(reg_outter_work_amount, ptr[reg_params + GET_OFF(outter_work_amount)]);

        if (jcp_.with_scales) {
            mov(reg_scales, ptr[reg_params + GET_OFF(p_scales)]);
        }

        Xbyak::Label convert_reorder_inner_loop_label;
        Xbyak::Label convert_reorder_inner_end_label;
        Xbyak::Label convert_reorder_outter_loop_label;
        Xbyak::Label convert_reorder_outter_end_label;

        if (jcp_.with_scales && jcp_.broadcast_scales) {
            uni_vmovss(Xmm(vmm_scales.getIdx()), ptr[reg_scales]);
            uni_vbroadcastss(vmm_scales, Xmm(vmm_scales.getIdx()));
        }

        L(convert_reorder_outter_loop_label);
        {
            cmp(reg_outter_work_amount, 1);
            jl(convert_reorder_outter_end_label, T_NEAR);

            size_t tail_size = jcp_.inner_work_amount % vec_size;
            mov(reg_inner_work_amount, jcp_.inner_work_amount);
            mov(reg_in_aux, reg_in);
            mov(reg_out_aux, reg_out);
            if (jcp_.with_scales && !jcp_.broadcast_scales) {
                mov(reg_scales, ptr[reg_params + GET_OFF(p_scales)]);
            }

            L(convert_reorder_inner_loop_label);
            {
                cmp(reg_inner_work_amount, vec_size);
                jl(convert_reorder_inner_end_label, T_NEAR);

                convert_reorder(vec_size);

                sub(reg_inner_work_amount, vec_size);

                jmp(convert_reorder_inner_loop_label, T_NEAR);
            }
            L(convert_reorder_inner_end_label);
            if (tail_size) {
                convert_reorder(tail_size);
            }

            dec(reg_outter_work_amount);
            add(reg_in, jcp_.src_prc.size() * jcp_.src_stride);
            add(reg_out, jcp_.dst_prc.size() * jcp_.dst_stride);

            jmp(convert_reorder_outter_loop_label, T_NEAR);
        }
        L(convert_reorder_outter_end_label);

        this->postamble();

        load_emitter->emit_data();
        store_emitter->emit_data();
    }

    void convert_reorder(size_t step) {
        bool is_tail = step < vec_size;

        load(vmm_in, reg_in_aux, jcp_.src_prc, step, is_tail);

        if (jcp_.src_prc == Precision::I32) {
            if (jcp_.with_scales) {
                if (!jcp_.broadcast_scales) {
                    load(vmm_scales, reg_scales, Precision::FP32, step, is_tail);
                    add(reg_scales,  sizeof(float) * step);
                }
                uni_vmulps(vmm_in, vmm_in, vmm_scales);
            }
        }

        store(reg_out_aux, vmm_in, jcp_.dst_prc, step);

        if (!is_tail) {
            add(reg_in_aux, jcp_.src_prc.size() * step);
            add(reg_out_aux, jcp_.dst_prc.size() * step);
        }
    }
#undef GET_OFF

    inline void load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, Precision src_prc, const int& elt_num, bool fill) {
        load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())},
            std::make_shared<load_emitter_context>(src_prc, Precision::FP32, elt_num, 0, fill, "float_min"),
            pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }
    inline void store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, Precision dst_prc, const int& elt_num) {
        store_emitter->emit_code({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
            std::make_shared<store_emitter_context>(Precision::FP32, dst_prc, elt_num, 0),
            pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }

    size_t vec_size;

    Xmm xmm_tmp = Xmm(0);
    Vmm vmm_scales = Vmm(0);
    Vmm vmm_in = Vmm(1);

    Reg64 reg_in = r8;
    Reg64 reg_in_aux = r9;
    Reg64 reg_out = r10;
    Reg64 reg_out_aux = r11;
    Reg64 reg_scales = r12;
    Reg64 reg_inner_work_amount = r14;
    Reg64 reg_outter_work_amount = r15;
    Reg64 reg_params = abi_param1;

    const std::vector<size_t> pool_aux_gpr_idxs = { static_cast<size_t>(rsi.getIdx()), static_cast<size_t>(rbp.getIdx()) };
    const std::vector<size_t> pool_aux_vmm_idxs = { static_cast<size_t>(xmm_tmp.getIdx()) };

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};


bool MHA::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto mha = std::dynamic_pointer_cast<const MHANode>(op);
        if (!mha) {
            errorMessage = "Only MHA from CPU internal opset is supported";
            return false;
        }

        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op dynamic shapes";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MHA::MHA(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng,
        WeightsSharing::Ptr &cache) : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto mha = std::dynamic_pointer_cast<const MHANode>(op);
    if (mha->with_fq_scales()) {
        fqScales0 = mha->get_fq_scales(0);
        fqScales1 = mha->get_fq_scales(1);
        fqScales2 = mha->get_fq_scales(2);
    }
}

void MHA::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    dataPrecision = getOriginalInputPrecisionAtPort(0);

    if (!one_of(dataPrecision, Precision::FP32, Precision::BF16, Precision::I8))
        THROW_ERROR << "doesn't support " << dataPrecision.name() << " precision";

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, Precision::FP32},
                          {LayoutType::ncsp, Precision::FP32},
                          {LayoutType::ncsp, dataPrecision}},
                         {{LayoutType::ncsp, dataPrecision}},
                         ref_any,
                         isDynamicNode());
}

void MHA::init_brgemm(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& brgKernel, bool use_amx) {
    brgemm_t brgDesc;
    brgemm_strides_t strides {ctx.M * ctx.K, ctx.K * ctx.N};

    auto isa = use_amx ? isa_any
                       : ctx.dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : avx512_core_vnni;
    auto status = brgemm_desc_init(&brgDesc, isa, brgemm_strd, ctx.dt_in0, ctx.dt_in1,
            false, false, brgemm_row_major, 1.f, ctx.beta, ctx.LDA, ctx.LDB, ctx.LDC, ctx.M, ctx.N, ctx.K, &strides);
    if (status != dnnl_success) {
        THROW_ERROR << "cannot be executed due to invalid brgconv params";
    }

    ctx.is_with_amx = use_amx;
    status = brgemm_init_tiles(brgDesc, ctx.palette);
    if (use_amx) {
        amx_tile_configure(ctx.palette);
    }

    ctx.is_with_comp = dataPrecision == Precision::I8 && !ctx.is_with_amx;

    brgemm_kernel_t* brgKernel_ = nullptr;
    status = brgemm_kernel_create(&brgKernel_, brgDesc);
    if (status != dnnl_success) {
        THROW_ERROR << "cannot be executed due to invalid brgconv params";
    }
    brgKernel.reset(brgKernel_);
}

void MHA::init_brgemm_copy_a(std::unique_ptr<jit_brgemm_matmul_copy_a_t>& brgCopyKernel, size_t K, size_t K_blk, size_t K_tail,
        size_t LDA, dnnl_data_type_t dt_in0) {
    brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_tag = dnnl_abcd;
    brgCopyKernelConf.K = K;
    brgCopyKernelConf.K_tail = K_tail;
    brgCopyKernelConf.K_blk = K_blk;
    brgCopyKernelConf.use_buffer_a_tail_only = false;
    brgCopyKernelConf.LDA = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.s8s8_compensation_required = false;
    brgCopyKernelConf.wei_zp_type = dnnl::impl::cpu::x64::none;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.a_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(dt_in0));
    brgCopyKernelConf.transposed_A = false;

    create_brgemm_matmul_copy_a(brgCopyKernel, &brgCopyKernelConf);
}

void MHA::init_brgemm_copy_b(std::unique_ptr<jit_brgemm_matmul_copy_b_t>& brgCopyKernel, size_t N, size_t N_blk, size_t N_tail, size_t LDB, size_t K,
        bool is_with_amx, dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1) {
    brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.wei_dt = dt_in1;
    brgCopyKernelConf.wei_n_blk = N_blk;
    brgCopyKernelConf.wei_tag = dnnl_abcd;
    brgCopyKernelConf.copy_B_wei_stride = 0;
    brgCopyKernelConf.LDB = LDB;
    brgCopyKernelConf.N = N;
    brgCopyKernelConf.N_tail = N_tail;
    brgCopyKernelConf.N_blk = N_blk;
    brgCopyKernelConf.K = K;
    brgCopyKernelConf.K_blk = K;
    brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;

    if (is_with_amx) {
        brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16_amx_bf16 : avx512_core_bf16_amx_int8;
        brgCopyKernelConf.s8s8_compensation_required = false;
    } else {
        brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : avx512_core_vnni;
        brgCopyKernelConf.s8s8_compensation_required = dt_in0 == dnnl_data_type_t::dnnl_s8;
    }

    brgCopyKernelConf.has_zero_point_a = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;

    create_brgemm_matmul_copy_b(brgCopyKernel, &brgCopyKernelConf);
}

void MHA::prepareParams() {
    auto transpose = [](const std::vector<size_t>& vec, const std::vector<size_t>& order) -> std::vector<size_t> {
        std::vector<size_t> new_vec(vec.size());
        for (int i = 0; i < vec.size(); i++) {
            new_vec[i] = vec[order[i]];
        }
        return new_vec;
    };

    const auto memDescTranspose0In0 = getParentEdgeAt(0)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>();
    const auto memDescTranspose1In0 = getParentEdgeAt(1)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>();
    const auto memDescMulIn1 = getParentEdgeAt(2)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>();
    const auto memDescAddIn1 = getParentEdgeAt(3)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>();
    const auto memDescTranspose2In0 = getParentEdgeAt(4)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>();
    const auto memDescOut = getChildEdgeAt(0)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>();

    dimsTranspose0In0 = memDescTranspose0In0->getBlockDims();
    dimsTranspose1In0 = memDescTranspose1In0->getBlockDims();
    dimsMulIn1 = memDescMulIn1->getBlockDims();
    dimsAddIn1 = memDescAddIn1->getBlockDims();
    dimsTranspose2In0 = memDescTranspose2In0->getBlockDims();
    dimsOut = memDescOut->getBlockDims();

    strTranspose0In0 = memDescTranspose0In0->getStrides();
    strTranspose1In0 = memDescTranspose1In0->getStrides();
    strMulIn1 = memDescMulIn1->getStrides();
    strAddIn1 = memDescAddIn1->getStrides();
    strTranspose2In0 = memDescTranspose2In0->getStrides();
    strOut = memDescOut->getStrides();

    std::vector<size_t> orderTranspose0 = {0, 2, 1, 3};
    dimsMatMul0In0 = transpose(dimsTranspose0In0, orderTranspose0);

    std::vector<size_t> orderTranspose1 = {0, 2, 3, 1};
    dimsMatMul0In1 = transpose(dimsTranspose1In0, orderTranspose1);

    dimsMatMul0Out = {dimsMatMul0In0[0], dimsMatMul0In0[1], dimsMatMul0In0[2], dimsMatMul0In1[3]};

    std::vector<size_t> orderTranspose2 = {0, 2, 1, 3};
    dimsMatMul1In1 = transpose(dimsTranspose2In0, orderTranspose2);

    bool isAMXSupported = mayiuse(avx512_core_bf16_amx_int8) || mayiuse(avx512_core_bf16_amx_bf16);

    size_t numThreads = parallel_get_max_threads();

    size_t matmulOptimalM = 32;
    size_t matmulOptimalN = dataPrecision == Precision::I8 ? 64 : 32;
    size_t matmulOptimalK = dataPrecision == Precision::I8 ? 64 : 32;

    batch0 = dimsMatMul0Out[0];
    batch1 = dimsMatMul0Out[1];

    M = dimsMatMul0In0[2];
    M_blk = matmulOptimalM;
    M_tail = M % M_blk;

    N0 = dimsMatMul0In1[3];
    K0 = dimsMatMul0In0[3];

    brg0VnniFactor = 4 / dataPrecision.size();
    bool brg0WithAMX = isAMXSupported && dataPrecision != Precision::FP32 && (K0 % brg0VnniFactor == 0) && (N0 % brg0VnniFactor == 0);

    N0_blk = dataPrecision == Precision::FP32 ? N0 : matmulOptimalN;
    N0_tail = N0 % N0_blk;
    K0_blk = brg0WithAMX ? matmulOptimalK : K0;
    K0_tail = K0 % K0_blk;

    size_t brg0BaseIdx = -1;
    for (size_t m = 0; m < 2; m++) {
        for (size_t k = 0; k < 2; k++) {
            for (size_t n = 0; n < 2; n++) {
                auto& brgemmCtx = brgCtxs0[getBrgIdx(m, k, n)];

                auto M_ = m ? M_tail
                            : M < M_blk ? 0 : M_blk;
                auto N_ = n ? N0_tail : N0 - N0_tail;
                auto K_ = k ? K0_tail : K0 - K0_tail;
                auto beta = k && brgCtxs0[getBrgIdx(m, 0, n)].K != 0 ? 1.0f : 0.0f;

                brgemmCtx.M = M_;
                brgemmCtx.N = N_;
                brgemmCtx.K = K_;
                brgemmCtx.LDA = batch1 * K0;
                brgemmCtx.LDB = rnd_up(N0, N0_blk);
                brgemmCtx.LDC = N0;
                brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(dataPrecision));
                brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(dataPrecision));
                brgemmCtx.beta = beta;

                // don't create brgemm kernels for empty tiles
                if (M_ != 0 && K_ != 0 && N_ != 0) {
                    if (brg0BaseIdx == -1)
                        brg0BaseIdx = getBrgIdx(m, k, n);
                    init_brgemm(brgemmCtx, brgKernels0[getBrgIdx(m, k, n)], brg0WithAMX);
                }
            }
        }
    }

    auto& brgemmCtx0 = brgCtxs0[brg0BaseIdx];

    // TODO: matrix A copy should be performed to enable AMX matmuls for arbitrary shapes
    // if (brgemmCtx0.is_with_amx && K0_tail) {
    //     init_brgemm_copy_a(brgCopyAKernel0, K0, K0_blk, K0_tail, brgemmCtx0.LDA, brgemmCtx0.dt_in0);
    // }

    if (brgemmCtx0.is_with_amx || dataPrecision == Precision::I8 || dataPrecision == Precision::BF16) {
        init_brgemm_copy_b(brgCopyBKernel0, N0, N0_blk, N0_tail, brgemmCtx0.LDB, brgemmCtx0.K,
            brgemmCtx0.is_with_amx, brgemmCtx0.dt_in0, brgemmCtx0.dt_in1);
    }

    dimsMatMul1Out = {dimsMatMul0Out[0], dimsMatMul0Out[1], dimsMatMul0Out[2], dimsMatMul1In1[3]};

    N1 = dimsMatMul1Out[3];
    K1 = dimsMatMul0Out[3];

    brg1VnniFactor = 4 / dataPrecision.size();
    bool brg1WithAMX = isAMXSupported && dataPrecision != Precision::FP32 && (K1 % brg1VnniFactor == 0) && (N1 % brg1VnniFactor == 0);

    N1_blk = dataPrecision == Precision::FP32 ? N1 : matmulOptimalN;
    N1_tail = N1 % N1_blk;
    K1_blk = brg1WithAMX ? matmulOptimalK : K1;
    K1_tail = K1 % K1_blk;

    size_t brg1BaseIdx = -1;
    for (size_t m = 0; m < 2; m++) {
        for (size_t k = 0; k < 2; k++) {
            for (size_t n = 0; n < 2; n++) {
                auto& brgemmCtx = brgCtxs1[getBrgIdx(m, k, n)];

                auto M_ = m ? M_tail
                            : M < M_blk ? 0 : M_blk;
                auto N_ = n ? N1_tail : N1 - N1_tail;
                auto K_ = k ? K1_tail : K1 - K1_tail;

                auto beta = k && brgCtxs1[getBrgIdx(m, 0, n)].K != 0 ? 1.0f : 0.0f;
                brgemmCtx.M = M_;
                brgemmCtx.N = N_;
                brgemmCtx.K = K_;
                brgemmCtx.LDA = K1;
                brgemmCtx.LDB = dataPrecision == Precision::FP32 ? batch1 * N1 : rnd_up(N1, N1_blk);
                brgemmCtx.LDC = dataPrecision == Precision::FP32 ? batch1 * N1 : N1;
                brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(dataPrecision));
                brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(dataPrecision));
                brgemmCtx.beta = beta;

                // don't create brgemm kernels for empty tiles
                if (M_ != 0 && K_ != 0 && N_ != 0) {
                    if (brg1BaseIdx == -1)
                        brg1BaseIdx = getBrgIdx(m, k, n);

                    init_brgemm(brgemmCtx, brgKernels1[getBrgIdx(m, k, n)], brg1WithAMX);
                }
            }
        }
    }

    auto& brgemmCtx1 = brgCtxs1[brg1BaseIdx];
    if (brgemmCtx1.is_with_amx || dataPrecision == Precision::I8 || dataPrecision == Precision::BF16) {
        init_brgemm_copy_b(brgCopyBKernel1, batch1 * N1, N1_blk, N1_tail, brgemmCtx1.LDB, brgemmCtx1.K,
            brgemmCtx1.is_with_amx, brgemmCtx1.dt_in0, brgemmCtx1.dt_in1);
    }

    Precision accPrecision = dataPrecision == Precision::I8 ? Precision::I32 : Precision::FP32;

    bufferMatMul0In0Size = M_blk * rnd_up(K0, K0_blk) * dataPrecision.size();
    bufferMatMul0In1Size = rnd_up(K0, brg0VnniFactor) * rnd_up(N0, N0_blk) * dataPrecision.size();
    bufferMatMul0OutSize = brgemmCtx0.M * N0 * accPrecision.size();
    bufferMatMul1In1Size = rnd_up(K1, brg1VnniFactor) * rnd_up(N1, N1_blk) * dataPrecision.size();
    bufferMatMul1OutSize = brgemmCtx1.M * N1 * accPrecision.size();
    bufferCompensation0Size = rnd_up(N0, N0_blk);
    bufferCompensation1Size = rnd_up(N1, N1_blk);

    if (brgCopyAKernel0) {
        bufferMatMul0In0.resize(numThreads * bufferMatMul0In0Size);
    }
    bufferMatMul0In1.resize(numThreads * bufferMatMul0In1Size);
    bufferMatMul0Out.resize(numThreads * bufferMatMul0OutSize);
    bufferMatMul1In1.resize(numThreads * bufferMatMul1In1Size);
    bufferMatMul1Out.resize(numThreads * bufferMatMul1OutSize);
    if (brgemmCtx0.is_with_comp) {
        bufferCompensation0.resize(numThreads * bufferCompensation0Size);
    }
    if (brgemmCtx1.is_with_comp) {
        bufferCompensation1.resize(numThreads * bufferCompensation1Size);
    }

    if (brgemmCtx0.is_with_amx || brgemmCtx1.is_with_amx) {
        wsp.resize(numThreads * wsp_size_per_thread);
    }

    jit_mul_add_softmax_compile_params jcp;
    jcp.src_prc = dataPrecision == Precision::I8 ? Precision::I32 : Precision::FP32;
    jcp.dst_prc = dataPrecision;
    jcp.work_amount = N0;
    jcp.with_scales0 = !fqScales0.empty();
    jcp.broadcast_scales0 = fqScales0.size() == 1;
    jcp.with_scales1 = !fqScales1.empty();
    jcp.broadcast_scales1 = fqScales1.size() == 1;

    if (mayiuse(cpu_isa_t::avx512_core)) {
        mulAddSoftmaxKernel.reset(new jit_mul_add_softmax_kernel<cpu_isa_t::avx512_core>(jcp));
    } else if (mayiuse(cpu_isa_t::avx2)) {
        mulAddSoftmaxKernel.reset(new jit_mul_add_softmax_kernel<cpu_isa_t::avx2>(jcp));
    } else if (mayiuse(cpu_isa_t::sse41)) {
        mulAddSoftmaxKernel.reset(new jit_mul_add_softmax_kernel<cpu_isa_t::sse41>(jcp));
    } else {
        THROW_ERROR << "cannot create jit eltwise kernel";
    }

    if (dataPrecision != Precision::FP32) {
        jit_convert_reorder_compile_params jcp;
        jcp.src_prc = dataPrecision == Precision::I8 ? Precision::I32 : Precision::FP32;
        jcp.dst_prc = dataPrecision;
        jcp.inner_work_amount = N1;
        jcp.with_scales = !fqScales2.empty();
        jcp.broadcast_scales = fqScales2.size() == 1;
        jcp.src_stride = N1;
        jcp.dst_stride = batch1 * N1;

        if (mayiuse(cpu_isa_t::avx512_core)) {
            convertReorderKernel.reset(new jit_convert_reorder_kernel<cpu_isa_t::avx512_core>(jcp));
        } else if (mayiuse(cpu_isa_t::avx2)) {
            convertReorderKernel.reset(new jit_convert_reorder_kernel<cpu_isa_t::avx2>(jcp));
        } else if (mayiuse(cpu_isa_t::sse41)) {
            convertReorderKernel.reset(new jit_convert_reorder_kernel<cpu_isa_t::sse41>(jcp));
        } else {
            THROW_ERROR << "cannot create jit eltwise kernel";
        }
    }

    if (mulAddSoftmaxKernel)
        mulAddSoftmaxKernel->create_ker();

    if (convertReorderKernel)
        convertReorderKernel->create_ker();
}

template<typename T>
static void reorder2D(const T* pin, T* pout, const std::vector<size_t>& dimsOut,
               const std::vector<size_t>& stridesOut, const std::vector<size_t>& stridesIn) {
    for (int i0 = 0; i0 < dimsOut[0]; i0++) {
        for (int i1 = 0; i1 < dimsOut[1]; i1++) {
            pout[i0 * stridesOut[0] + i1 * stridesOut[1]] = pin[i0 * stridesIn[0] + i1 * stridesIn[1]];
        }
    }
}

void MHA::callBrgemm(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& brgKernel, const void* pin0, const void* pin1, void* pout, void* wsp) {
    if (ctx.is_with_amx)
        amx_tile_configure(ctx.palette);
    if (ctx.is_with_comp) {
        brgemm_post_ops_data_t post_ops_data;
        brgemm_kernel_execute_postops(brgKernel.get(), 1, pin0, pin1, nullptr, pout, pout, post_ops_data, wsp);
    } else {
        brgemm_kernel_execute(brgKernel.get(), 1, pin0, pin1, nullptr, pout, wsp);
    }
}

template<typename data_type, typename acc_type>
void MHA::mhaImpl(dnnl::stream strm) {
    const data_type* pTranspose0In0 = reinterpret_cast<const data_type*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    const data_type* pTranspose1In0 = reinterpret_cast<const data_type*>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    const float* pMulIn1 = reinterpret_cast<const float*>(getParentEdgeAt(2)->getMemoryPtr()->GetPtr());
    const float* pAddIn1 = reinterpret_cast<const float*>(getParentEdgeAt(3)->getMemoryPtr()->GetPtr());
    const data_type* pTranspose2In0 = reinterpret_cast<const data_type*>(getParentEdgeAt(4)->getMemoryPtr()->GetPtr());
    data_type* pout = reinterpret_cast<data_type*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    parallel_for2d(dimsMatMul0Out[0], dimsMatMul0Out[1], [&](size_t i0, size_t i1) {
        size_t threadNum = parallel_get_thread_num();

        auto pTranspose0In0_aux = pTranspose0In0 + i0 * strTranspose0In0[0] + i1 * strTranspose0In0[2]; // order 0213
        auto pTranspose1In0_aux = pTranspose1In0 + i0 * strTranspose1In0[0] + i1 * strTranspose1In0[2]; // order 0231

        auto pAddIn1_aux = pAddIn1 + i0 * strAddIn1[0]; // order 0231

        auto bufferMatMul0In1_local = reinterpret_cast<void*>(bufferMatMul0In1.data() + threadNum * bufferMatMul0In1Size);
        auto bufferMatMul0Out_local = reinterpret_cast<void*>(bufferMatMul0Out.data() + threadNum * bufferMatMul0OutSize);
        auto bufferMatMul1In1_local = reinterpret_cast<void*>(bufferMatMul1In1.data() + threadNum * bufferMatMul1In1Size);
        auto bufferMatMul1Out_local = reinterpret_cast<void*>(bufferMatMul1Out.data() + threadNum * bufferMatMul1OutSize);

        auto pTranspose1Out_aux = brgCopyBKernel0 ? reinterpret_cast<data_type*>(bufferMatMul1In1_local)
                                                    : reinterpret_cast<data_type*>(bufferMatMul0In1_local);
        auto pTranspose2In0_aux = pTranspose2In0 + i0 * strTranspose2In0[0] + i1 * strTranspose2In0[2]; // order 0213

        reorder2D(pTranspose1In0_aux, reinterpret_cast<data_type*>(pTranspose1Out_aux), {K0, N0}, {N0, 1},
                                                                    {strTranspose1In0[3], strTranspose1In0[1]});

        auto bufferCompensation0_aux = !bufferCompensation0.empty()
            ? bufferCompensation0.data() + threadNum * bufferCompensation0Size
            : nullptr;
        auto bufferCompensation1_aux = !bufferCompensation1.empty()
            ? bufferCompensation1.data() + threadNum * bufferCompensation1Size
            : nullptr;

        auto wsp_local = !wsp.empty() ? wsp.data() + threadNum * wsp_size_per_thread : nullptr;

        auto pMatMul0In1 = reinterpret_cast<const data_type*>(pTranspose1Out_aux);
        if (brgCopyBKernel0) {
            for (size_t nb = 0; nb < div_up(N0, N0_blk); nb++) {
                auto pCopyKernel0In = pMatMul0In1 + nb * N0_blk;
                auto pCopyKernel0Out = reinterpret_cast<data_type*>(bufferMatMul0In1_local) + nb * N0_blk * brg0VnniFactor;

                auto ctx = jit_brgemm_matmul_copy_b_t::ctx_t();

                const bool is_N_tail = (N0 - nb * N0_blk < N0_blk);
                ctx.current_N_blk = is_N_tail ? N0_tail : N0_blk;
                ctx.src = pCopyKernel0In;
                ctx.tr_src = pCopyKernel0Out;
                ctx.compensation_ptr = bufferCompensation0_aux + nb * N0_blk;
                ctx.zp_a_compensation_ptr = nullptr;
                ctx.zp_a_neg_value_ptr = nullptr;
                ctx.current_K_start = 0;
                ctx.current_K_iters = K0;

                (*brgCopyBKernel0)(&ctx);
            }

            pMatMul0In1 = reinterpret_cast<data_type*>(bufferMatMul0In1_local);
        }

        auto pMatMul1In1 = reinterpret_cast<const data_type*>(pTranspose2In0_aux);
        if (brgCopyBKernel1) {
            for (size_t nb = 0; nb < div_up(N1, N1_blk); nb++) {
                auto pCopyKernel1In = pMatMul1In1 + nb * N1_blk;
                auto pCopyKernel1Out = reinterpret_cast<data_type*>(bufferMatMul1In1_local) + nb * N1_blk * brg1VnniFactor;

                auto ctx = jit_brgemm_matmul_copy_b_t::ctx_t();

                const bool is_N_tail = (N1 - nb * N1_blk < N1_blk);
                ctx.current_N_blk = is_N_tail ? N1_tail : N1_blk;
                ctx.src = pCopyKernel1In;
                ctx.tr_src = pCopyKernel1Out;
                ctx.compensation_ptr = bufferCompensation1_aux + nb * N1_blk;
                ctx.zp_a_compensation_ptr = nullptr;
                ctx.zp_a_neg_value_ptr = nullptr;
                ctx.current_K_start = 0;
                ctx.current_K_iters = K1;

                (*brgCopyBKernel1)(&ctx);
            }

            pMatMul1In1 = reinterpret_cast<const data_type*>(bufferMatMul1In1_local);
        }

        for (size_t mb = 0; mb < div_up(M, M_blk); mb++) {
            const bool is_M_tail = (M - mb * M_blk < M_blk);
            auto cur_M_blk = is_M_tail ? M_tail : M_blk;

            auto pMatMul0In0 = reinterpret_cast<const data_type*>(pTranspose0In0_aux) + mb * M_blk * batch1 * K0;

            // TODO: matrix A copy should be performed to enable AMX matmuls for arbitrary shapes
            // if (brgCopyAKernel0) {
            //     auto bufferMatMul0In0_local = reinterpret_cast<void*>(bufferMatMul0In0.data() + threadNum * bufferMatMul0In0Size);

            //     auto pCopyKernel0In = pMatMul0In0;
            //     auto pCopyKernel0Out = reinterpret_cast<data_type*>(bufferMatMul0In0_local);

            //     auto ctx = jit_brgemm_matmul_copy_a_t::ctx_t();

            //     ctx.current_M_blk = cur_M_blk;
            //     ctx.zp_b_compensation_buffer_ptr = nullptr;
            //     ctx.zp_a_compensation_result_ptr = nullptr;
            //     ctx.zp_b_neg_value_ptr = nullptr;
            //     ctx.zp_ab_comp_ptr = nullptr;
            //     ctx.src = pCopyKernel0In;
            //     ctx.tr_src = pCopyKernel0Out;
            //     ctx.current_K_start = 0;
            //     ctx.current_K_blk = K0;

            //     (*brgCopyAKernel0)(&ctx);

            //     pMatMul0In0 = reinterpret_cast<const data_type*>(bufferMatMul0In0_local);
            // }

            auto pMatMul0Out = reinterpret_cast<acc_type*>(bufferMatMul0Out_local);

            size_t brgIdx0 = getBrgIdx(0, 0, 0);
            size_t K0_step0 = brgCtxs0[brgIdx0].K;
            size_t K0_step1 = brgCtxs0[brgIdx0].K * brgCtxs0[brgIdx0].LDB;
            size_t N0_step0 = brgCtxs0[brgIdx0].N * brg0VnniFactor;
            size_t N0_step1 = brgCtxs0[brgIdx0].N;
            for (size_t n = 0; n < 2; n++) {
                for (size_t k = 0; k < 2; k++) {
                    size_t mIdx = is_M_tail ? 1 : 0;
                    auto& brgemmCtx = brgCtxs0[getBrgIdx(mIdx, k, n)];

                    auto wsp = brgemmCtx.is_with_comp
                        ? reinterpret_cast<void*>(bufferCompensation0_aux + n * N0_step1)
                        : reinterpret_cast<void*>(wsp_local);

                    if (brgemmCtx.K != 0 && brgemmCtx.N != 0) {
                        callBrgemm(brgemmCtx, brgKernels0[getBrgIdx(mIdx, k, n)],
                            pMatMul0In0 + k * K0_step0, pMatMul0In1 + k * K0_step1 + n * N0_step0, pMatMul0Out + n * N0_step1, wsp);
                    }
                }
            }

            for (size_t m = 0; m < cur_M_blk; m++) {
                jit_mul_add_softmax_call_args call_args;
                call_args.p_mul_in0 = reinterpret_cast<acc_type*>(pMatMul0Out) + m * N0;
                call_args.p_mul_in1 = std::is_same<data_type, int8_t>::value ? pMulIn1 : pMulIn1 + i1;
                call_args.p_add_in1 = pAddIn1_aux;
                call_args.p_out = reinterpret_cast<data_type*>(pMatMul0Out) + m * N0;
                call_args.p_buffer = reinterpret_cast<float*>(pMatMul0Out) + m * N0;
                call_args.p_scales0 = fqScales0.data();
                call_args.p_scales1 = fqScales1.data();

                (*mulAddSoftmaxKernel)(&call_args);
            }

            auto pMatMul1In0 = reinterpret_cast<const data_type*>(bufferMatMul0Out_local);
            auto pOut_aux = pout + i0 * strOut[0] + i1 * strOut[2];

            auto pMatMul1Out = std::is_same<data_type, acc_type>::value
                ? reinterpret_cast<acc_type*>(pOut_aux) + mb * M_blk * batch1 * N1
                : reinterpret_cast<acc_type*>(bufferMatMul1Out_local);

            size_t brgIdx1 = getBrgIdx(0, 0, 0);
            size_t K1_step0 = brgCtxs1[brgIdx1].K;
            size_t K1_step1 = brgCtxs1[brgIdx1].K * brgCtxs1[brgIdx1].LDB;
            size_t N1_step0 = brgCtxs1[brgIdx1].N * brg1VnniFactor;
            size_t N1_step1 = brgCtxs1[brgIdx1].N;
            for (size_t n = 0; n < 2; n++) {
                for (size_t k = 0; k < 2; k++) {
                    size_t mIdx = is_M_tail ? 1 : 0;
                    auto& brgemmCtx = brgCtxs1[getBrgIdx(mIdx, k, n)];

                    auto wsp = brgemmCtx.is_with_comp
                        ? reinterpret_cast<void*>(bufferCompensation1_aux + n * N1_step1)
                        : reinterpret_cast<void*>(wsp_local);

                    if (brgemmCtx.K != 0 && brgemmCtx.N != 0) {
                        callBrgemm(brgemmCtx, brgKernels1[getBrgIdx(mIdx, k, n)],
                            pMatMul1In0 + k * K1_step0, pMatMul1In1 + k * K1_step1 + n * N1_step0, pMatMul1Out + n * N1_step1, wsp);
                    }
                }
            }

            if (convertReorderKernel) {
                jit_convert_reorder_call_args call_args;
                call_args.p_in = pMatMul1Out;
                call_args.p_out = reinterpret_cast<data_type*>(pOut_aux) + mb * M_blk * batch1 * N1;
                call_args.p_scales = fqScales2.data();
                call_args.outter_work_amount = cur_M_blk;

                (*convertReorderKernel)(&call_args);
            }
        }
    });
}

void MHA::execute(dnnl::stream strm) {
    switch (dataPrecision) {
        case Precision::FP32: mhaImpl<float, float>(strm); break;
        case Precision::BF16: mhaImpl<bfloat16_t, float>(strm); break;
        case Precision::I8:   mhaImpl<int8_t, int32_t>(strm); break;
        default: THROW_ERROR << "doesn't support precision: " << dataPrecision;
    }
}

void MHA::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool MHA::created() const {
    return getType() == Type::MHA;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
