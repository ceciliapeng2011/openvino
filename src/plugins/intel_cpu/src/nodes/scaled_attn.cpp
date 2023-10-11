// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.h"

#include <dnnl_extension_utils.h>
#include <onednn/dnnl.h>

#include <algorithm>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <ie_ngraph_utils.hpp>
#include <string>
#include <shape_inference/shape_inference_internal_dyn.hpp>
#include <vector>

#include "common/cpu_memcpy.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/plain_tensor.hpp"
#include <openvino/op/scaled_dot_product_attention.hpp>

#ifdef OV_CPU_WITH_MLAS
#    include "mlas/sgemm.hpp"
#endif

#include "utils/plain_tensor.hpp"
#include "utils/profiler.hpp"
#include "scaled_attn_softmax.hpp"
#include "scaled_attn_dot_product.hpp"
#include "scaled_attn_acc_value.hpp"
#include "scaled_attn_reduce.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::Extensions::Cpu::XARCH;

namespace ov {
namespace intel_cpu {
namespace node {

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

//============================ kernels ============================
enum KernelTypes { KT_REF, KT_ONEDNN, KT_MLAS};

// default implementation: reference
template <KernelTypes KType, typename T>
struct MHA_kernel {
    MHA_kernel() = default;

    template <typename D>
    float dot_product(const D* a, const D* b, int len, int stride_b = 1) {
        float result = 0;
        if (stride_b == 1) {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        } else {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i * stride_b]);
        }
        return result;
    }

    void softmax(float* a, int len) {
        float max = *std::max_element(a, a + len);
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            a[i] = exp(a[i] - max);
            sum += a[i];
        }
        float scale = 1.0f / sum;
        for (int i = 0; i < len; i++) {
            a[i] *= scale;
        }
    }

    template <typename D>
    void accumulate(float* acc, const D* v, int len, float weight = 1.0f) {
        for (int i = 0; i < len; i++) {
            acc[i] += static_cast<float>(v[i]) * weight;
        }
    }

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, q_len, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor<T>& query,
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<T>& output_emb,
                    bool has_out_transpose,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        auto k_stride_s = present_key.stride(3);
        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        parallel_for2d(B, H, [&](size_t b, size_t h) {
            std::vector<float> attn_score(kv_len);
            std::vector<float> word_vec(head_size, 0.0f);

            for (size_t m = 0; m < q_len; m++) {
                // dot-product to get attention scores
                auto* q = &query.at({b, h, m, 0});
                // how many key/values can be accessed causally
                auto ncausal = kv_len;
                // no causall mask is set and it's not fused into attention_mask
                if (auto_causal)
                    ncausal = kv_len - q_len + m + 1;
                for (size_t n = 0; n < ncausal; n++) {
                    auto* k = &present_key.at({b, h, n, 0});
                    attn_score[n] = dot_product(q, k, head_size, k_stride_s) * d_scale;

                    // apply alibi tensor
                    if (alibi_mask)
                        attn_score[n] += alibi_mask.at({b, h, m, n}, true);

                    // apply attention mask (maybe combined with causal_mask)
                    if (attention_mask)
                        attn_score[n] += attention_mask.at({b, h, m, n}, true);

                    // apply causal_mask
                    if (causal_mask) {
                        bool is_zero = causal_mask.at({b, h, m, n}, true) == 0;
                        if (select_nfltmax_at_0) {
                            if (is_zero)
                                attn_score[n] = -FLT_MAX;
                        } else {
                            if (!is_zero) {
                                attn_score[n] = -FLT_MAX;
                            }
                        }
                    }
                }

                // softmax
                softmax(&attn_score[0], ncausal);

                // linearly combine value
                word_vec.assign(head_size, 0.0f);
                for (size_t n = 0; n < ncausal; n++) {
                    auto* v = &present_value.at({b, h, n, 0});
                    accumulate(word_vec.data(), v, head_size, attn_score[n]);
                }

                // output [B, L1, H*head_size]
                auto* out = has_out_transpose ? &output_emb.at({b, m, h * head_size}) : &output_emb.at({b, h, m});
                std::copy(word_vec.begin(), word_vec.end(), out);
            }
        });
    }
};

template <typename T>
struct MHA_kernel<KT_ONEDNN, T> {
    // q: [B, H, q_len, S]
    // k: [B, H, kv_len, S]
    // v: [B, H, kv_len, S]
    dnnl::memory::desc q_md;
    dnnl::memory::desc k_md;
    dnnl::memory::desc weight_md;
    dnnl::memory::desc v_md;
    dnnl::memory::desc out_md;
    dnnl::memory attn_score;
    dnnl::memory attn_weight;
    dnnl::matmul qk_prim;
    dnnl::matmul wv_prim;
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    void prepare_prim(dnnl::stream strm, size_t B, size_t H, size_t q_len, size_t kv_len, size_t S, bool has_out_transpose) {
        auto make_dnnl_dims = [](const std::vector<size_t>& dims) {
            dnnl::memory::dims dnnl_dims(dims.size());
            for (size_t i = 0; i < dims.size(); i++)
                dnnl_dims[i] = static_cast<dnnl::memory::dim>(dims[i]);
            return dnnl_dims;
        };
        auto qkv_dt = precision_of<T>::value == Precision::FP32 ? dt::f32 : dt::bf16;
        dnnl::memory::desc cur_q_md(make_dnnl_dims({B, H, q_len, S}), qkv_dt, tag::abcd);
        dnnl::memory::desc cur_k_md(make_dnnl_dims({B, H, kv_len, S}), qkv_dt, tag::abcd);
        if (cur_q_md == q_md && cur_k_md == k_md)
            return;

        q_md = cur_q_md;
        k_md = cur_k_md;
        dnnl::memory::desc attn_md(make_dnnl_dims({B, H, q_len, kv_len}), dt::f32, tag::abcd);
        k_md = k_md.permute_axes({0, 1, 3, 2});
        auto qk_pd = dnnl::matmul::primitive_desc(strm.get_engine(), q_md, k_md, attn_md);
        qk_prim = dnnl::matmul(qk_pd);

        weight_md = dnnl::memory::desc(make_dnnl_dims({B, H, q_len, kv_len}), qkv_dt, tag::abcd);
        v_md = dnnl::memory::desc(make_dnnl_dims({B, H, kv_len, S}), qkv_dt, tag::abcd);
        out_md = dnnl::memory::desc(make_dnnl_dims({B, H, q_len, S}), qkv_dt, tag::abcd);
        if (has_out_transpose)
            out_md = out_md.permute_axes({0, 2, 1, 3});
        auto wv_pd = dnnl::matmul::primitive_desc(strm.get_engine(), weight_md, v_md, out_md);
        wv_prim = dnnl::matmul(wv_pd);

        if (!attn_score || attn_md.get_size() > attn_score.get_desc().get_size()) {
            attn_score = dnnl::memory(attn_md, strm.get_engine());
            attn_weight = dnnl::memory(weight_md, strm.get_engine());
        }
    }

    void exec_qk(dnnl::stream strm, PlainTensor<T>& query, PlainTensor<T>& present_key) {
        dnnl::memory q(q_md, strm.get_engine(), query.data());
        dnnl::memory k(k_md, strm.get_engine(), present_key.data());
        qk_prim.execute(strm, {{DNNL_ARG_SRC, q},
                               {DNNL_ARG_WEIGHTS, k},
                               {DNNL_ARG_DST, attn_score}});
    }

    void exec_kv(dnnl::stream strm, PlainTensor<T>& present_value, PlainTensor<T>& output_emb) {
        dnnl::memory v(v_md, strm.get_engine(), present_value.data());
        dnnl::memory out(out_md, strm.get_engine(), output_emb.data());
        wv_prim.execute(strm, {{DNNL_ARG_SRC, attn_weight}, {DNNL_ARG_WEIGHTS, v}, {DNNL_ARG_DST, out}});
    }

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi          [B, H, q_len, kv_len]
    // output_emb    [B, L1, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor<T>& query,
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<T>& output_emb,
                    bool has_out_transpose,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;
        prepare_prim(strm, B, H, q_len, kv_len, head_size, has_out_transpose);
        exec_qk(strm, query, present_key);

        PlainTensor<float> score;
        score.resize({B, H, q_len, kv_len}, static_cast<float*>(attn_score.get_data_handle()));
        PlainTensor<T> weight;
        weight.resize({B, H, q_len, kv_len}, static_cast<T*>(attn_weight.get_data_handle()));
        // softmax
        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t m) {
            // apply attention mask & sofmax
            auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
            attn_softmax(&score.at({b, h, m, 0}),
                         &weight.at({b, h, m, 0}),
                         d_scale,
                         alibi_mask ? &alibi_mask.at({b, h, m, 0}, true) : nullptr,
                         attention_mask ? &attention_mask.at({b, h, m, 0}, true) : nullptr,
                         causal_mask ? &causal_mask.at({b, h, m, 0}, true) : nullptr,
                         select_nfltmax_at_0,
                         ncausal,
                         kv_len,
                         precision_of<T>::value);
        });
        exec_kv(strm, present_value, output_emb);
    }
};

#ifdef OV_CPU_WITH_MLAS
template <>
struct MHA_kernel<KT_MLAS, float> {
    size_t m_block_size;
    // buffer to hold qk temp
    std::vector<PlainTensor<float>> qk_buffers;

    MHA_kernel() {
        m_block_size = std::getenv("MBLK") ? atoi(std::getenv("MBLK")) : 4;
        qk_buffers.resize(parallel_get_max_threads(), PlainTensor<float>(true));
    }

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor<float>& query,
                    PlainTensor<float>& present_key,
                    PlainTensor<float>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<float>& output_emb,
                    bool has_out_transpose,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto h_group_num = present_key.size(1);
        size_t h_each_group_len = H / h_group_num;

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);
        auto k_stride_s = present_key.stride(3);

        auto m_blocks = (q_len + m_block_size - 1) / m_block_size;
        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        parallel_for3d(B, H, m_blocks, [&](size_t b, size_t h, size_t m_blk) {
            size_t thread_id = static_cast<size_t>(parallel_get_thread_num());
            auto& qk_buf = qk_buffers[thread_id];

            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;

            auto kv_len_cache_align = (((kv_len * sizeof(float)) + 63) / 64 * 64) / sizeof(float);
            qk_buf.resize({m_block_size, kv_len_cache_align});
            const float* q_ptr = &query.at({b, h, m_start, 0});
            const float* k_ptr = &present_key.at({b, h / h_each_group_len, 0, 0});
            const float* v_ptr = &present_value.at({b, h / h_each_group_len, 0, 0});

            float* alibi_ptr = nullptr;
            auto alibi_stride = 0;
            if (alibi_mask) {
                alibi_ptr = &alibi_mask.at({b, h, 0, 0}, true);
                if (alibi_mask.size(2) > 1)
                    alibi_stride = alibi_mask.stride(2);
            }
            float* attn_mask_ptr = nullptr;
            auto attn_mask_stride = 0;
            if (attention_mask) {
                attn_mask_ptr = &attention_mask.at({b, h, 0, 0}, true);
                if (attention_mask.size(2) > 1)
                    attn_mask_stride = attention_mask.stride(2);
            }
            uint8_t* cmask_ptr = nullptr;
            auto cmask_stride = 0;
            if (causal_mask) {
                cmask_ptr = &causal_mask.at({b, h, 0, 0}, true);
                if (causal_mask.size(2) > 1)
                    cmask_stride = causal_mask.stride(2);
            }

            float* qk = &(qk_buf.at({0, 0}));
            auto qk_m_stride = qk_buf.stride(0);

            if (k_stride_s == 1)
                mlas_sgemm("N",
                           "T",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(2),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);
            else
                mlas_sgemm("N",
                           "N",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(3),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);

            for (size_t m = m_start; m < m_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
                attn_softmax(qk + (m - m_start) * qk_m_stride,
                             qk + (m - m_start) * qk_m_stride,
                             d_scale,
                             alibi_ptr + m * alibi_stride,
                             attn_mask_ptr + m * attn_mask_stride,
                             cmask_ptr + m * cmask_stride,
                             select_nfltmax_at_0,
                             ncausal,
                             kv_len,
                             Precision::FP32);
            }
            mlas_sgemm("N",
                       "N",
                       m_cnt,
                       head_size,
                       kv_len,
                       1.0f,
                       qk,
                       qk_m_stride,
                       v_ptr,
                       present_value.stride(2),
                       0.f,
                       has_out_transpose ? &output_emb.at({b, m_start, h * head_size}) : &output_emb.at({b, h, m_start}),
                       has_out_transpose ? output_emb.stride(1) : output_emb.stride(2),
                       1);
        });
    }
};
#endif

// 2nd token case : only 1 token in query
template <typename RT>
struct MHA_1Token {
    PlainTensor<float> m_attn_w;
    PlainTensor<float> m_temp;

    MHA_1Token() : m_attn_w(true), m_temp(true) {}

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(PlainTensor<RT>& query,
                    PlainTensor<RT>& present_key,
                    PlainTensor<RT>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<RT>& output_emb,
                    const PlainTensor<int32_t>& beams,
                    bool has_out_transpose,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto S = query.size(3);
        auto kv_len = present_key.size(2);
        auto h_group_num = present_key.size(1);
        size_t h_each_group_len = 0;
        if (h_group_num != H) {
            h_each_group_len = H / h_group_num;
        }

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(S);

        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        // use per-token kernel, for each k,v token
        //  attn mask is a matrix of q_len(kv_len)
        m_attn_w.resize({B, H, q_len, kv_len});

        if (h_each_group_len) {
            parallel_for3d(B, H / h_each_group_len, kv_len, [&](size_t b, size_t h_group, size_t pk) {
                // which batch item should be used at postion pk?
                auto b_kv = beams ? beams.at({b, pk}) : b;
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                        auto sum = attn_dot_product(&query.at({b, h, pq, 0}),
                                                          &present_key.at({b_kv, h_group, pk, 0}),
                                                        S,
                                            precision_of<RT>::value);
                        m_attn_w.at({b, h, pq, pk}) = sum;
                    }
                }
            });
        } else {
            parallel_for3d(B, H, kv_len, [&](size_t b, size_t h, size_t pk) {
                // which batch item should be used at postion pk?
                auto b_kv = beams ? beams.at({b, pk}) : b;
                for (size_t pq = 0; pq < q_len; pq++) {
                    auto sum = attn_dot_product(&query.at({b, h, pq, 0}),
                                                      &present_key.at({b_kv, h, pk, 0}),
                                                      S,
                                                      precision_of<RT>::value);
                    m_attn_w.at({b, h, pq, pk}) = sum;
                }
            });
        }

        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            // apply attention mask & sofmax
            auto ncausal = auto_causal ? (kv_len - q_len + pq + 1) : kv_len;
            float* alibi_ptr = alibi_mask ? &alibi_mask.at({b, h, pq, 0}, true) : nullptr;
            float* attn_mask_ptr = attention_mask ? &attention_mask.at({b, h, pq, 0}, true) : nullptr;
            uint8_t* cmask_ptr = causal_mask ? &causal_mask.at({b, h, pq, 0}, true) : nullptr;
            attn_softmax(&m_attn_w.at({b, h, pq, 0}),
                         &m_attn_w.at({b, h, pq, 0}),
                         d_scale,
                         alibi_ptr,
                         attn_mask_ptr,
                         cmask_ptr,
                         select_nfltmax_at_0,
                         ncausal,
                         kv_len,
                         Precision::FP32);
        });

        // attn_w * V
        auto nthr = parallel_get_max_threads();
        m_temp.resize({static_cast<size_t>(nthr), B, q_len, H, S});
        // m_attn_w {B, H, q_len, kv_len}
        if (h_each_group_len == 0) {
            parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
                size_t start{0}, end{0};
                splitter(B * H * kv_len, nthr, ithr, start, end);

                memset(&m_temp.at({ithr, 0, 0, 0, 0}), 0, m_temp.stride(0) * sizeof(float));

                size_t b, h, pv;
                parallel_it_init(start, b, B, h, H, pv, kv_len);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.at({b, pv}) : b;
                    auto* v = &present_value.at({b_kv, h, pv, 0});
                    for (size_t pq = 0; pq < q_len; pq++) {
                        auto* out = &m_temp.at({ithr, b, pq, h, 0});
                        auto weight = m_attn_w.at({b, h, pq, pv});
                        attn_acc_value(out, weight, v, S, precision_of<RT>::value);
                    }
                    parallel_it_step(b, B, h, H, pv, kv_len);
                }
            });
        } else {
            parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
                size_t start{0}, end{0};
                splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

                memset(&m_temp.at({ithr, 0, 0, 0, 0}), 0, m_temp.stride(0) * sizeof(float));

                size_t b, h_group, pv;
                parallel_it_init(start, b, B, h_group, h_group_num, pv, kv_len);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.at({b, pv}) : b;
                    auto* v = &present_value.at({b_kv, h_group, pv, 0});
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                            auto* out = &m_temp.at({ithr, b, pq, h, 0});
                            auto weight = m_attn_w.at({b, h, pq, pv});
                            attn_acc_value(out, weight, v, S, precision_of<RT>::value);
                        }
                    }
                    parallel_it_step(b, B, h_group, h_group_num, pv, kv_len);
                }
            });
        }

        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            auto* temp = &m_temp.at({0, b, pq, h, 0});
            size_t temp_stride = m_temp.stride(0);
            auto* dst = has_out_transpose ? &output_emb.at({b, pq, h*S}) : &output_emb.at({b, h, pq});
            attn_reduce(dst, temp, nthr, S, temp_stride, precision_of<RT>::value);
        });
    }
};

template <KernelTypes KType, typename T>
struct AttentionExecutor : public ScaledDotProductAttention::Executor {
    PlainTensor<T> q_input;           // f32[B, L1, H*S] / [B, H, L1, S]
    PlainTensor<T> k_input;           // f32[B, L1, H*S]
    PlainTensor<T> v_input;           // f32[B, L1, H*S]
    PlainTensor<T> k_cache;           // f32[B, H, max_kvLen, S]
    PlainTensor<T> v_cache;           // f32[B, H, max_kvLen, S]
    PlainTensor<int32_t> beam_table;  // i32[B, max_kvLen]
    PlainTensor<float> attn_mask;     // f32[B, qLen + kvLen]
    PlainTensor<float> cos_tab;       // f32[max_kv_len, rotary_dims//2]
    PlainTensor<float> sin_tab;       // f32[max_kv_len, rotary_dims//2]

    PlainTensor<T> output_emb;        // f32[B, L1, H*S]

    MHA_kernel<KType, T> kernel;
    MHA_1Token<T> kernel_1tok;

    PlainTensor<T> m_query_emb;  // query with RoPE position embedding

    void execute(dnnl::stream strm, ScaledDotProductAttention* node) override {
        q_input.reset(node->getParentEdgeAt(0)->getMemoryPtr());
        k_input.reset(node->getParentEdgeAt(1)->getMemoryPtr());
        v_input.reset(node->getParentEdgeAt(2)->getMemoryPtr());
        attn_mask.reset(node->getParentEdgeAt(3)->getMemoryPtr());
        bool has_out_transpose = node->is_out_transpose();
        auto rope_type = node->get_rope_type();

        size_t B, H, L1, L0, S;
        if (rope_type != -1) {
            k_cache.reset(node->getParentEdgeAt(4)->getMemoryPtr());
            v_cache.reset(node->getParentEdgeAt(5)->getMemoryPtr());
            cos_tab.reset(node->getParentEdgeAt(6)->getMemoryPtr());
            sin_tab.reset(node->getParentEdgeAt(7)->getMemoryPtr());
            // q: [B, L1, H*S]
            B = q_input.size(0);
            L1 = q_input.size(1);
            L0 = attn_mask.size(1) - L1;
            S = k_cache.size(-1);
        } else {
            // q, k, v: [B, H, L0, S]
            B = q_input.size(0);
            H = q_input.size(1);
            L1 = q_input.size(2);
            L0 = k_input.size(2) - L1;
            S = q_input.size(-1);
        }

        if (has_out_transpose)
            node->redefineOutputMemory({{B, L1, H * S}});
        else
            node->redefineOutputMemory({{B, H, L1, S}});

        ov::intel_cpu::PlainTensor<T> output_emb(node->getChildEdgeAt(0)->getMemoryPtr());
        attn_mask.assert_dims({B, 1, 0, L0 + L1}, true);
        PlainTensor<T> present_key, present_value;

        if (rope_type != -1) {
            auto half_rotary_dims = cos_tab.size(-1);
            cos_tab.assert_dims({0, half_rotary_dims}, true);
            sin_tab.assert_dims({0, half_rotary_dims}, true);

            q_input.assert_dims({B, L1, H * S});
            k_input.assert_dims({B, L1, H * S});
            v_input.assert_dims({B, L1, H * S});

            auto rope_q = q_input.reshape({B, L1, H, S});
            auto rope_k = k_input.reshape({B, L1, H, S});
            auto rope_v = v_input.reshape({B, L1, H, S});

            // kv cache is just a partial view of a big buffer
            m_query_emb.resize({B, H, L1, S});

            present_key = k_cache.index({{0, static_cast<int>(B)},
                                            {0, static_cast<int>(H)},
                                            {0, static_cast<int>(L0 + L1)},
                                            {}});
            present_value = v_cache.index({{0, static_cast<int>(B)},
                                                {0, static_cast<int>(H)},
                                                {0, static_cast<int>(L0 + L1)},
                                                {}});

            auto rotary_dims = half_rotary_dims * 2;

            parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t p) {
                auto p1 = p + L0;
                size_t position_id = p1;

                auto* q_embed = &m_query_emb.at({b, h, p, 0});
                auto* cos = &cos_tab({position_id, 0});
                auto* sin = &sin_tab({position_id, 0});
                T* q;
                T* k;
                T* v;

                q = &rope_q.at({b, p, h, 0});
                k = &rope_k.at({b, p, h, 0});
                v = &rope_v.at({b, p, h, 0});
                auto* present_k = &present_key.at({b, h, p1, 0});    // f32[B, H, L0+L1, 64]
                auto* present_v = &present_value.at({b, h, p1, 0});  // f32[B, H, L0+L1, 64]

                size_t s = 0;
                if (rope_type > 0) {
                    // gptneox RoPE
                    for (size_t i = 0; s < half_rotary_dims; i++, s++) {
                        q_embed[s] = cos[i] * q[s] + sin[i] * (-q[s + half_rotary_dims]);
                        present_k[s] = cos[i] * k[s] + sin[i] * (-k[s + half_rotary_dims]);
                        present_v[s] = v[s];
                    }
                    for (size_t i = 0; s < rotary_dims; i++, s++) {
                        q_embed[s] = cos[i] * q[s] + sin[i] * (q[i]);
                        present_k[s] = cos[i] * k[s] + sin[i] * (k[i]);
                        present_v[s] = v[s];
                    }
                } else {
                    // gptj RoPE
                    for (size_t i = 0; s < rotary_dims; i++, s += 2) {
                        q_embed[s] = cos[i] * q[s] - sin[i] * q[s + 1];
                        q_embed[s + 1] = cos[i] * q[s + 1] + sin[i] * q[s];

                        present_k[s] = cos[i] * k[s] - sin[i] * k[s + 1];
                        present_k[s + 1] = cos[i] * k[s + 1] + sin[i] * k[s];

                        present_v[s] = v[s];
                        present_v[s + 1] = v[s + 1];
                    }
                }

                for (; s < S; s++) {
                    q_embed[s] = q[s];
                    present_k[s] = k[s];
                    present_v[s] = v[s];
                }
            });
        } else {
            q_input.assert_dims({B, H, L1, S});
            k_input.assert_dims({B, H, L0 + L1, S});
            v_input.assert_dims({B, H, L0 + L1, S});
            m_query_emb = q_input;
            present_key = k_input;
            present_value = v_input;
        }

        if (L1 > 1) {
            // multi-token version
            kernel(strm, m_query_emb, present_key, present_value, {}, attn_mask, output_emb, has_out_transpose);
        } else {
            // 1-token version
            kernel_1tok(m_query_emb, present_key, present_value, {}, attn_mask, output_emb, beam_table, has_out_transpose);
        }
    }
};

ScaledDotProductAttention::ScaledDotProductAttention(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto node = std::dynamic_pointer_cast<const ov::op::v12::ScaledDotProductAttention>(op);
    m_is_causal = node->get_is_causal();
}

void ScaledDotProductAttention::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto srcPrecision = getOriginalInputPrecisionAtPort(0);

    auto rtPrecision = srcPrecision;

    if (rtPrecision == Precision::BF16) {
        m_executor = std::make_shared<AttentionExecutor<KT_ONEDNN, ov::bfloat16>>();
    } else {
#ifdef OV_CPU_WITH_MLAS
        m_executor = std::make_shared<AttentionExecutor<KT_MLAS, float>>();
#else
        m_executor = std::make_shared<AttentionExecutor<KT_ONEDNN, float>>();
#endif
    }

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(1), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(2), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, Precision::FP32, getInputShapeAtPort(3), false, -1);

    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void ScaledDotProductAttention::execute(dnnl::stream strm) {
    m_executor->execute(strm, this);
}

bool ScaledDotProductAttention::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = std::dynamic_pointer_cast<const ov::op::v12::ScaledDotProductAttention>(op);
        if (!node) {
            errorMessage = "Only ScaledDotProductAttention operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
