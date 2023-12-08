// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stateful_transpose_sdpa_fusion.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"

namespace ov {
namespace intel_cpu {

StatefulTransposeSDPAFusion::StatefulTransposeSDPAFusion() {
    MATCHER_SCOPE(StatefulTransposeSDPAFusion);
    using namespace ov::pass::pattern;

    auto past_k = wrap_type<opset6::ReadValue>();
    auto past_v = wrap_type<opset6::ReadValue>();
    auto convert_past_k = wrap_type<opset1::Convert>({past_k});
    auto convert_past_v = wrap_type<opset1::Convert>({past_v});
    auto concat_input_k = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past_k, convert_past_k});
    auto concat_input_v = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past_v, convert_past_v});
    auto concat_k = wrap_type<opset6::Concat>({concat_input_k, any_input()});
    auto concat_v = wrap_type<opset6::Concat>({concat_input_v, any_input()});

    auto constant_k = wrap_type<opset6::Constant>();
    auto constant_v = wrap_type<opset6::Constant>();

    auto order_k = wrap_type<opset6::Constant>();
    auto order_v = wrap_type<opset6::Constant>();
    auto transpose_k = wrap_type<opset6::Transpose>({concat_k, order_k});
    auto transpose_v = wrap_type<opset6::Transpose>({concat_v, order_v});

    auto order_q = wrap_type<opset6::Constant>();
    auto q_input = any_input();
    auto transpose_q = wrap_type<opset6::Transpose>({q_input, order_q});
    auto sdp0 = wrap_type<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v});
    auto sdp1 = wrap_type<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v, any_input()});
    auto sdp2 = wrap_type<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v, any_input(), any_input()});
    auto sdp = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sdp0, sdp1, sdp2});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto find_assign = [&](const ov::Output<ov::Node>& out, opset6::Assign*& assign, opset1::Convert*& cvt) {
            auto present_to = out.get_target_inputs();
            if (present_to.size() != 2)
                return;
            for (auto& to : present_to) {
                auto to_node = to.get_node();
                if (auto convert = dynamic_cast<opset1::Convert*>(to_node)) {
                    auto cvt_targets = convert->get_output_target_inputs(0);
                    if (cvt_targets.size() == 1) {
                        to_node = cvt_targets.begin()->get_node();
                        cvt = convert;
                    }
                }
                assign = dynamic_cast<opset6::Assign*>(to_node);
                if (assign)
                    return;
            }
        };

        std::shared_ptr<opset1::Convert> read_cvt_k_node, read_cvt_v_node;
        const auto sdp_node = ov::as_type_ptr<opset13::ScaledDotProductAttention>(root);
        const auto past_k_node = ov::as_type_ptr<opset6::ReadValue>(pattern_map.at(past_k).get_node_shared_ptr());
        const auto past_v_node = ov::as_type_ptr<opset6::ReadValue>(pattern_map.at(past_v).get_node_shared_ptr());
        const auto concat_k_node = ov::as_type_ptr<opset6::Concat>(pattern_map.at(concat_k).get_node_shared_ptr());
        const auto concat_v_node = ov::as_type_ptr<opset6::Concat>(pattern_map.at(concat_v).get_node_shared_ptr());
        if (pattern_map.count(convert_past_k)) {
            read_cvt_k_node = ov::as_type_ptr<opset1::Convert>(pattern_map.at(convert_past_k).get_node_shared_ptr());
            read_cvt_v_node = ov::as_type_ptr<opset1::Convert>(pattern_map.at(convert_past_v).get_node_shared_ptr());
        }
        opset6::Assign* assign_k_node = nullptr, *assign_v_node = nullptr;
        opset1::Convert* assign_cvt_k_node = nullptr, *assign_cvt_v_node = nullptr;
        find_assign(concat_k_node, assign_k_node, assign_cvt_k_node);
        if (!assign_k_node)
            return false;
        if (past_k_node->get_variable_id() != assign_k_node->get_variable_id())
            return false;

        find_assign(concat_v_node, assign_v_node, assign_cvt_v_node);
        if (!assign_v_node)
            return false;
        if (past_v_node->get_variable_id() != assign_v_node->get_variable_id())
            return false;
        auto args = sdp_node->input_values();
        args[0] = pattern_map.at(q_input).get_node_shared_ptr()->output(0);
        args[1] = concat_k_node->input_value(1);
        args[2] = concat_v_node->input_value(1);
        args.push_back(read_cvt_k_node ? read_cvt_k_node->output(0) : past_k_node->output(0));
        args.push_back(read_cvt_v_node ? read_cvt_v_node->output(0) : past_v_node->output(0));
        ov::intel_cpu::ScaledDotProductAttentionWithKVCache::Config config;

        const auto order_q_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_q).get_node_shared_ptr());
        const auto order_k_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_k).get_node_shared_ptr());
        const auto order_v_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_v).get_node_shared_ptr());
        const auto& vec_order_q = order_q_node->cast_vector<int32_t>();
        const auto& vec_order_k = order_k_node->cast_vector<int32_t>();
        const auto& vec_order_v = order_v_node->cast_vector<int32_t>();
        if (vec_order_q != vec_order_k || vec_order_q != vec_order_v) // the transpose order of q/k/v should be the same.
            return false;

        config.is_causal = sdp_node->get_causal();
        config.fuse_concat = true;
        const auto& permute_axes = vec_order_k;
        config.permute_axes.resize(permute_axes.size());
        for (size_t i = 0; i < permute_axes.size(); i++) {
            config.permute_axes[i] = static_cast<size_t>(permute_axes[i]);
        }
        auto& old_node = sdp_node;
        auto new_node = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, {new_node->output(0)});
        if (assign_cvt_k_node)
            assign_cvt_k_node->set_arguments({new_node->output(1)});
        else
            assign_k_node->set_arguments({new_node->output(1)});

        if (assign_cvt_v_node)
            assign_cvt_v_node->set_arguments({new_node->output(2)});
        else
            assign_v_node->set_arguments({new_node->output(2)});

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdp, matcher_name);
    this->register_matcher(m, callback);
}

SDPATransposeReshapeFusion::SDPATransposeReshapeFusion() {
    MATCHER_SCOPE(SDPATransposeReshapeFusion);
    using namespace ov::pass::pattern;

    auto sdp0 = wrap_type<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>({any_input(), any_input(), any_input(), any_input(), any_input()});
    auto sdp1 = wrap_type<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>({any_input(), any_input(), any_input(), any_input(), any_input(), any_input()});
    auto sdp2 = wrap_type<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>({any_input(), any_input(), any_input(), any_input(), any_input(), any_input(), any_input()});
    auto sdpa = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sdp0, sdp1, sdp2});

    // post SDPA Transpose and Reshape
    auto transpose_order = wrap_type<opset6::Constant>();
    auto transpose_out = wrap_type<opset6::Transpose>({sdpa->output(0), transpose_order});

    auto reshape_order = wrap_type<opset6::Constant>();
    auto reshape_out = wrap_type<opset6::Reshape>({transpose_out, reshape_order});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        ov::NodeVector new_ops;

        const auto reshape_node = ov::as_type_ptr<opset6::Reshape>(root);
        const auto reshape_order_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(reshape_order).get_node_shared_ptr());
        const auto transpose_node = ov::as_type_ptr<opset6::Transpose>(pattern_map.at(transpose_out).get_node_shared_ptr());
        const auto transpose_order_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(transpose_order).get_node_shared_ptr());

        auto sdp_node = std::dynamic_pointer_cast<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(transpose_node->get_input_node_shared_ptr(0));
        if (!sdp_node) {
            return false;
        }

        ov::intel_cpu::ScaledDotProductAttentionWithKVCache::Config config;
        config = sdp_node->get_config();
        // update post_permute
        // output_logits BHLS
        // The actual index of B is reverse_permute[0], H is reverse_permute[1], L is reverse_permute[2], S is reverse_permute[3]
        const auto& permute_axes = transpose_order_node->cast_vector<size_t>();
        OPENVINO_ASSERT(permute_axes.size() == 4);
        const auto& reshape_axes = reshape_order_node->cast_vector<size_t>();
        OPENVINO_ASSERT(reshape_axes.size() == 3);

        auto get_reverse_order = [] (const std::vector<size_t>& order) -> std::vector<size_t> {
            std::vector<size_t> reverse;
            reverse.resize(order.size());
            for (size_t i = 0; i < order.size(); i++) {
                const auto itr = std::find(order.begin(), order.end(), i);
                assert(itr != order.end());
                reverse[i] = std::distance(order.begin(), itr);
            }
            return reverse;
        };
        const auto permute_reverse = get_reverse_order(permute_axes);

        OPENVINO_ASSERT(std::abs(static_cast<int64_t>(permute_reverse[1]) - static_cast<int64_t>(permute_reverse[3])) == 1); // HxS

        config.post_permute = permute_reverse;

        auto& old_node = reshape_node;
        auto new_node = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(sdp_node->input_values(), config);
        new_node->set_friendly_name(sdp_node->get_friendly_name()+"/fuseTranspose");
        new_ops.push_back(new_node);

        ov::copy_runtime_info(old_node, new_ops);
        const auto& sdpa_out1 = sdp_node->get_output_target_inputs(1);
        OPENVINO_ASSERT(sdpa_out1.size() == 1);
        sdpa_out1.begin()->get_node()->set_arguments({new_node->output(1)});

        const auto& sdpa_out2 = sdp_node->get_output_target_inputs(2);
        OPENVINO_ASSERT(sdpa_out2.size() == 1);
        sdpa_out2.begin()->get_node()->set_arguments({new_node->output(2)});

        ov::replace_node(old_node, {new_node->output(0)});

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_out, matcher_name);
    this->register_matcher(m, callback);
}
}  // namespace intel_cpu
}  // namespace ov