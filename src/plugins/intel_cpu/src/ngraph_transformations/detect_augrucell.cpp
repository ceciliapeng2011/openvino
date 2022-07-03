// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detect_augrucell.hpp"
#include "op/augru_cpu.hpp"
#include "op/augruseq_cpu.hpp"
#include <numeric>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "snippets/op/subgraph.hpp"

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

using namespace ngraph;
using namespace opset9;

/**
 * @ingroup ie_transformation_common_api
 * @brief GRUCellDecomposition transformation decomposes GRUCell layer with inputs X, H, W, R, B
 * to Add, Split, MatMul, Multiply and Subtract ops according to the formula:
                (.) - Denotes element-wise multiplication.
                *   - Denotes dot product.
                ,   - Denotes concat.
                f, g  - are activation functions

                zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
                rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
                ot = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # when linear_before_reset := false # (default)
                ot = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset:= true
                ~ut = 1 - at*zt
                Ht = (1 - zt) (.) ot + zt (.) Ht-1
 * *
 */
ov::intel_cpu::AUGRUCellCompose::AUGRUCellCompose() {
    MATCHER_SCOPE(AUGRUCellCompose);
    auto mul_input1_ptn = pattern::any_input(pattern::rank_equals(2));
    auto mul_input2_ptn = pattern::any_input(pattern::rank_equals(2));

    auto mul_input3_ptn = pattern::any_input(pattern::rank_equals(2));
    auto mul_input4_ptn = pattern::any_input(pattern::rank_equals(2));

    auto mul1_ptn = pattern::wrap_type<Multiply>({mul_input1_ptn, mul_input2_ptn});
    auto mul2_ptn = pattern::wrap_type<Multiply>({mul_input3_ptn, mul_input4_ptn});
    auto H_out_ptn = pattern::wrap_type<Add>({mul1_ptn, mul2_ptn});

    matcher_pass_callback callback = [=](pattern::Matcher &m) {
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        auto H_out = std::dynamic_pointer_cast<Add>(m.get_match_root());
        if (!H_out)
            return false;

        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        const auto& pattern_map = m.get_pattern_value_map();
        auto mul_1 = pattern_map.at(mul1_ptn).get_node_shared_ptr();
        auto mul_2 = pattern_map.at(mul2_ptn).get_node_shared_ptr();
        if (!mul_1 || !mul_2)
            return false;

        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        NodeVector new_ops;

        // try the left branch
        auto Ht_1 = mul_1->input_value(1); //Ht-1 (batch_size, hidden_size)
        std::shared_ptr<Concat> concat_1 = nullptr;
        for (const auto&input : Ht_1.get_target_inputs()) {
            const auto& node = input.get_node()->shared_from_this();
            if (std::dynamic_pointer_cast<Concat>(node)) {
                concat_1 = std::dynamic_pointer_cast<Concat>(node);
                break;
            }
        }
        if (!concat_1) { // try the right branch
            Ht_1 = mul_2->input_value(1); //Ht-1 (batch_size, hidden_size)
            for (const auto&input : Ht_1.get_target_inputs()) {
                const auto& node = input.get_node()->shared_from_this();
                if (std::dynamic_pointer_cast<Concat>(node)) {
                    concat_1 = std::dynamic_pointer_cast<Concat>(node);
                    // swap mul_1, mul_2
                    auto tmp = mul_1;
                    mul_1 = mul_2;
                    mul_2 = tmp;
                    break;
                }
            }
        }

        if (!concat_1) {
            return false;
        }
        
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        auto Xt = concat_1->input_value(0); //Xt (batch_size, input_size)

        //
        auto hat_Ut = std::dynamic_pointer_cast<Multiply>(mul_1->get_input_node_shared_ptr(0));
        if (!hat_Ut)
            return false;

        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        // node substrace is transformed with "multiply + add" instead... why?
        // original: (1 - At) * Ut
        //auto Aminus = std::dynamic_pointer_cast<Subtract>(hat_Ut->get_input_node_shared_ptr(0));
        // if (!Aminus)
        //     return false;
        //auto A = Aminus->input_value(1); // Activation (batch_size, 1)
        // transformed to: ( 1 + (-1)*At ) * Ut
        Output<Node> A;
        const auto Add_ = std::dynamic_pointer_cast<Add>(hat_Ut->get_input_node_shared_ptr(0));
        const auto Aminus = std::dynamic_pointer_cast<Multiply>(Add_->get_input_node_shared_ptr(1));
        if (Aminus) {
            A = Aminus->input_value(0); // Activation (batch_size, 1)
        } else {
            std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
            const auto reshape_ = std::dynamic_pointer_cast<Reshape>(Add_->get_input_node_shared_ptr(1));
            if (!reshape_) return false;
            std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
            const auto A_ = std::dynamic_pointer_cast<Multiply>(reshape_->get_input_node_shared_ptr(0));
            if (!A_) return false;
            std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
            A = std::make_shared<Reshape>(A_->input_value(0), reshape_->input_value(1), false);
        }

        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        // fake W, R, B here  
        const std::size_t hidden_size = 36;
        const std::size_t input_size = hidden_size;
        const auto W = Constant::create(Xt.get_element_type(), Shape{3 * hidden_size, input_size}, {1.0});
        W->set_friendly_name("W");
        const auto R = Constant::create(Xt.get_element_type(), Shape{3 * hidden_size, input_size}, {1.0});
        R->set_friendly_name("R");
        const auto B = Constant::create(Xt.get_element_type(), Shape{3 * hidden_size}, {0.0});
        B->set_friendly_name("B");

        //
        auto augrucell = std::make_shared<ov::intel_cpu::AUGRUCellNode>(Xt, Ht_1, A, W, R, B, hidden_size);    
        new_ops.push_back(augrucell);

        //
        augrucell->set_friendly_name(H_out->get_friendly_name());
        copy_runtime_info({mul_1, mul_2, H_out}, new_ops);  // TODO: how to copy rt_info from a complicate subgraph?
        replace_node(H_out, augrucell);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(H_out_ptn, matcher_name);
    register_matcher(m, callback);
}

ov::intel_cpu::FuseAUGRUCell::FuseAUGRUCell() {
    MATCHER_SCOPE(FuseAUGRUCell);
    auto H_out_ptn = pattern::wrap_type<snippets::op::Subgraph>({pattern::any_input(pattern::rank_equals(2)),
                                                                pattern::any_input(pattern::rank_equals(2)),
                                                                pattern::any_input(pattern::rank_equals(2)),
                                                                pattern::any_input(pattern::rank_equals(2))});
    matcher_pass_callback callback = [=](pattern::Matcher &m) {
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        auto H_out = std::dynamic_pointer_cast<snippets::op::Subgraph>(m.get_match_root());
        if (!H_out)
            return false;

        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        NodeVector new_ops;

        auto A = H_out->input_value(0); // Activation (batch_size, 1)
        auto Ht_1 = H_out->input_value(2); //Ht-1 (batch_size, hidden_size)
        std::shared_ptr<Concat> concat_1 = nullptr;
        for (const auto&input : Ht_1.get_target_inputs()) {
            const auto& node = input.get_node()->shared_from_this();
            if (std::dynamic_pointer_cast<Concat>(node)) {
                concat_1 = std::dynamic_pointer_cast<Concat>(node);
            }
        }
        if (!concat_1)
            return false;
        
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        auto Xt = concat_1->input_value(0); //Xt (batch_size, input_size)

        // fake W, R, B here  
        const std::size_t hidden_size = 36;
        const std::size_t input_size = hidden_size;
        static const auto W = Constant::create(Xt.get_element_type(), Shape{3 * hidden_size, input_size}, {1.0});
        W->set_friendly_name("W");
        static const auto R = Constant::create(Xt.get_element_type(), Shape{3 * hidden_size, input_size}, {1.0});
        R->set_friendly_name("R");
        static const auto B = Constant::create(Xt.get_element_type(), Shape{3 * hidden_size}, {0.0});
        B->set_friendly_name("B");

        //
        auto augrucell = std::make_shared<ov::intel_cpu::AUGRUCellNode>(Xt, Ht_1, A, W, R, B, hidden_size);    
        new_ops.push_back(augrucell);

        //
        augrucell->set_friendly_name(H_out->get_friendly_name());
        copy_runtime_info({concat_1, H_out}, new_ops);  // TODO: how to copy rt_info from a complicate subgraph?
        replace_node(H_out, augrucell);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(H_out_ptn, matcher_name);
    register_matcher(m, callback);
}


ov::intel_cpu::FuseAUGRUCell2Sequence::FuseAUGRUCell2Sequence() {
    MATCHER_SCOPE(FuseAUGRUCell2Sequence);
    // std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
    auto is_supported_augru_cell = [](const std::shared_ptr<Node>& n) {
        // std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        // std::cout << "got: " << n->get_friendly_name() << ", type: " << n->get_type_name() << 
        // ov::is_type<ov::intel_cpu::AUGRUCellNode>(n) << ov::is_type<RNNCell>(n) << n->get_type_info().get_version() <<
        // ("cpu_plugin_opset"==n->get_type_info().get_version()) << ("RNNCell" == std::string(n->get_type_name())) << std::endl;
        return "cpu_plugin_opset"==n->get_type_info().get_version() && "RNNCell" == std::string(n->get_type_name());//pattern::has_class<ov::intel_cpu::AUGRUCellNode>()(n) || pattern::has_class<RNNCell>()(n);
    };
    auto any_augru = std::make_shared<pattern::op::Label>(pattern::any_input(), is_supported_augru_cell);

    matcher_pass_callback callback = [=](pattern::Matcher &m) {
        const auto sequence_len = 100;
        const auto num_directions = 1;
        NodeVector old_ops;
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;

        auto first_augru_cell = std::dynamic_pointer_cast<ov::intel_cpu::AUGRUCellNode>(m.get_match_root());
        if (!first_augru_cell)
            return false;
        old_ops.push_back(first_augru_cell);
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        std::cout << "the first AUGRUCell got: " << first_augru_cell->get_friendly_name() << std::endl;

        // X [batch_size, seq_length, input_size]
        // split -> reshape -> augrucell(in_port:0)
        auto x_reshape = std::dynamic_pointer_cast<Reshape>(first_augru_cell->input_value(0).get_node_shared_ptr());
        if (!x_reshape)
            return false;
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        auto x_split = std::dynamic_pointer_cast<Split>(x_reshape->input_value(0).get_node_shared_ptr());
        if (!x_split)
            return false;
        old_ops.push_back(x_split);
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        if (x_split->get_output_size() != sequence_len) return false;
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        auto X = x_split->input_value(0);

        // initial_hidden_state [batch_size, num_directions, hidden_size]
        // const -> augrucell(in_port:1)
        auto initial_hidden_state = std::make_shared<Unsqueeze>(first_augru_cell->input_value(1), Constant::create(ngraph::element::i32, Shape{1}, {1}));

        // A [batch_size, seq_length, 1]
        // VariadicSplit -> subgraph -> reshape -> augrucell (in_port: 2)
        auto atten_reshape = std::dynamic_pointer_cast<Reshape>(first_augru_cell->input_value(2).get_node_shared_ptr());
        if (!atten_reshape)
            return false;
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        auto atten_sub = std::dynamic_pointer_cast<snippets::op::Subgraph>(atten_reshape->input_value(0).get_node_shared_ptr());
        if (!atten_sub)
            return false;
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        auto atten_split = std::dynamic_pointer_cast<VariadicSplit>(atten_sub->input_value(0).get_node_shared_ptr());
        if (!atten_split)
            return false;
        old_ops.push_back(atten_split);                  
        if (atten_split->get_output_size() != sequence_len) return false;
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        auto A = atten_split->input_value(0);

        // fake W, R, B here  
        const std::size_t hidden_size = 36;
        const std::size_t input_size = hidden_size;
        const auto W = Constant::create(X.get_element_type(), Shape{num_directions, 3 * hidden_size, input_size}, {1.0});
        W->set_friendly_name("W");
        const auto R = Constant::create(X.get_element_type(), Shape{num_directions, 3 * hidden_size, input_size}, {1.0});
        R->set_friendly_name("R");
        const auto B = Constant::create(X.get_element_type(), Shape{num_directions, 3 * hidden_size}, {0.0});
        B->set_friendly_name("B");

        // seq_lengths [batch_size]
        const size_t batch_dim = 0;
        auto batch_dimension = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_source(
            X,
            {batch_dim});
        auto seq_lengths_scalar = Constant::create(ngraph::element::i32, {}, {sequence_len});
        auto seq_lengths = ngraph::op::util::make_try_fold<Broadcast>(seq_lengths_scalar, batch_dimension);         

        const auto augru_sequence = std::make_shared<ov::intel_cpu::AUGRUSequenceNode>(X,
                                                                                initial_hidden_state,
                                                                                A,
                                                                                seq_lengths,
                                                                                W,
                                                                                R,
                                                                                B,
                                                                                first_augru_cell->get_hidden_size(),
                                                                                ngraph::op::RecurrentSequenceDirection::FORWARD,
                                                                                first_augru_cell->get_activations(),
                                                                                first_augru_cell->get_activations_alpha(),
                                                                                first_augru_cell->get_activations_beta(),
                                                                                first_augru_cell->get_clip(),
                                                                                first_augru_cell->get_linear_before_reset());

        // iteratedly grab all cells
        std::function <std::shared_ptr<ov::intel_cpu::AUGRUCellNode> (std::shared_ptr<ov::intel_cpu::AUGRUCellNode>)> get_sequence_end;
        get_sequence_end = [&get_sequence_end](std::shared_ptr<ov::intel_cpu::AUGRUCellNode> augru_c) {
            auto childs = augru_c->output(0).get_target_inputs();            
            for (auto &child : childs) {
                const auto c = child.get_node()->shared_from_this();
                const auto next_cell = std::dynamic_pointer_cast<ov::intel_cpu::AUGRUCellNode>(c);
                if (next_cell)
                    return get_sequence_end(next_cell);
            }
            // no child of augrucellnode type. then itself is the last.
            return augru_c;
        };

        auto last_augru_cell = get_sequence_end(first_augru_cell);
        std::cout << "the last AUGRUCell got: " << last_augru_cell->get_friendly_name() << std::endl;   
        old_ops.push_back(last_augru_cell);

        //        
        auto squeeze_sequence = std::make_shared<Squeeze>(augru_sequence->output(1), Constant::create(ngraph::element::i32, Shape{1}, {1}));
        squeeze_sequence->set_friendly_name(last_augru_cell->get_friendly_name());

        replace_node(last_augru_cell, squeeze_sequence);
        copy_runtime_info(old_ops, squeeze_sequence);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(any_augru, matcher_name);
    register_matcher(m, callback);
}