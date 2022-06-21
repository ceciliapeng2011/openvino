// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detect_augrucell.hpp"
#include "op/augru_cpu.hpp"
#include <numeric>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

#include "itt.hpp"

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

    std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;

    auto mul_input1_ptn = pattern::any_input(pattern::rank_equals(2));
    auto mul_input2_ptn = pattern::any_input(pattern::rank_equals(2));

    auto mul_input3_ptn = pattern::any_input(pattern::rank_equals(2));
    auto mul_input4_ptn = pattern::any_input(pattern::rank_equals(2));

    auto mul1_ptn = pattern::wrap_type<Multiply>({mul_input1_ptn, mul_input2_ptn});
    auto mul2_ptn = pattern::wrap_type<Multiply>({mul_input3_ptn, mul_input4_ptn});
    auto H_out_ptn = pattern::wrap_type<Add>({mul1_ptn, mul2_ptn});

    // auto Xt = pattern::any_input(pattern::rank_equals(2));
    // auto Ht = pattern::any_input(pattern::rank_equals(2));
    // auto HX_concat = pattern::wrap_type<Concat>({Xt, Ht});

    // // Xt*(W^T),Ht-1*(R^T) where W equals R for z and r.
    // auto matmul_1 = pattern::wrap_type<MatMul>({HX_concat, pattern::any_input(pattern::has_static_shape())});
    // auto add_1 = pattern::wrap_type<Add>({matmul_1, pattern::any_input(pattern::has_static_shape())});
    // auto sigmoid = pattern::wrap_type<Sigmoid>({add_1});

    // // split output0: rt, output1: zt
    // auto num_splits = pattern::wrap_type<Constant>();
    // auto split = pattern::wrap_type<Split>({sigmoid, num_splits});

    // // Xt*(W^T),Ht-1*(R^T) where W equals R for h.
    // auto mul_4 = pattern::wrap_type<Multiply>({split/*->output(0)*/, Ht}); // rt (.) Ht-1
    // auto X_rt_concat = pattern::wrap_type<Concat>({Xt, mul_4});
    // auto matmul_2 = pattern::wrap_type<MatMul>({X_rt_concat, pattern::any_input(pattern::has_static_shape())});
    // auto add_2 = pattern::wrap_type<Add>({matmul_2, pattern::any_input(pattern::has_static_shape())});
    // auto tanh = pattern::wrap_type<Tanh>({add_2});  //ot

    // auto At = pattern::any_input(pattern::rank_equals(2)); // attention
    // auto sub_at = pattern::wrap_type<Subtract>({pattern::wrap_type<Constant>(), At});
    // auto hat_ut = pattern::wrap_type<Multiply>({pattern::any_input(), split/*->output(1)*/}); //~ut

    // auto mul_6 = pattern::wrap_type<Multiply>({hat_ut, Ht}); //~ut*Ht-1
    // auto sub_hat_ut = pattern::wrap_type<Subtract>({pattern::wrap_type<Constant>(), hat_ut});
    // auto mul_7 = pattern::wrap_type<Multiply>({sub_hat_ut, tanh});

    // auto H_out = pattern::wrap_type<Add>({mul_6, mul_7});

    std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;

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

        auto Ht_1 = mul_1->input_value(1); //Ht-1 (batch_size, hidden_size)
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

        //
        auto hat_Ut = std::dynamic_pointer_cast<Multiply>(mul_1->get_input_node_shared_ptr(0));
        if (!hat_Ut)
            return false;

        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        // node substrace is transformed with "multiply + add" instead... why?
        //auto Aminus = std::dynamic_pointer_cast<Subtract>(hat_Ut->get_input_node_shared_ptr(0));
        // if (!Aminus)
        //     return false;
        //auto A = Aminus->input_value(1); // Activation (batch_size, 1)
        const auto Add_ = std::dynamic_pointer_cast<Add>(hat_Ut->get_input_node_shared_ptr(0));
        const auto Aminus = std::dynamic_pointer_cast<Multiply>(Add_->get_input_node_shared_ptr(1));
        if (!Aminus)
            return false;
        auto A = Aminus->input_value(0); // Activation (batch_size, 1)

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
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        auto augrucell = std::make_shared<ov::intel_cpu::AUGRUCellNode>(Xt, Ht_1, A, W, R, B, hidden_size);    
        new_ops.push_back(augrucell);

        //
        augrucell->set_friendly_name(H_out->get_friendly_name());
        copy_runtime_info({mul_1, mul_2, H_out}, new_ops);  // TODO: how to copy rt_info from a complicate subgraph?
        replace_node(H_out, augrucell);
        std::cout << "################" <<  __FILE__ << ": " << __LINE__ << std::endl;
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(H_out_ptn, matcher_name);
    register_matcher(m, callback);
}
