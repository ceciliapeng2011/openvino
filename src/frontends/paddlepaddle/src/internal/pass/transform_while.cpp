// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "paddlepaddle_frontend/exceptions.hpp"

#include "internal/op/conditional_block.hpp"
#include "internal/op/while.hpp"

#include "internal/pass/transform_while.hpp"

#include "../../default_opset.hpp"
#include "openvino/pass/constant_folding.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace opset8;

ov::frontend::pdpd::pass::TransformWhile::TransformWhile(std::vector<std::shared_ptr<Function>> functions) {
    auto while_label = ngraph::pattern::wrap_type<ov::op::internal::While>();

    matcher_pass_callback callback = [functions](pattern::Matcher& m) -> bool {
        const auto& while_node = std::dynamic_pointer_cast<ov::op::internal::While>(m.get_match_root());
        if (!while_node)
            return false;
        const auto& inputs = while_node->input_values();
        const auto trip_count = Constant::create(element::i64, {1}, {-1});
        const auto& cond = inputs.back();
        const auto cond_name = cond.get_node_shared_ptr()->get_friendly_name();
        auto loop = std::make_shared<Loop>(trip_count, cond);
        auto sub_model = functions[while_node->m_sub_block];
        loop->set_function(sub_model);

        const auto& parameters = sub_model->get_parameters();
        for (size_t i = 0; i < parameters.size(); i++) {
            const auto& param_name = inputs[i].get_node()->get_friendly_name();
            loop->set_merged_input(parameters[i], inputs[i], sub_model->output(param_name));
        }
        int64_t idx = -1;
        for (size_t i = 0; i < sub_model->get_results().size(); i++) {
            if (sub_model->output(i).get_tensor().get_any_name() == cond_name)
                idx = static_cast<int64_t>(i);
        }
        FRONT_END_GENERAL_CHECK(idx != -1, "could not find condition node in outputs.");

        loop->set_special_body_ports(Loop::SpecialBodyPorts{-1, idx});

        // replace output
        const auto& results = sub_model->get_results();
        OutputVector outputs(results.size());
        for (size_t i = 0; i < results.size(); i++) {
            auto out = loop->get_iter_value(results[i], -1);
            while_node->output(i).replace(out);
        }

        loop->add_node_control_dependents(while_node);
        loop->add_node_control_dependencies(while_node);
        while_node->clear_control_dependents();

        loop->set_friendly_name(loop->get_friendly_name());
        copy_runtime_info(while_node, loop);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(while_label, "while_loop");
    this->register_matcher(m, callback);
}