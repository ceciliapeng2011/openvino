// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/insert_convert_on_inputs.hpp"
#include "snippets/snippets_isa.hpp"

#include "ngraph/type.hpp"
#include "ngraph/node.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

// We should recursivelly (after full sequences of ConvertTruncation) go through inputs and
// insert ConvertSaturation with supported element type before eltwises
// NOTE: JUST EXAMPLE:
//                             Parameter I8
//                        ConvertTruncation U8
//                  /              |               \
// ConvertTruncation F32  ConvertTruncation I32  ConvertTruncation BF16
//      Eltwise           ConvertSaturation FP32 ConvertTruncation I32
//        <>                    Eltwise          ConvertSaturation FP32
//                                 <>                    Eltwise
bool insertConvertSaturationAfterNode(const std::shared_ptr<ov::Node>& node, const ov::element::Type element_type) {
    bool rewritten = false;
    for (const auto& output : node->outputs()) {
        for (auto consumer : output.get_target_inputs()) {
            const auto output_shared_node = consumer.get_node()->shared_from_this();
            // Go down through ConvertTruncation sequence
            if (auto existing_convert_t = ov::as_type_ptr<ngraph::snippets::op::ConvertTruncation>(output_shared_node)) {
                rewritten = insertConvertSaturationAfterNode(existing_convert_t, element_type);
                continue;
            }

            // Check if ConvertSaturation already exists with supported element type or not and insert ConvertSaturation with supported element type
            auto existing_convert_s = ov::as_type_ptr<ngraph::snippets::op::ConvertSaturation>(output_shared_node);
            if ((!existing_convert_s && !ov::is_type<ov::op::v0::Result>(output_shared_node) && consumer.get_element_type() != element_type) ||
                (existing_convert_s && existing_convert_s->get_destination_type() != element_type)) {
                const auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(node, element_type);
                consumer.replace_source_output(convert);
                rewritten |= true;
            }
        }
    }
    return rewritten;
}

ngraph::snippets::pass::InsertConvertOnInputs::InsertConvertOnInputs(const ov::element::Type exec_type) {
    MATCHER_SCOPE(InsertConvertOnInputs);

    auto param_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Parameter>();
    auto scalar_pattern = pattern::wrap_type<opset1::Constant>(
        [=](Output<Node> output) -> bool { return ngraph::shape_size(output.get_shape()) == 1; });
    auto input = std::make_shared<pattern::op::Or>(OutputVector{ param_pattern, scalar_pattern });

    ngraph::matcher_pass_callback callback = [this, exec_type](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertConvertOnInputs")
        auto root = m.get_match_root();

        auto rewritten = insertConvertSaturationAfterNode(root, exec_type);

        return rewritten;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(input, matcher_name);
    register_matcher(m, callback);
}
