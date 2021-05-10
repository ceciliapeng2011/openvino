// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "range.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs range (const NodeContext& node) {
    auto start = node.get_ng_input("Start");
    auto stop = node.get_ng_input("End");
    auto step = node.get_ng_input("Step");

    bool keep_dims = false;
    const auto axis = ngraph::opset6::Constant::create(element::u64, Shape{}, {0});
    auto start_scalar = std::make_shared<ngraph::opset6::ReduceMin>(start, axis, keep_dims);
    auto stop_scalar = std::make_shared<ngraph::opset6::ReduceMin>(stop, axis, keep_dims);
    auto step_scalar = std::make_shared<ngraph::opset6::ReduceMin>(step, axis, keep_dims);

    //TODO to support other data types other than FP32
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Range>(start_scalar, stop_scalar, step_scalar, element::f32)}, {"Out"});
}

}}}}