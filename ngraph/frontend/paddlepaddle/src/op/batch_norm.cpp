//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <ngraph/opsets/opset6.hpp>
#include "batch_norm.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs batch_norm (const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto gamma = node.get_ng_input("Scale");
    auto beta = node.get_ng_input("Bias");
    auto mean = node.get_ng_input("Mean");
    auto variance = node.get_ng_input("Variance");
    auto data_layout = node.get_attribute<std::string>("data_layout");
    PDPD_ASSERT((data_layout == "NCHW" || data_layout == "NHWC"), "Not supported input data layout!");
    if(data_layout == "NCHW") {
        return node.default_single_output_mapping({std::make_shared<ngraph::opset6::BatchNormInference>(
                data, gamma, beta, mean, variance, node.get_attribute<float>("epsilon"))}, {"Y"});
    } else if(data_layout == "NHWC") {
        auto rank = static_cast<unsigned long>(data.get_partial_shape().rank().get_length());
        auto input_order = ngraph::opset6::Constant::create(ngraph::element::i64, {4}, {0, 3, 1, 2});
        auto data_nhwc = std::make_shared<ngraph::opset6::Transpose>(data, input_order);
        auto node_batch_norm = std::make_shared<ngraph::opset6::BatchNormInference>(
                data_nhwc, gamma, beta, mean, variance, node.get_attribute<float>("epsilon"));
        auto output_order = ngraph::opset6::Constant::create(ngraph::element::i64, {4}, {0, 1, 2, 3});
        return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Transpose>(node_batch_norm, output_order)}, {"Y"});
    }
}

}}}}