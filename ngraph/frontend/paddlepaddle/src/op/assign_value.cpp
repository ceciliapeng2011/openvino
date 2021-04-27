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
#include "assign_value.hpp"
namespace ngraph {
    namespace frontend {
        namespace pdpd {
            namespace op {

                NamedOutputs assign_value (const NodeContext& node) {

                    std::vector<int32_t> shape = node.get_attribute<std::vector<int32_t>>("shape");
                    auto dtype = node.get_attribute<ngraph::element::Type>("dtype");
                    std::shared_ptr<Node> const_node;
                    switch (dtype) {
                        case element::i32:
                        {
                            auto values = node.get_attribute<std::vector<int32_t>>("int32_values");
                            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
                            break;
                        }
                        case element::f32:
                        {
                            std::vector<float> values = node.get_attribute<std::vector<float>>("fp32_values");
                            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
                            break;
                        }
                        case element::boolean:
                        {
                            auto values = node.get_attribute<std::vector<int32_t>>("bool_values");
                            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
                            break;
                        }
                        default:
                        {
                            auto values = node.get_attribute<std::vector<int64_t>>("int64_values");
                            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
                            break;
                        }
                    }

                    return node.default_single_output_mapping({const_node}, {"Out"});
                }

            }
        }
    }
}
