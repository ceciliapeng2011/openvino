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
#include "rnn.hpp"
#include "lstm.hpp"

namespace ngraph {
    namespace frontend {
        namespace pdpd {
            namespace op {

                OutputVector rnn (const NodeContext& node) {
                    auto mode = node.get_attribute<std::string>("mode");
                    MY_ASSERT(mode == "LSTM", "RNN only support LSTM now");
                    return lstm(node);
                }

            }
        }
    }
}
