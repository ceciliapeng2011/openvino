// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

#include "ngraph/op/util/activation_functions.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"

namespace ov {
namespace intel_cpu {

class AUGRUCellNode : public ngraph::op::util::RNNCellBase {
public:
    OPENVINO_OP("RNNCell", "cpu_plugin_opset");

    AUGRUCellNode() = default;

    AUGRUCellNode(const Output<Node>& X,
            const Output<Node>& initial_hidden_state,
            const Output<Node>& A,
            const Output<Node>& W,
            const Output<Node>& R,
            const Output<Node>& B,            
            std::size_t hidden_size,
            const std::vector<std::string>& activations = std::vector<std::string>{"sigmoid", "tanh"},
            const std::vector<float>& activations_alpha = {},
            const std::vector<float>& activations_beta = {},
            float clip = 0.f,
            bool linear_before_reset = false);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    bool get_linear_before_reset() const { return m_linear_before_reset; }
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    bool m_linear_before_reset;
};

}   // namespace intel_cpu
}   // namespace ov
