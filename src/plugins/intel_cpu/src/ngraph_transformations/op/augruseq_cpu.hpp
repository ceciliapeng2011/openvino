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

class AUGRUSequenceNode : public ngraph::op::util::RNNCellBase {
public:
    OPENVINO_OP("RNNSequence", "cpu_plugin_opset");

    AUGRUSequenceNode() = default;

    AUGRUSequenceNode(const Output<Node>& X,
                const Output<Node>& H_t,
                const Output<Node>& A,
                const Output<Node>& sequence_lengths,
                const Output<Node>& W,
                const Output<Node>& R,
                const Output<Node>& B,
                size_t hidden_size,
                op::RecurrentSequenceDirection direction,
                const std::vector<std::string>& activations = std::vector<std::string>{"sigmoid", "tanh"},
                const std::vector<float>& activations_alpha = {},
                const std::vector<float>& activations_beta = {},
                float clip = 0.f,
                bool linear_before_reset = false);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    bool get_linear_before_reset() const { return m_linear_before_reset; }
    op::RecurrentSequenceDirection get_direction() const {
        return m_direction;
    }
private:
    op::RecurrentSequenceDirection m_direction;
    bool m_linear_before_reset;
};

}   // namespace intel_cpu
}   // namespace ov
