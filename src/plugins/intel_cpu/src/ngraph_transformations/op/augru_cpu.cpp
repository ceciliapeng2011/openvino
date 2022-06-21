// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "augru_cpu.hpp"
#include "../itt.hpp"

using namespace std;

ov::intel_cpu::AUGRUCellNode::AUGRUCellNode(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& A,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         const Output<Node>& B,
                         size_t hidden_size,
                         const vector<string>& activations,
                         const vector<float>& activations_alpha,
                         const vector<float>& activations_beta,
                         float clip,
                         bool linear_before_reset)
    : RNNCellBase({X, initial_hidden_state, A, W, R, B}, hidden_size, clip, activations, activations_alpha, activations_beta),
      m_linear_before_reset(linear_before_reset) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::AUGRUCellNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(AUGRUCellNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ov::intel_cpu::AUGRUCellNode>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4),new_args.at(5),
                                      m_hidden_size, m_activations, m_activations_alpha, m_activations_beta, m_clip,
                                      m_linear_before_reset);
}

void ov::intel_cpu::AUGRUCellNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(AUGRUCellNode_validate_and_infer_types);
    element::Type arg_type = get_input_element_type(0);

    PartialShape output_shape{PartialShape::dynamic(2)};

    if (get_input_partial_shape(0).is_static()) {
        int64_t batch_size = get_input_partial_shape(0).get_shape()[0];
        output_shape = {batch_size, m_hidden_size};
    }

    set_output_type(0, arg_type, output_shape);
}

bool ov::intel_cpu::AUGRUCellNode::visit_attributes(ngraph::AttributeVisitor &visitor) {
    INTERNAL_OP_SCOPE(AUGRUCellNode_visit_attributes);
    visitor.on_attribute("linear_before_reset", m_linear_before_reset);
    return true;
}
