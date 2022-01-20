// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
class TensorArrayWrite : public Op {
public:
    OPENVINO_OP("TensorArrayWrite", "internal");
    BWDCMP_RTTI_DECLARATION;

    TensorArrayWrite() = default;

    TensorArrayWrite(const Output<Node>& input, const Output<Node>& index, const std::string& output_name);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    std::string m_output_name;

private:
};

}  // namespace internal
}  // namespace op
}  // namespace ov
