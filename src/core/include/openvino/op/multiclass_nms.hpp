// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/multiclass_nms_base.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief MulticlassNms operation
///
class OPENVINO_API MulticlassNms : public util::MulticlassNmsBase {
public:
    OPENVINO_OP("MulticlassNms", "opset8", op::util::MulticlassNmsBase);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Constructs a conversion operation.
    MulticlassNms() = default;

    /// \brief Constructs a MulticlassNms operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param attrs Attributes of the operation
    MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v8

namespace v9 {
/// \brief MulticlassNms operation
///
class OPENVINO_API MulticlassNms : public util::MulticlassNmsBase {
public:
    OPENVINO_OP("MulticlassNms", "opset9", op::util::MulticlassNmsBase);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Constructs a conversion operation.
    MulticlassNms() = default;

    /// \brief Constructs a MulticlassNms operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param attrs Attributes of the operation
    MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs);

    /// \brief Constructs a MulticlassNms operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param roisnum Node producing the number of rois
    /// \param attrs Attributes of the operation
    MulticlassNms(const Output<Node>& boxes,
                  const Output<Node>& scores,
                  const Output<Node>& roisnum,
                  const Attributes& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

protected:
    bool validate() override;
    /// \brief infer shape and type
    ///
    /// \param static_output Indicate to produce an upper bound for the number of possible selected boxes.
    /// Hence it produce the static output shapes.
    /// \param ignore_bg_class Indicate to remove the background class when produce the upper bound shapes.
    void infer_shape_types(const bool output_static = false, const bool ignore_bg_class = false);
};
}  // namespace v9
}  // namespace op
}  // namespace ov
