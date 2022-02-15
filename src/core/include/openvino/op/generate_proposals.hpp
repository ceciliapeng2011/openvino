// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v6 {
/// \brief An operation GenerateProposalsSingleImage
/// computes ROIs and their scores based on input data.
class OPENVINO_API GenerateProposalsSingleImage : public Op {
public:
    OPENVINO_OP("GenerateProposalsSingleImage", "opset6", op::Op, 6);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Structure that specifies attributes of the operation
    struct Attributes {
        // minimum box width & height
        float min_size;
        // specifies NMS threshold
        float nms_threshold;
        // number of top-n proposals after NMS
        int64_t post_nms_count;
        // number of top-n proposals before NMS
        int64_t pre_nms_count;
    };

    GenerateProposalsSingleImage() = default;
    /// \brief Constructs a GenerateProposalsSingleImage operation.
    ///
    /// \param im_info Input image info
    /// \param anchors Input anchors
    /// \param deltas Input deltas
    /// \param scores Input scores
    /// \param attrs Operation attributes
    GenerateProposalsSingleImage(const Output<Node>& im_info,
                                                      const Output<Node>& anchors,
                                                      const Output<Node>& deltas,
                                                      const Output<Node>& scores,
                                                      const Attributes& attrs);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attrs() const {
        return m_attrs;
    }

private:
    Attributes m_attrs;
};
}  // namespace v6

namespace v9 {
/// \brief An operation GenerateProposalsSingleImage
/// computes ROIs and their scores based on input data.
class OPENVINO_API GenerateProposalsSingleImage : public Op {
public:
    OPENVINO_OP("GenerateProposalsSingleImage", "opset9", op::Op, 9);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Structure that specifies attributes of the operation
    struct Attributes {
        // minimum box width & height
        float min_size;
        // specifies NMS threshold
        float nms_threshold;
        // number of top-n proposals after NMS
        int64_t post_nms_count;
        // number of top-n proposals before NMS
        int64_t pre_nms_count;
        // specify whether the bbox is normalized or not.
        // For example if *normalized* is true, width = x_right - x_left
        // If *normalized* is false, width = x_right - x_left + 1.
        bool normalized = true;
        // specify eta parameter for adaptive NMS in generate proposals
        float nms_eta = 1.0;
    };

    GenerateProposalsSingleImage() = default;
    /// \brief Constructs a GenerateProposalsSingleImage operation.
    ///
    /// \param im_info Input image info
    /// \param anchors Input anchors
    /// \param deltas Input deltas
    /// \param scores Input scores
    /// \param attrs Operation attributes
    GenerateProposalsSingleImage(const Output<Node>& im_info,
                                                      const Output<Node>& anchors,
                                                      const Output<Node>& deltas,
                                                      const Output<Node>& scores,
                                                      const Attributes& attrs);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attrs() const {
        return m_attrs;
    }

private:
    Attributes m_attrs;
};
}  // namespace v9
}  // namespace op
}  // namespace ov
