// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
class StatefulTransposeSDPAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StatefulTransposeSDPAFusion", "0");
    StatefulTransposeSDPAFusion();
};

class SDPATransposeReshapeFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SDPATransposeReshapeFusion", "0");
    SDPATransposeReshapeFusion();
};

}   // namespace intel_cpu
}   // namespace ov