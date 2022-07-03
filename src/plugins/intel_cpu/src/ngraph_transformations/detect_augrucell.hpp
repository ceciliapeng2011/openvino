// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class AUGRUCellCompose : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("AUGRUCellCompose", "0");
    AUGRUCellCompose();
};

class FuseAUGRUCell : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseAUGRUCell", "0");
    FuseAUGRUCell();
};

class FuseAUGRUCell2Sequence : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseAUGRUCell2Sequence", "0");
    FuseAUGRUCell2Sequence();
};
}   // namespace intel_cpu
}   // namespace ov
