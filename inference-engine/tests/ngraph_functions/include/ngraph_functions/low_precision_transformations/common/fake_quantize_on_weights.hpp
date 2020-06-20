// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizeOnWeights: public FakeQuantizeOnData {
public:
    FakeQuantizeOnWeights();

    FakeQuantizeOnWeights(
        const size_t quantizationLevel,
        const ngraph::Shape& constantShape,
        const std::vector<float>& lowValues,
        const std::vector<float>& highValues);

    virtual ~FakeQuantizeOnWeights();

    bool empty() const override;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeOnWeights& data) {
    return out << "_" << data.constantShape << "_" << data.lowValues << "_" << data.highValues;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
