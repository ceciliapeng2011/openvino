// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/convolution_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ConvolutionFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData,
    const FakeQuantizeOnWeights& fqOnWeights) {
    const float k = 50.f;

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto fakeQuantizeOnActivations = fqOnData.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input, precision, fqOnData.quantizationLevel, fqOnData.constantShape,
            fqOnData.lowValues, fqOnData.highValues, fqOnData.lowValues, fqOnData.highValues);

    const auto weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
        std::vector<float>(inputShape[1] * inputShape[1], 1));

    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        fqOnData.empty() ? input : fakeQuantizeOnActivations,
        fqOnWeights.empty() ? weights->output(0) :
        ngraph::builder::makeFakeQuantize(
            weights, precision, fqOnWeights.quantizationLevel, fqOnWeights.constantShape,
            fqOnWeights.lowValues, fqOnWeights.highValues, fqOnWeights.lowValues, fqOnWeights.highValues),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const FakeQuantizeOnWeights& fakeQuantizeOnWeights) {
    return nullptr;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
