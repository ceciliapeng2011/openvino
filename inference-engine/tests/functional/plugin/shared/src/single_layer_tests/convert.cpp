// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "single_layer_tests/convert.hpp"

namespace LayerTestsDefinitions {

std::string ConvertLayerTest::getTestCaseName(const testing::TestParamInfo<ConvertParamsTuple> &obj) {
    InferenceEngine::Precision inputPrecision, targetPrecision;
    std::string targetName;
    std::vector<std::vector<size_t>> inputShape;
    std::tie(inputShape, inputPrecision, targetPrecision, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "targetPRC=" << targetPrecision.name() << "_";
    result << "inputPRC=" << inputPrecision.name() << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

void ConvertLayerTest::SetUp() {
    InferenceEngine::Precision inputPrecision, targetPrecision;
    std::vector<std::vector<size_t>> inputShape;
    std::tie(inputShape, inputPrecision, targetPrecision, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto targetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(targetPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, inputShape);
    auto convert = std::make_shared<ngraph::opset3::Convert>(params.front(), targetPrc);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(convert)};
    function = std::make_shared<ngraph::Function>(results, params, "Convert");
}

TEST_P(ConvertLayerTest, CompareWithRefs) {
    Run();
};
}  // namespace LayerTestsDefinitions